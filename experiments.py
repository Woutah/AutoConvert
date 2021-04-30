import argparse
import logging
import os
import pickle
from math import ceil

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn
from numpy.random import RandomState
from sklearn.preprocessing import StandardScaler

import utility
from autovc.model_vc import Generator
from config import Config
from data_converter import Converter
from parallel_wavegan.utils import read_hdf5

from autovc.synthesis import build_model_melgan, melgan


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    print(torch.cuda.get_device_name(0))

converter = Converter(device)

import yaml

from parallel_wavegan.utils import download_pretrained_model, load_model


def logmelfilterbank(audio,
                     sampling_rate,
                     fft_size=1024,
                     hop_size=256,
                     win_length=None,
                     window="hann",
                     num_mels=80,
                     fmin=None,
                     fmax=None,
                     eps=1e-10,
                     ):
    """Compute log-Mel filterbank feature.

    Args:
        audio (ndarray): Audio signal (T,).
        sampling_rate (int): Sampling rate.
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length. If set to None, it will be the same as fft_size.
        window (str): Window function type.
        num_mels (int): Number of mel basis.
        fmin (int): Minimum frequency in mel basis calculation.
        fmax (int): Maximum frequency in mel basis calculation.
        eps (float): Epsilon value to avoid inf in log calculation.

    Returns:
        ndarray: Log Mel filterbank feature (#frames, num_mels).

    """
    # get amplitude spectrogram
    x_stft = librosa.stft(audio, n_fft=fft_size, hop_length=hop_size,
                          win_length=win_length, window=window, pad_mode="reflect")
    spc = np.abs(x_stft).T  # (#frames, #bins)

    # get mel basis
    fmin = 0 if fmin is None else fmin
    fmax = sampling_rate / 2 if fmax is None else fmax
    mel_basis = librosa.filters.mel(sampling_rate, fft_size, num_mels, fmin, fmax)

    return np.log10(np.maximum(eps, np.dot(spc, mel_basis.T)))

    

download_pretrained_model("vctk_multi_band_melgan.v2", "melgan") #download model
vocoder_conf = "melgan/vctk_multi_band_melgan.v2/config.yml"
with open(vocoder_conf) as f:
    config = yaml.load(f, Loader=yaml.Loader)


#================================================Loading/preprocessing=========================================================
# audio, sr = sf.read(utility.get_full_path(".\\input\\p225\\p225_001.wav"))
audio, sr = sf.read(utility.get_full_path(".\\input\\Wouter\\6.wav"))
# trim silence
if config["trim_silence"]:
    audio, _ = librosa.effects.trim(audio,
                                    top_db=config["trim_threshold_in_db"],
                                    frame_length=config["trim_frame_size"],
                                    hop_length=config["trim_hop_size"])

if "sampling_rate_for_feats" not in config:
    x = audio
    # sampling_rate = config["sampling_rate"]
    sampling_rate=sr
    hop_size = config["hop_size"]
else:
    # NOTE(kan-bayashi): this procedure enables to train the model with different
    #   sampling rate for feature and audio, e.g., training with mel extracted
    #   using 16 kHz audio and 24 kHz audio as a target waveform
    x = librosa.resample(audio, sampling_rate, config["sampling_rate_for_feats"])
    # sampling_rate = config["sampling_rate_for_feats"]
    assert config["hop_size"] * config["sampling_rate_for_feats"] % fs == 0, \
        "hop_size must be int value. please check sampling_rate_for_feats is correct."
    hop_size = config["hop_size"] * config["sampling_rate_for_feats"] // fs

# extract feature
mel = logmelfilterbank(x,
                        sampling_rate=sampling_rate,
                        hop_size=hop_size,
                        fft_size=config["fft_size"],
                        win_length=config["win_length"],
                        window=config["window"],
                        num_mels=config["num_mels"],
                        fmin=config["fmin"],
                        fmax=config["fmax"])

                    
# make sure the audio length and feature length are matched
audio = np.pad(audio, (0, config["fft_size"]), mode="reflect")
audio = audio[:len(mel) * config["hop_size"]]
assert len(mel) * config["hop_size"] == len(audio)

# apply global gain
if config["global_gain_scale"] > 0.0:
    audio *= config["global_gain_scale"]
if np.abs(audio).max() >= 1.0:
    logging.warn(f"Loaded audio file causes clipping. "
                    f"it is better to re-consider global gain scale. Now exiting.")
    exit(0)

#================================================Normalization=========================================================
# restore scaler
scaler = StandardScaler()
if config["format"] == "hdf5": 
    scaler.mean_ = read_hdf5("./parallel_wavegan/stats.h5", "mean")
    scaler.scale_ = read_hdf5("./parallel_wavegan/stats.h5", "scale")
# elif config["format"] == "npy":
    # scaler.mean_ = np.load(args.stats)[0]
    # scaler.scale_ = np.load(args.stats)[1]
else:
    raise ValueError("support only hdf5 (and normally npy - but not now) format.")
# from version 0.23.0, this information is needed
scaler.n_features_in_ = scaler.mean_.shape[0]
mel = scaler.transform(mel)

# plt.imshow(mel)
# plt.show()

#==============================================Put it through network==================================================
# converter.output_to_wav([[mel]])
print(f"Now loading in pretrained melGAN model")
download_pretrained_model("vctk_multi_band_melgan.v2", "melgan")
model = load_model("melgan/vctk_multi_band_melgan.v2/checkpoint-1000000steps.pkl")
model.remove_weight_norm()
model = model.eval().to(device)


result = model.inference(torch.tensor(mel, dtype=torch.float).to(device)).view(-1)
# from playsound import playsound

# import pyaudio
# p = pyaudio.PyAudio()

# stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
#                 channels=wf.getnchannels(),
#                 rate=wf.getframerate(),
#                 output=True)
x, sr = sf.read(utility.get_full_path(".\\input\\p225\\p225_001.wav"))
spec_225 = converter._wav_to_spec(x, sr,  utility.get_full_path(".\\input\\p225\\p225_001.wav"))
# ax[1].imshow(np.swapaxes(spec_225, 0, 1))
# ax[1].set_title("225")

model = build_model_melgan().to("cuda")

out = melgan(model, "cuda", spec_225)

sf.write("output/test.wav", out, 24000)


wavedata = result.detach().numpy()
# data = wf.readframes(CHUNK)
stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=24000,
                output=True,
                # output_device_index=0
                )
# i = 0
# chunksize = 8096 * 3
# data = [1]
# while len(data) != 0:
#     slice = (len(wavedata) - chunksize * i, len(wavedata) - min(chunksize * (i+1), len(wavedata)))
#     data = wavedata[slice[0]:slice[1]]
#     print(f"writing {slice}, of len {len(data)}")
#     stream.write(data)
#     i+=1
    # data = wav.readframes(CHUNK)
    
stream.write(wavedata)
# import time
# time.sleep(2)
# stream.flush()
stream.stop_stream()
stream.close()

p.terminate()
# player.set_media(result)
# player.play()


sf.write( "./output/experiment_melgan.wav",
            result.detach().numpy(), config["sampling_rate"], "PCM_16")
print(f"DONE!")