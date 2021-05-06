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

    

download_pretrained_model("vctk_multi_band_melgan.v2", "./vocoders/melgan") #download model
model = load_model("./vocoders/melgan/vctk_multi_band_melgan.v2/checkpoint-1000000steps.pkl")
model.remove_weight_norm()
model = model.eval().to(device)

vocoder_conf = "melgan/vctk_multi_band_melgan.v2/config.yml"
with open(vocoder_conf) as f:
    config = yaml.load(f, Loader=yaml.Loader)

#=========================Get autovc spectrogram==========================
# # directories
# input_dir = Config.dir_paths["input"]
# converted_data_dir = Config.dir_paths["metadata"]
# output_file_dir = Config.dir_paths["output"]
# metadata_name = Config.metadata_name


# audio, sr = sf.read(utility.get_full_path(".\\input\\Wouter\\6.wav"))

# speaker_emb = np.load(os.path.join("./output", "p226_emb.npy")) #load a speaker embedding


# source_speaker = "p225"
# source_list =  ["p225_024"]
# target_speaker = "Wouter"
# target_list = ["1", "2", "3", "4", "5", "6", "7"]



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# spec_dir = Config.dir_paths["spectrograms"]

# spec = converter._wav_to_spec(audio, )
name, spec = np.load(".\\output\\spects_p225xp226_sources_p225_024.pkl", allow_pickle=True)[0] #0 is name

# plt.imshow(spec)
# plt.show()
sr = 16000
n_fft = 1024
win_length = 1024
hop_length = 256
n_mels = 80
fmin = 90
fmax = 7600
ref_level_db = 16
min_level_db = -100
def _db_to_amp(x):
    return np.power(10.0, x * 0.05)

spec = (np.clip(spec, 0, 1) * -min_level_db) + min_level_db
spec = _db_to_amp(spec + ref_level_db)
#================================================Normalization=========================================================
# restore scaler
scaler = StandardScaler()
if config["format"] == "hdf5": 
    scaler.mean_ = read_hdf5(".\\vocoders\\melgan\\stats.h5", "mean")
    scaler.scale_ = read_hdf5(".\\vocoders\\melgan\\stats.h5", "scale")
# elif config["format"] == "npy":
    # scaler.mean_ = np.load(args.stats)[0]
    # scaler.scale_ = np.load(args.stats)[1]
else:
    raise ValueError("support only hdf5 (and normally npy - but not now) format.")
# from version 0.23.0, this information is needed
scaler.n_features_in_ = scaler.mean_.shape[0]
# spec = scaler.transform(spec)

# plt.imshow(mel)
# plt.show()

#==============================================Put audio through autovc generator==================================================
# converter.output_to_wav([[mel]])


result = model.inference(torch.tensor(spec, dtype=torch.float).to(device)).view(-1)
# from playsound import playsound

import pyaudio
p = pyaudio.PyAudio()

# stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
#                 channels=wf.getnchannels(),
#                 rate=wf.getframerate(),
#                 output=True)

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
    
stream.write(np.concatenate([wavedata, wavedata, wavedata, wavedata]))
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