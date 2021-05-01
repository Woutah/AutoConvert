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

import logging
log = logging.getLogger(__name__) #
import yaml

from parallel_wavegan.utils import download_pretrained_model, load_model
from data_converter import Converter



def MelganConverter(Converter):
    def __init__(self, device, melgan_config_path):

        self.super()
        print("initializing melganconverter")
        # self._device = device
        # log.info(f"Using device {self._device}")
        # if not melgan_config:

        
        vocoder_conf = "melgan/vctk_multi_band_melgan.v2/config.yml"
        try:
            with open(vocoder_conf) as f:
                config = yaml.load(f, Loader=yaml.Loader)
        except:
            log.error(f"Unable to laod in yaml config at path: {melgan_config_path}, unknown desired sampling rate... exiting...")

        download_pretrained_model("vctk_multi_band_melgan.v2", "melgan") #download model
        self.melgan_model = None


    @staticmethod
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

    def _wav_to_spec(self, wav, sample_rate, wav_path = None, introduce_noise=False):
        """Convert wav file to a mel spectrogram

        Args:
            wav (numpy array): audio data either 1-d (mono) or 2-d (stereo)
            sample_rate (int): the sampling rate of the .wav (sf.read[1])
            wav_path (str): Path to original wav file
            note that these two variables can be loaded using: 
                wavfile, sample_rate = sf.read(os.path.join(input_dir, speaker, fileName))

        Returns:
            np.array: Mel spectrogram
        """
        if self.config["trim_silence"]:
            wav, _ = librosa.effects.trim(wav,
                                            top_db=self.config["trim_threshold_in_db"],
                                            frame_length=self.config["trim_frame_size"],
                                            hop_length=self.config["trim_hop_size"])
        
        if sample_rate != self.config["sampling_rate"]: #Resampling
            # log.inf("Resampling ")
            wav = librosa.resample(wav, sample_rate, self.config["sampling_rate"])

        mel = logmelfilterbank( #Create mel spectrogram using the melGAN settings
                        wav,  
                        sampling_rate=self.config["sampling_rate"],
                        hop_size=self.config["hop_size"],
                        fft_size=self.config["fft_size"],
                        win_length=self.config["win_length"],
                        window=self.config["window"],
                        num_mels=self.config["num_mels"],
                        fmin=self.config["fmin"],
                        fmax=self.config["fmax"])
        
        # make sure the audio length and feature length are matched
        wav = np.pad(wav, (0, self.config["fft_size"]), mode="reflect")
        wav = wav[:len(mel) * self.config["hop_size"]]
        assert len(mel) * self.config["hop_size"] == len(wav)

        #================================================Normalization=========================================================
        # restore scaler
        scaler = StandardScaler()
        if self.config["format"] == "hdf5": 
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

    