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
from data_converter_melgan import MelganConverter
from parallel_wavegan.utils import read_hdf5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    print(torch.cuda.get_device_name(0))

converter = Converter(device)

import yaml

from parallel_wavegan.utils import download_pretrained_model, load_model

#==================Get melgan model==========================
print(f"Now loading in pretrained melGAN model")
download_pretrained_model("vctk_multi_band_melgan.v2", "./vocoders/melgan")
model = load_model("melgan/vctk_multi_band_melgan.v2/checkpoint-1000000steps.pkl")
model.remove_weight_norm()
model = model.eval().to(device)


#================Read audio=========================
audio, sr = sf.read("./input/p226/p226_003.wav")

#===============
melganconverter = MelganConverter(device, Config.dir_paths["melgan_config_path"],  Config.dir_paths["melgan_stats_path"])
# spect = melganconverter._wav_to_melgan_spec(audio, sr)
spect = np.load(".\\flax_example_test_dataset\\p230\\p230_002_mic1..npy") #load example


result = model.inference(torch.tensor(spect, dtype=torch.float).to(device)).view(-1)
wav = result.detach().numpy() 
utility.play_wav_from_npy(wav)