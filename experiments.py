import argparse
import logging
import os
import pickle
from math import ceil

import numpy as np
import soundfile as sf
import torch
from numpy.random import RandomState

import matplotlib.pyplot as plt

import utility
from autovc.model_vc import Generator
from config import Config
from data_converter import Converter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    print(torch.cuda.get_device_name(0))

converter = Converter(device)

# fig, ax = plt.subplots(1,3)

# x, sr = sf.read(utility.get_full_path(".\\input\\Wouter\\wouter_please_call_stella.wav"))
# spec_wouter = converter._wav_to_spec(x, sr)
# ax[0].imshow(np.swapaxes(spec_wouter, 0, 1))
# ax[0].set_title("Wouter")

x, sr = sf.read(utility.get_full_path(".\\input\\p225\\p225_001.wav"))
spec_225 = converter._wav_to_spec(x, sr)
# ax[1].imshow(np.swapaxes(spec_225, 0, 1))
# ax[1].set_title("225")




# x, sr = sf.read(utility.get_full_path(".\\input\\p270_001.wav"))
# spec_270 = converter._wav_to_spec(x, sr)
# ax[2].imshow(np.swapaxes(spec_270, 0, 1))
# ax[2].set_title("270")

plt.show()