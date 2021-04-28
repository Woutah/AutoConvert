import os
from data_converter import Converter
import torch
import pickle
from math import ceil
import numpy as np
from autovc.model_vc import Generator
import soundfile as sf
import logging
from config import Config
import argparse

from numpy.random import RandomState
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    print(torch.cuda.get_device_name(0))


converter = Converter(device)

S = 
spec_dir = Config.dir_paths["spectrograms"]
specs = converter._wav_to_spec(input_dir, spec_dir)


# spect_convert_list =  [('Wouter_test_wav_to_spect_to_wav', specs["Wouter"]["6"])] #6 = "This is a test sentence"

# converter.output_to_wav( spect_convert_list )

# print("Done")
# input_data = converter.wav_to_input(input_dir, source_speaker, target_speaker, source_list, target_list, converted_data_dir, metadata_name)

# output_data = inference(output_file_dir, device, input_data=input_data)