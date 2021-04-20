import os

from spect_converter import SpectConverter

# audio file directory
input_dir = './input' # TODO: to config file

# spectrogram directory
output_dir = './spectograms' # TODO: to config file

spect_converter = SpectConverter()

spect_converter.wav_to_spect(input_dir, output_dir)


    

