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
from config import Config


class MelganConverter(Converter):
    def __init__(self, device, melgan_config_path, melgan_stats_path, *args, **kwargs):
        super(MelganConverter, self).__init__(device, *args, **kwargs)
        print("initializing melganconverter")
        # self._device = device
        # log.info(f"Using device {self._device}")
        # if not melgan_config:

        
        # vocoder_conf = "melgan/vctk_multi_band_melgan.v2/config.yml"
        try:
            with open(melgan_config_path) as f:
                self.melgan_config = yaml.load(f, Loader=yaml.Loader)
        except:
            log.error(f"Unable to laod in yaml config at path: {melgan_config_path}, unknown desired settings (sampling rate etc.) exiting...")
            exit(0)
        download_pretrained_model("vctk_multi_band_melgan.v2", Config.dir_paths["melgan_download_location"]) #download model
        self.melgan_model = None
        self.melgan_stats_path = melgan_stats_path


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

    def _wav_to_melgan_spec(self, wav, sample_rate, introduce_noise = False, wav_path = None):
        """Convert wav file to a mel spectrogram using the methods used by the melgan model (e.g. 24khz when using the default dict),
        this is different from the normal AutoVC mel-spectrograms conversion methods and would thus have different results.

        This method should probably be avoided when calculating speech embeddings, as the speaker encoder is trained on 16khz data with the normal spectrogram format


        Args:
            wav (numpy array): audio data either 1-d (mono) or 2-d (stereo)
            sample_rate (int): the sampling rate of the .wav (sf.read[1])
            wav_path (str): Path to original wav file
            note that these two variables can be loaded using: 
                wavfile, sample_rate = sf.read(os.path.join(input_dir, speaker, fileName))

        Returns:
            np.array: Mel spectrogram (converted using melgan spec)
        """
        print("Converting using wav to melgan!")
        if self.melgan_config["trim_silence"]:
            wav, _ = librosa.effects.trim(wav,
                                            top_db=self.melgan_config["trim_threshold_in_db"],
                                            frame_length=self.melgan_config["trim_frame_size"],
                                            hop_length=self.melgan_config["trim_hop_size"])

        if introduce_noise:
            log.error(f"Introduce_noise is set tot {introduce_noise}, however, this is not implemented. Exiting...")
            exit(0)

        if sample_rate != self.melgan_config["sampling_rate"]: #Resampling
            # log.inf("Resampling ")
            wav = librosa.resample(wav, sample_rate, self.melgan_config["sampling_rate"])
            print(f"Wav file with sr {sample_rate} != {self.melgan_config['sampling_rate']}, Now resampling to {self.melgan_config['sampling_rate']}")

        mel = self.logmelfilterbank( #Create mel spectrogram using the melGAN settings
                        wav,  
                        sampling_rate=self.melgan_config["sampling_rate"],
                        hop_size=self.melgan_config["hop_size"],
                        fft_size=self.melgan_config["fft_size"],
                        win_length=self.melgan_config["win_length"],
                        window=self.melgan_config["window"],
                        num_mels=self.melgan_config["num_mels"],
                        fmin=self.melgan_config["fmin"],
                        fmax=self.melgan_config["fmax"])
        
        # make sure the audio length and feature length are matched
        wav = np.pad(wav, (0, self.melgan_config["fft_size"]), mode="reflect")
        wav = wav[:len(mel) * self.melgan_config["hop_size"]]
        assert len(mel) * self.melgan_config["hop_size"] == len(wav)

        #================================================Normalization=========================================================
        # restore scaler
        scaler = StandardScaler()
        if self.melgan_config["format"] == "hdf5":
            scaler.mean_ = read_hdf5(self.melgan_stats_path, "mean")
            scaler.scale_ = read_hdf5(self.melgan_stats_path, "scale")
        # elif config["format"] == "npy":
            # scaler.mean_ = np.load(args.stats)[0]
            # scaler.scale_ = np.load(args.stats)[1]
        else:
            raise ValueError("support only hdf5 (and normally npy - but not now) format.... cannot load in scaler mean/scale, exiting")
            exit(0)
        # from version 0.23.0, this information is needed
        scaler.n_features_in_ = scaler.mean_.shape[0]
        mel = scaler.transform(mel)
        return mel

    

    
    def generate_train_data(self, input_dir, output_dir, output_file):
        """Preprocesses input data for training, then output to `output_dir` (pickled) as a dict of form:
            {
                "source" : {
                    "speaker1" : {
                        "emb" : <speaker_embedding []>
                        "utterances" : {
                            "utterance1" : [ <part1 []>, ... , <partn []> ]
                            ...
                        }
                    }
                    ...
                }
                "target" : {
                    "speaker1" : {
                        "emb" : <speaker_embedding []>
                    }
                    ...
                }
            }

        Args:
            input_dir (str): Path to input folder containing wav files
            output_dir (str): Path to train folder to contain spectograms and metadata files
            output_file (str): Name of the metadata file
        """
        
        spec_dir_autovc = Config.dir_paths["spectrograms"] # Where to save generated spects #TODO: make sure this folder exists? 
        # spec_dir_encoder = Config.dir_paths["melgan_spectrograms"]
        spec_dir_melgan = output_dir #Save melgan-spectrograms in output_dir, as these will be used for training 

        log.info(f"Trying to generate (melgan) training data using dir_spectrograms_normal:{spec_dir_autovc}, dir_spectrograms_melgan: {spec_dir_melgan}, output dir: {output_dir}")

        #========================================== Melgan spectrograms===========================
        # Convert audio to melgan(!) spectrograms (using self._wav_to_melgan_spec function)
        spects_melgan = self._wav_dir_to_spec_dir(input_dir, spec_dir_melgan, skip_existing=True, conversion_method=self._wav_to_melgan_spec)  #TODO: introduce noise!!!!

        #===========================================Speaker embeddings using AutoVC spectrograms===========================
        # Convert audio to spectrograms using the autovc method (in order to make embeddings)
        spects_autovc = self._wav_dir_to_spec_dir(input_dir, spec_dir_autovc, skip_existing=True)

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
   
        embeddings = self._spec_to_embedding(output_dir, input_data=spects_autovc) #Create embeddings using spects_autovc

        #==============================================Metadata generation================================================
        metadata = self._make_train_metadata(spec_dir_melgan, embeddings) #create+ save metadata using the output folder (where encoder spectrograms (melgram))
        
        with open(os.path.join(output_dir, output_file), 'wb') as handle:
            pickle.dump(metadata, handle)



    def wav_to_convert_input(self, input_dir, source, target, source_list, output_dir, output_file, skip_existing=True):
        """Convert wav files to input metadata (used by convert.py to generate examples)

        Args:
            input_dir (str): Path to input directory
            source (str): Name of source speaker in the input directory
            target (str): Name of target speaker in the input directory
            source_list (list): List of source utterences to convert
            output_dir (str): Path to output directory
            output_file (str): Name of output file

        Returns:
            dict: Metadata object (See README.md for format)
        """
        log.info("Calling wav_to_convert_input from melgan converter")
        spec_dir_autovc = Config.dir_paths["spectrograms"] # Where to save generated (autovc method) spects
        spec_dir_melgan = output_dir #Where to save spectrograms generated by the MelGAN method
        
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
            
        speakers = [source, target]

        #========================================== Melgan spectrograms===========================
        # Convert audio to melgan(!) spectrograms
        spects = self._wav_dir_to_spec_dir(input_dir, spec_dir_melgan, speakers, skip_existing=skip_existing, conversion_method=self._wav_to_melgan_spec)

        #===========================================Speaker embeddings using AutoVC spectrograms===========================
        if not skip_existing or not self._check_embeddings(spec_dir_autovc, speakers):
            # Convert audio to spectrograms
            spects = self._wav_dir_to_spec_dir(input_dir, spec_dir_autovc, speakers, skip_existing=skip_existing)

            # Generate speaker embeddings, put them in the actual dir
            embeddings = self._spec_to_embedding(spec_dir_melgan, spects, skip_existing=skip_existing) 
        
        #==========================================Create conversion metadata========================================
        metadata = self._create_metadata(spec_dir_melgan, source, target, source_list) #create metadata in encoder directory
        
        with open(os.path.join(output_dir, output_file), 'wb') as handle:
            pickle.dump(metadata, handle) 
        
        return metadata