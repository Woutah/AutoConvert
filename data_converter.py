import logging
import math
import os
import pickle
from collections import OrderedDict
import librosa

import numpy as np
import soundfile as sf
import torch
from librosa.filters import mel
from librosa import resample
from scipy import signal
from scipy.signal import get_window

from autovc.model_bl import D_VECTOR
from autovc.synthesis import build_model, wavegen
from config import Config

log = logging.getLogger(__name__)

class Converter:
	"""
	Convert audio data to and from the AutoVC format
	"""
	
	def __init__(self, device):
		self._device = device
		log.info("Using device {}".format(self._device))
  
	
	def _butter_highpass(self, cutoff, fs, order=5):
		nyq = 0.5 * fs
		normal_cutoff = cutoff / nyq
		b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
		return b, a
		
		
	def _pySTFT(self, x, fft_length=1024, hop_length=256):
		
		x = np.pad(x, int(fft_length//2), mode='reflect')
		
		noverlap = fft_length - hop_length
		shape = x.shape[:-1]+((x.shape[-1]-noverlap)//hop_length, fft_length)
		strides = x.strides[:-1]+(hop_length*x.strides[-1], x.strides[-1])
		result = np.lib.stride_tricks.as_strided(x, shape=shape,
												strides=strides)
		
		fft_window = get_window('hann', fft_length, fftbins=True)
		result = np.fft.rfft(fft_window * result, n=fft_length).T
		
		return np.abs(result)
	
	
	def _wav_to_spec(self, wavfile, sample_rate, introduce_noise = False):
		"""Convert wav file to a mel spectrogram

		Args:
			wavfile (numpy array): audio data either 1-d (mono) or 2-d (stereo)
			sample_rate (int): the sampling rate of the .wav (sf.read[1])
			note that these two variables can be loaded using: 
				wavfile, sample_rate = sf.read(os.path.join(input_dir, speaker, fileName))

		Returns:
			np.array: Mel spectrogram
		"""
		
		mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
		min_level = np.exp(-100 / 20 * np.log(10))
		b, a = self._butter_highpass(30, 16000, order=5)

		#dirName, subdirList, _ = next(os.walk(input_dir)) 
		
		if sample_rate != Config.audio_sr:
			wavfile = librosa.resample(wavfile, sample_rate, Config.audio_sr)
			print("resampled to {}".format(Config.audio_sr))
		
		# Remove drifting noise
		wav = signal.filtfilt(b, a, wavfile)

		if introduce_noise:
			log.info(f"Introducing random noise into wav.file")
			wav = wav * 0.96 + (prng.rand(wav.shape[0])-0.5)*1e-06
		# add a little random noise for model robustness

		# Compute spectrogram
		D = self._pySTFT(wav).T
		# Convert to mel and normalize
		D_mel = np.dot(D, mel_basis)
		D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
		S = np.clip((D_db + 100) / 100, 0, 1)    
		
		# Save spectrogram    
		# np.save(os.path.join(output_dir, speaker, fileName[:-4]), S.astype(np.float32), allow_pickle=False)
		# spects[speaker][fileName[:-4]] = S.astype(np.float32)
				
		print("Converted input files to spectrograms...")
		return S
	
	def _wac_dir_to_spec_dir(self, input_dir, output_dir, speakers):
		"""Convert wav files in folder `input_dir` to a mel spectrogram

		Args:
			input_dir (str): Path to input directory
			output_dir (str): Path to output directory

		Returns:
			np.array: Mel spectrogram
		"""
		mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
		min_level = np.exp(-100 / 20 * np.log(10))
		b, a = self._butter_highpass(30, 16000, order=5)

		#dirName, subdirList, _ = next(os.walk(input_dir)) 
		
		spects = {}

		#for subdir in sorted(subdirList): #TODO: load from file if already exist? parameter that determines whether result should be saved
		for speaker in speakers:
			print(speaker)   
			if not os.path.exists(os.path.join(output_dir, speaker)):
				os.makedirs(os.path.join(output_dir, speaker))
				
			_,_, fileList = next(os.walk(os.path.join(input_dir, speaker)))
			
			spects[speaker] = {}
			#prng = RandomState(int(subdir[1:])) 
			for fileName in sorted(fileList):
				x, sr = sf.read(os.path.join(input_dir, speaker, fileName))
				S = self._wav_to_spec(x, sr)

				# Save spectrogram    
				np.save(os.path.join(output_dir, speaker, fileName[:-4]), S.astype(np.float32), allow_pickle=False)
				spects[speaker][fileName[:-4]] = S.astype(np.float32)
				
		print("Converted input files to spectrograms...")
		return spects


	def _load_spec_data(self, input_dir):
		"""Load spectrograms from a directory

		Args:
			input_dir (str): Path to input directory

		Returns:
			dict: Loaded mel spectrograms
		"""
		spects = {}
		# Directory containing mel-spectrograms
		dirName, subdirList, _ = next(os.walk(input_dir))
		log.debug('Found directory: %s' % dirName)
		
		for speaker in sorted(subdirList):
			_, _, fileList = next(os.walk(os.path.join(dirName,speaker)))
			
			spects[speaker] = {}
			for file in fileList:
				spect = np.load(os.path.join(dirName, speaker, file))
				spects[speaker][file[:-4]] = spect
		
		return spects
				
		
	def _create_metadata(self, input_dir, source, target, source_list, len_crop=128):
		"""Create conversion metadata using the format described in the README.md

		Args:
			input_dir (str): Path to input directory
			source (str): Name of source speaker
			target (str): Name of target speaker
			source_list (list): List of source utterances to convert
			len_crop (int, optional): Length of the audio cropping. Defaults to 128.

		Returns:
			dict: Metadata object
		"""
		metadata = {"source" : {source : {"utterances" : {}}}, "target" : {target : {}}} # TODO: extend to multiple sources and targets
		
		# Source speaker embedding
		speaker_emb = np.load(os.path.join(input_dir, source, source + "_emb.npy"))
		metadata["source"][source]["emb"] = speaker_emb
		
		# Source speaker utterances
		for utterance in source_list:
			spect = np.load(os.path.join(input_dir, source, utterance + ".npy"))
			
			# Split utterance 
			spect_count = math.ceil(spect.shape[0]/len_crop) #Get amount of ~2-second spectrograms
			# frames_per_spec = int(spect.shape[0]/spect_count) #get frames per spectrogram
			frames_per_spec = 128
			spects = []
			i = 0
			for i in range(spect_count - 1):
				spects.append(spect[frames_per_spec * i: frames_per_spec * (i+1), :] )

			spects.append(spect[frames_per_spec * (i+1): , :] ) #append the rest
			print("Amount of parts: {}".format(len(spects)))
			metadata["source"][source]["utterances"][utterance] = spects
		
		# Target speaker embedding
		speaker_emb = np.load(os.path.join(input_dir, target, target + "_emb.npy"))
		metadata["target"][target]["emb"] = speaker_emb
		
		# for utterance in target_list:
		#     spect = np.load(os.path.join(input_dir, target, utterance + ".npy"))
			
		#     metadata["target"][target]["utterances"][utterance] = spect
			
		return metadata


	def _spec_to_embedding(self, output_dir, input_data=None, input_dir=None): 
		"""Generates speaker embeddings from spectograms from internal memory or a directory

		Args:
			output_dir (str): Path to output directory
			input_data (list], optional): Spectogram from internal memory. Defaults to None.
			input_dir (str, optional): Path to input directory. Defaults to None.

		Returns:
			list: List of speaker embeddings
		"""
		# Load speaker encoder
		network_dir = Config.dir_paths["networks"]
		speaker_encoder_name = Config.pretrained_names["speaker_encoder"]
		speaker_encoder = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval().to(self._device)
		c_checkpoint = torch.load(os.path.join(network_dir, speaker_encoder_name), map_location=self._device)     
		
		new_state_dict = OrderedDict()
		for key, val in c_checkpoint['model_b'].items():
			new_key = key[7:]
			new_state_dict[new_key] = val
		speaker_encoder.load_state_dict(new_state_dict)
		
		#TODO: use existing embedding if file exists?
		
		num_uttrs = 10 # TODO: Why not just use all files?
		len_crop = 128
		
		if input_data is not None:
			spects = input_data
		elif input_dir is not None:
			spects = self._load_spec_data(input_dir)

		speaker_embeddings = {}
		for speaker in sorted(spects.keys()):
			log.info('Processing speaker: %s' % speaker)
			
			utterances_list = spects[speaker]
			
			# make speaker embedding
			assert len(utterances_list) >= num_uttrs 
			idx_uttrs = np.random.choice(len(utterances_list), size=num_uttrs, replace=False)
			
			embs = []
			for i in range(num_uttrs): # TODO: weird stuff? while loop seems to load first valid file multiple times if multiple invalids 
				file = list(utterances_list.keys())[idx_uttrs[i]]
				spect = utterances_list[file]
				
				candidates = np.delete(np.arange(len(utterances_list)), idx_uttrs)
				
				# choose another utterance if the current one is too short
				while spect.shape[0] < len_crop:
					idx_alt = np.random.choice(candidates)
					spect = utterances_list[idx_alt]
					candidates = np.delete(candidates, np.argwhere(candidates==idx_alt))
					
				left = np.random.randint(0, spect.shape[0]-len_crop)
				melsp = torch.from_numpy(spect[np.newaxis, left:left+len_crop, :]).to(self._device)
				emb = speaker_encoder(melsp[:128, :])#TODO: remove :128
				embs.append(emb.detach().squeeze().cpu().numpy())     
			
			speaker_embeddings[speaker] = np.mean(embs, axis=0)
			
			np.save(os.path.join(output_dir, speaker, "{}_emb".format(speaker)), 
								 speaker_embeddings[speaker], allow_pickle=False)
			   
			
		print("Extracted speaker embeddings...")
		return speaker_embeddings


	def wav_to_input(self, input_dir, source, target, source_list, output_dir, output_file):
		"""Conver wav files to input metadata

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
		spec_dir = Config.dir_paths["spectrograms"] # Where to save generated spects
		
		if not os.path.exists(output_dir):
			os.mkdir(output_dir)
			
		speakers = [source, target]
		
		# Convert audio to spectrograms
		spects = self._wav_dir_to_spec_dir(input_dir, spec_dir, speakers)
		
		# Generate speaker embeddings
		embeddings = self._spec_to_embedding(spec_dir, input_data=spects)
		
		# Create conversion metadata
		metadata = self._create_metadata(spec_dir, source, target, source_list)
		
		with open(os.path.join(output_dir, output_file), 'wb') as handle:
			pickle.dump(metadata, handle) 
		
		return metadata
	
	def output_to_wav(self, output_data):
		"""Convert mel spectograms to audio files

		Args:
			output_data (list): List of mel spectograms to convert
		"""
		model = build_model().to(self._device)
		checkpoint = torch.load(os.path.join(Config.dir_paths["networks"], Config.pretrained_names["vocoder"]), map_location=self._device)
		model.load_state_dict(checkpoint["state_dict"])
		
		print("Starting vocoder...")
		for spect in output_data:
			name = spect[0]
			print(name)
			# TODO: enable this for wavenet conversion
			#------------------------------------------------
			# c = spect[1]   
			# waveform = wavegen(model, self._device, c=c)
			#------------------------------------------------
			
			# TODO: enable this for librosa conversion
			#------------------------------------------------
			import librosa
			c = spect[1].T
			import matplotlib.pyplot as plt
			import librosa.display
			plt.figure(figsize=(10, 4))
			c = (np.clip(c, 0, 1) * - -100) + 100 # https://github.com/auspicious3000/autovc/issues/14 I suspect other values are used in the AutoVC code but it seems to somewhat work
			#c = np.power(10.0, c * 0.05)
			c = librosa.db_to_amplitude(c, ref=20)
			
			librosa.display.specshow(librosa.power_to_db(c, ref=np.max),
									y_axis='mel', fmax=7600,
									x_axis='time')
			plt.colorbar(format='%+2.0f dB')
			plt.title('Mel spectrogram')
			plt.tight_layout()
			plt.show()
			
			waveform = librosa.feature.inverse.mel_to_audio(c, sr=16000, n_fft=1024, hop_length=256)
			name += "_librosa"
			#--------------------------------------------------
			
			sf.write(os.path.join(Config.dir_paths["output"], name + ".wav"), waveform, 16000)
