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
from numpy.random import RandomState
from scipy import signal
from scipy.signal import get_window

from autovc.model_bl import D_VECTOR
from config import Config

log = logging.getLogger(__name__)



class Converter:
    """
    Convert audio data to and from the AutoVC format
    """
    
    def __init__(self, device):
        self._device = device
        self._prng = RandomState(42) #TODO: should this be the same each time?
        
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

        mel_basis = mel(Config.audio_sr, Config.n_fft, fmin=Config.fmin, fmax=Config.fmax, n_mels=Config.n_mels).T
        min_level = np.exp(Config.min_level_db / 20 * np.log(10))
        b, a = self._butter_highpass(30, Config.audio_sr, order=5)

        # Resample wav if needed
        if sample_rate != Config.audio_sr:
            wav = librosa.resample(wav, sample_rate, Config.audio_sr)
            print(f"Wav file with sr {sample_rate} != {Config.audio_sr}, Now resampling to {Config.audio_sr}, then try to write to {wav_path}")

            if wav_path:
                sf.write(wav_path, wav, Config.audio_sr) # Write downsampled file
        
        # Remove drifting noise
        wav = signal.filtfilt(b, a, wav)

        # add a little random noise for model robustness
        if introduce_noise:
            log.info(f"Introducing random noise into wav.file")
            
            wav = wav * 0.96 + (self._prng.rand(wav.shape[0])-0.5)*1e-06
        

        # Compute spectrogram
        D = self._pySTFT(wav, fft_length=Config.n_fft, hop_length=Config.hop_length).T
        # Convert to mel and normalize
        D_mel = np.dot(D, mel_basis)
        D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - Config.ref_level_db # amp to db
        S = np.clip((D_db - Config.min_level_db) / -Config.min_level_db, 0, 1) # clip between 0-1
        
        return S
    
    
    def _wav_dir_to_spec_dir(self, input_dir, output_dir, speakers=None, introduce_noise=False, skip_existing = True):
        """Convert wav files in folder `input_dir` to a mel spectrogram, then puts them (numpy 2d spectrograms) in `output_dir`

        Args:
            input_dir (str): Path to input directory
            output_dir (str): Path to output directory

        Returns:
            np.array: Mel spectrogram
        """
        spects = {}
  
        # If no speakers specified, load all speakers
        if speakers is None:
            _, sub_dirs, _ = next(os.walk(input_dir))
            speakers = sub_dirs

        #for subdir in sorted(subdirList): #TODO: load from file if already exist? parameter that determines whether result should be saved
        for speaker in speakers:
            print(speaker)   
            if not os.path.exists(os.path.join(output_dir, speaker)):
                os.makedirs(os.path.join(output_dir, speaker))
                
            _,_, fileList = next(os.walk(os.path.join(input_dir, speaker)))
            
            spects[speaker] = {}
            #prng = RandomState(int(subdir[1:])) 
            for fileName in sorted(fileList):
                save_name = os.path.join(output_dir, speaker, fileName[:-4]) + ".npy"
                if skip_existing and os.path.exists(save_name): #if skip existing is set to true, and result already exists
                    log.info(f"Loading spectrogram of {fileName} from {save_name} ")
                    S = np.load(save_name) #Reload predefined spect #TODO: skip load
                else:
                    log.info(f"Spect of {fileName} does not yet exist at: {save_name} ")
                    wav_path = os.path.join(input_dir, speaker, fileName)
                    x, sr = sf.read(wav_path)
                    S = self._wav_to_spec(x, sr, introduce_noise)

                    # Save spectrogram    
                    np.save(save_name, S.astype(np.float32), allow_pickle=False)
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


            dict: Metadata object
        """
        metadata = {"source" : {source : {"utterances" : {}}}, "target" : {target : {}}} # TODO: extend to multiple sources and targets
        
        # Source speaker embedding
        speaker_emb = np.load(os.path.join(input_dir, source, source + "_emb.npy"))
        metadata["source"][source]["emb"] = speaker_emb
        
        # Source speaker utterances
        for utterance in source_list:
            spect = np.load(os.path.join(input_dir, source, utterance + ".npy"))
            
            spects = []
            if len_crop > 0:
            
                # Split utterance 
                spect_count = math.ceil(spect.shape[0]/len_crop) #Get amount of ~2-second spectrograms
                # frames_per_spec = int(spect.shape[0]/spect_count) #get frames per spectrogram
                
                
                i = 0
                for i in range(spect_count - 1):
                    spects.append(spect[len_crop * i: len_crop * (i+1), :] )

                spects.append(spect[len_crop * (i+1): , :] ) #append the rest
                print("Amount of parts: {}".format(len(spects)))
            else:
                spects = [spect]
                
            
            metadata["source"][source]["utterances"][utterance] = spects
        
        # Target speaker embedding
        speaker_emb = np.load(os.path.join(input_dir, target, target + "_emb.npy"))
        metadata["target"][target]["emb"] = speaker_emb
            
        return metadata


    def _spec_to_embedding(self, output_dir, input_data, skip_existing = True): 
        """Generates speaker embeddings from spectograms from internal memory or a directory

        Args:
            output_dir (str): Path to output directory
            input_data (list], optional): Spectogram from internal memory. Defaults to None.
            skip_existing (bool): Whether output dir should be checked for output name first, if output already exists, load entry and skip processing

        Returns:
            list: List of speaker embeddings of the form:
                {
                    #'speaker' : emedding
                    'p225': array([ 0.00812339, ...e=float32),
                    'p226': .... etc. 
                }
            
        """
        # Load speaker encoder
        network_dir = Config.dir_paths["networks"]
        speaker_encoder_name = Config.pretrained_names["speaker_encoder"]
        speaker_encoder = D_VECTOR(**Config.wavenet_arch).eval().to(self._device)
        c_checkpoint = torch.load(os.path.join(network_dir, speaker_encoder_name), map_location=self._device)     
        
        new_state_dict = OrderedDict()
        for key, val in c_checkpoint['model_b'].items():
            new_key = key[7:]
            new_state_dict[new_key] = val
        speaker_encoder.load_state_dict(new_state_dict)
        
        
        num_uttrs = Config.emb_num_uttr # TODO: Why not just use all files?
        len_crop = Config.emb_len_crop
        
        # if input_data is not None:
        spects = input_data
        # elif input_dir is not None:
        #     spects = self._load_spec_data(input_dir)

        speaker_embeddings = {}
        for speaker in sorted(spects.keys()):
            
            save_path = os.path.join(output_dir, speaker, "{}_emb".format(speaker)) + ".npy"
            if skip_existing and os.path.exists(save_path):
                log.info(f'Embedding - loading from file for speaker: {speaker} ({save_path})')
                speaker_embeddings[speaker] = np.load(save_path, allow_pickle=False) 
                continue
            else:
                log.info(f'Embedding - Processing speaker: {speaker}')

            utterances_list = spects[speaker]
            # make speaker embedding
            assert len(utterances_list) >= num_uttrs 
            idx_uttrs = np.random.choice(len(utterances_list), size=num_uttrs, replace=False)
            
            embs = []
            for i in range(num_uttrs): # TODO: weird stuff? while loop seems to load first valid file multiple times if multiple invalids 
                file = list(utterances_list.keys())[idx_uttrs[i]]
                spect = utterances_list[file]
                
                candidates = np.delete(np.arange(len(utterances_list)), idx_uttrs[i])
                
                # choose another utterance if the current one is too short
                while spect.shape[0] < len_crop:
                    idx_alt = np.random.choice(candidates)
                    file = list(utterances_list.keys())[idx_alt]
                    spect = utterances_list[file]
                    candidates = np.delete(candidates, np.argwhere(candidates==idx_alt))
                    
                left = np.random.randint(0, spect.shape[0]-len_crop+1)
                melsp = torch.from_numpy(spect[np.newaxis, left:left+len_crop, :]).to(self._device)
                emb = speaker_encoder(melsp)
                embs.append(emb.detach().squeeze().cpu().numpy())     
            
            speaker_embeddings[speaker] = np.mean(embs, axis=0)
            
            np.save(save_path, 
                    speaker_embeddings[speaker], allow_pickle=False)
               
            
        print("Extracted speaker embeddings...")
        return speaker_embeddings


    def _check_embeddings(self, input_dir, speakers):
        for speaker in speakers:
            speaker_path = os.path.join(input_dir, speaker)
            embedding_path = os.path.join(speaker_path, f"{speaker}_emb.npy")
            if not os.path.exists(embedding_path):
                print(f"{speaker} embedding not found!")
                return False # TODO: Generate embedding here
        
        return True
                

    def wav_to_convert_input(self, input_dir, source, target, source_list, output_dir, output_file, split_spects=True, skip_existing=True):
        """Convert wav files to input metadata

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
        
        # Split utterence into ~2s parts or not
        if split_spects:
            len_crop = Config.len_crop
        else:
            len_crop = 0
        
        if not skip_existing or not self._check_embeddings(spec_dir, speakers):
            # Convert audio to spectrograms
            spects = self._wav_dir_to_spec_dir(input_dir, spec_dir, speakers, skip_existing=skip_existing)
            
            # Generate speaker embeddings
            embeddings = self._spec_to_embedding(spec_dir, spects, skip_existing=skip_existing) 
        
        # Create conversion metadata
        metadata = self._create_metadata(spec_dir, source, target, source_list, len_crop=len_crop)
        
        with open(os.path.join(output_dir, output_file), 'wb') as handle:
            pickle.dump(metadata, handle) 
        
        return metadata
    
 
    def _make_train_metadata(self, spects_dir, embeddings):
        """Creates train metadata

        Args:
            spects_dir (str): Path to spectogram folder
            embeddings (dict): Dictionary of speaker embeddings output from 

        Returns:
            list: Train metadata. See README.md for format
        """
        speakers = []

        for speaker in embeddings.keys():
            utterances = []
            utterances.append(speaker)
            utterances.append(embeddings[speaker])
   
            _, _, files = next(os.walk(os.path.join(spects_dir,speaker)))

            for file in sorted(files):
                if not "emb" in file: # Exclude embedding files
                    utterances.append(os.path.join(speaker,file))
            speakers.append(utterances)
   
        return speakers
 
 
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
        
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
   
        spects = self._wav_dir_to_spec_dir(input_dir, output_dir, introduce_noise=True) # TODO: noise for training data?
        embeddings = self._spec_to_embedding(output_dir, input_data=spects)
        metadata = self._make_train_metadata(output_dir, embeddings)
        
        with open(os.path.join(output_dir, output_file), 'wb') as handle:
            pickle.dump(metadata, handle)
 
    
    # def preprocess_melgan(self, mel):
    #     with open("melgan/vctk_parallel_wavegan.v1/config.yml") as f:
    #         config = yaml.load(f, Loader=yaml.Loader)
    #     stats = "melgan/vctk_parallel_wavegan.v1/stats.h5"
        
    #     lin_out = librosa.feature.inverse.mel_to_stft(mel, sr=Config.audio_sr, n_fft=Config.n_fft, fmin=Config.fmin, fmax=Config.fmax) 

    #     # Use MelGan mel format 
    #     mel_out = logmelfilterbank(lin_out,
    #                                 sampling_rate=config["sampling_rate"],
    #                                 hop_size=config["hop_size"],
    #                                 fft_size=config["fft_size"],
    #                                 win_length=config["win_length"],
    #                                 window=config["window"],
    #                                 num_mels=config["num_mels"],
    #                                 fmin=config["fmin"],
    #                                 fmax=config["fmax"])

    #     # Normalize melgan mel spect
    #     scaler = StandardScaler()
    #     if config["format"] == "hdf5":
    #         scaler.mean_ = read_hdf5(stats, "mean")
    #         scaler.scale_ = read_hdf5(stats, "scale")
    #     elif config["format"] == "npy":
    #         scaler.mean_ = np.load(stats)[0]
    #         scaler.scale_ = np.load(stats)[1]

    #     mel_out = scaler.transform(mel_out)
        
    #     return mel_out
 
 
