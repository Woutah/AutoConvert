import logging
import os
import pickle
from collections import OrderedDict

import numpy as np
import soundfile as sf
import torch
from librosa.filters import mel
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
    
    
    def _wav_to_spec(self, input_dir, output_dir):
        mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
        min_level = np.exp(-100 / 20 * np.log(10))
        b, a = self._butter_highpass(30, 16000, order=5)

        dirName, subdirList, _ = next(os.walk(input_dir)) 
        
        spects = {}

        for subdir in sorted(subdirList):
            
            if not os.path.exists(os.path.join(output_dir, subdir)):
                os.makedirs(os.path.join(output_dir, subdir))
                
            _,_, fileList = next(os.walk(os.path.join(dirName,subdir)))
            
            spects[subdir] = {}
            #prng = RandomState(int(subdir[1:])) 
            for fileName in sorted(fileList):
                # Read audio file
                x, _ = sf.read(os.path.join(dirName,subdir,fileName))
                # Remove drifting noise
                y = signal.filtfilt(b, a, x)
                # add a little random noise for model robustness
                #wav = y * 0.96 + (prng.rand(y.shape[0])-0.5)*1e-06 # TODO: remove for converting?
                wav = y
                # Compute spectogram
                D = self._pySTFT(wav).T
                # Convert to mel and normalize
                D_mel = np.dot(D, mel_basis)
                D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
                S = np.clip((D_db + 100) / 100, 0, 1)    
                
                # Save spectrogram    
                np.save(os.path.join(output_dir, subdir, fileName[:-4]), S.astype(np.float32), allow_pickle=False)
                spects[subdir][fileName[:-4]] = S.astype(np.float32)
                
        print("Converted input files to spectrograms...")
        return spects


    def _load_spec_data(self, input_dir):
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
                
        
    def _create_metadata(self, input_dir, output_dir, source, target, source_list, target_list, len_crop=128):
        metadata = {"source" : {source : {"utterances" : {}}}, "target" : {target : {"utterances" : {}}}} # TODO: extend to multiple sources and targets
        
        speaker_emb = np.load(os.path.join(input_dir, source, source + "_emb.npy"))
        metadata["source"][source]["emb"] = speaker_emb
        for utterance in source_list:
            spect = np.load(os.path.join(input_dir, source, utterance + ".npy"))
            
            spect_count = math.ceil(spect.shape[0]/len_crop) #Get amount of ~2-second spectrograms
            # frames_per_spec = int(spect.shape[0]/spect_count) #get frames per spectrogram
            frames_per_spec = 128
            spects = []
            i = 0
            for i in range(spect_count - 1):
                spects.append(spect[frames_per_spec * i: frames_per_spec * (i+1), :] )

            spects.append(spect[frames_per_spec * (i+1): , :] ) #append the rest
            
            
            metadata["source"][source]["utterances"][utterance] = spects
        
        speaker_emb = np.load(os.path.join(input_dir, target, target + "_emb.npy"))
        metadata["target"][target]["emb"] = speaker_emb
        for utterance in target_list:
            spect = np.load(os.path.join(input_dir, target, utterance + ".npy"))
            
            metadata["target"][target]["utterances"][utterance] = spect
            
        with open(os.path.join(output_dir, Config.metadata_name), 'wb') as handle:
            pickle.dump(metadata, handle) 
            
        return metadata


    def _spec_to_embedding(self, output_dir, device, input_data=None, input_dir=None): 
        speaker_encoder = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval().to(self._device)
        network_dir = Config.dir_paths["networks"]
        speaker_encoder_name = Config.pretrained_names["speaker_encoder"]
        c_checkpoint = torch.load(os.path.join(network_dir, speaker_encoder_name), map_location=device)     
        
        new_state_dict = OrderedDict()
        for key, val in c_checkpoint['model_b'].items():
            new_key = key[7:]
            new_state_dict[new_key] = val
        speaker_encoder.load_state_dict(new_state_dict)
        
        num_uttrs = 10
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
                emb = speaker_encoder(melsp)
                embs.append(emb.detach().squeeze().cpu().numpy())     
            
            speaker_embeddings[speaker] = np.mean(embs, axis=0)
            
            np.save(os.path.join(output_dir, speaker, "{}_emb".format(speaker)), 
                                 speaker_embeddings[speaker], allow_pickle=False)
               
            
        print("Extracted speaker embeddings...")
        return speaker_embeddings


    def wav_to_input(self, input_dir, source, target, source_list, target_list, output_dir, output_file, device):
        spec_dir = Config.dir_paths["spectrograms"]
        
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        
        spects = self._wav_to_spec(input_dir, spec_dir)
        embeddings = self._spec_to_embedding(spec_dir, device = device,input_data=spects)
        metadata = self._create_metadata(spec_dir, output_dir, source, target, source_list, target_list)
        
        return metadata
    
    def output_to_wav(self, output_data):
        model = build_model().to(self._device)
        checkpoint = torch.load(os.path.join(Config.dir_paths["networks"], Config.pretrained_names["vocoder"]))
        model.load_state_dict(checkpoint["state_dict"])
        
        print("Starting vocoder...")
        for spect in output_data:
            name = spect[0]
            c = spect[1]   
            waveform = wavegen(model, self._device, c=c)
            
            print(name)
            # c = np.transpose(spect[1])   
            
            # import matplotlib.pyplot as plt
            # plt.figure(figsize=(10, 4))
            # librosa.display.specshow(librosa.power_to_db(c,
            #                                             ref=np.max),
            #                         y_axis='mel', fmax=7600,
            #                         x_axis='time')
            # plt.colorbar(format='%+2.0f dB')
            # plt.title('Mel spectrogram')
            # plt.tight_layout()
            # plt.show()
            #waveform = mel_to_audio(c, sr=16000, n_fft=1024, hop_length=256)
            #waveform = mel_to_audio(c, sr=16000)
            
            sf.write(os.path.join(Config.dir_paths["output"], name + ".wav"), waveform, 16000)
