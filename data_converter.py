import os
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.signal import get_window
from scipy.signal import get_window
from librosa.filters import mel
from numpy.random import RandomState
import pickle
from autovc.model_bl import D_VECTOR
from collections import OrderedDict
import torch


class Converter:
    
    def __init__(self, device):
        self._device = device
  
    
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
    
    
    def _wav_to_spec(self, input_dir, output_dir): # TODO: to config
        mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
        min_level = np.exp(-100 / 20 * np.log(10))
        b, a = self._butter_highpass(30, 16000, order=5)

        dirName, subdirList, _ = next(os.walk(input_dir)) 

        for subdir in sorted(subdirList):
            print(subdir)
            
            if not os.path.exists(os.path.join(output_dir, subdir)):
                os.makedirs(os.path.join(output_dir, subdir))
                
            _,_, fileList = next(os.walk(os.path.join(dirName,subdir)))
            
            prng = RandomState(int(subdir[1:])) 
            for fileName in sorted(fileList):
                # Read audio file
                x, fs = sf.read(os.path.join(dirName,subdir,fileName))
                # Remove drifting noise
                y = signal.filtfilt(b, a, x)
                # Ddd a little random noise for model roubstness
                wav = y * 0.96 + (prng.rand(y.shape[0])-0.5)*1e-06
                # Compute spect
                D = self._pySTFT(wav).T
                # Convert to mel and normalize
                D_mel = np.dot(D, mel_basis)
                D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
                S = np.clip((D_db + 100) / 100, 0, 1)    
                # save spect    
                np.save(os.path.join(output_dir, subdir, fileName[:-4]),
                        S.astype(np.float32), allow_pickle=False)


    def _spec_to_metadata(self, input_dir, output_dir):
        C = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval().to(self._device)
        c_checkpoint = torch.load('networks/3000000-BL.ckpt') # TODO: to config
        
        new_state_dict = OrderedDict()
        for key, val in c_checkpoint['model_b'].items():
            new_key = key[7:]
            new_state_dict[new_key] = val
        C.load_state_dict(new_state_dict)
        num_uttrs = 10
        len_crop = 128

        # Directory containing mel-spectrograms
        dirName, subdirList, _ = next(os.walk(input_dir))
        print('Found directory: %s' % dirName)


        speakers = []
        for speaker in sorted(subdirList):
            print('Processing speaker: %s' % speaker)
            metadata = []
            metadata.append(speaker)
            _, _, fileList = next(os.walk(os.path.join(dirName,speaker)))
            
            # make speaker embedding
            assert len(fileList) >= num_uttrs
            idx_uttrs = np.random.choice(len(fileList), size=num_uttrs, replace=False)
            embs = []
            full_spects = []
            
            for i in range(num_uttrs): # TODO: weird stuff? while loop seems to load first valid file multiple times if multiple invalids 
                spect = np.load(os.path.join(dirName, speaker, fileList[idx_uttrs[i]]))
                full_spects.append(spect)
                
                candidates = np.delete(np.arange(len(fileList)), idx_uttrs)
                
                # choose another utterance if the current one is too short
                while spect.shape[0] < len_crop:
                    idx_alt = np.random.choice(candidates)
                    spect = np.load(os.path.join(dirName, speaker, fileList[idx_alt]))
                    candidates = np.delete(candidates, np.argwhere(candidates==idx_alt))
                    
                left = np.random.randint(0, spect.shape[0]-len_crop)
                melsp = torch.from_numpy(spect[np.newaxis, left:left+len_crop, :]).cuda()
                emb = C(melsp)
                embs.append(emb.detach().squeeze().cpu().numpy())     
            
            metadata.append(np.mean(embs, axis=0))
            metadata.append(np.array(full_spects))
            
            speakers.append(metadata)
            
        with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as handle: # TODO: config?
            pickle.dump(speakers, handle)

    def wav_to_input(self, input_dir, output_dir):
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        
        spec_dir = "./spectograms"
        self._wav_to_spec(input_dir, spec_dir)
        self._spec_to_metadata(spec_dir, output_dir)