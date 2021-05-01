import librosa
import numpy as np
from config import Config
from vocoders.base_vocoder import BaseVocoder


class GriffinLim(BaseVocoder):
    def __init__(self, device):
        super().__init__(device)
    
    
    def _denormalize_from_VC(self, mel):
        """Denormalizes AutoVC mel

        Args:
            mel (np.array): Mel spectogram output from AutoVC
        """
        def _db_to_amp(x):
            return np.power(10.0, x * 0.05)
        
        mel = (np.clip(mel, 0, 1) * -Config.min_level_db) + Config.min_level_db
        mel = _db_to_amp(mel + Config.ref_level_db)
        return mel.T
    
    
    def synthesize(self, mel):
        mel = self._denormalize_from_VC(mel)
        
        lin_out = librosa.feature.inverse.mel_to_stft(mel, sr=Config.audio_sr, n_fft=Config.n_fft, fmin=Config.fmin, fmax=Config.fmax) 
        waveform = librosa.griffinlim(lin_out, win_length=Config.win_length, hop_length=Config.hop_length)
        
        return waveform
