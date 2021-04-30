import librosa
import numpy as np

import soundfile as sf

from autovc.synthesis import build_model, wavegen
import torch
from autovc.model_vc import Generator
import math
import pickle

device = "cuda"
auto_VC = Generator(32,256,512,32).eval().to(device)
g_checkpoint = torch.load('networks/autovc.ckpt', map_location=device) 
auto_VC.load_state_dict(g_checkpoint['model'])

metadata = pickle.load(open('autovc/metadata.pkl', "rb"))
spkr_to_embed = {entry[0] : entry[1] for entry in metadata}

sr = 16000
n_fft = 1024
win_length = 1024
hop_length = 256
n_mels = 80
fmin = 90
fmax = 7600
ref_level_db = 20
min_level_db = -100


def pad_seq(x, base=32):
    len_out = int(base * math.ceil(float(x.shape[0])/base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0,len_pad),(0,0)), 'constant'), len_pad

def _amp_to_db(x):
    min_level = np.exp(min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))

def _db_to_amp(x):
    return np.power(10.0, x * 0.05)

def normalize_for_VC(mel):
    # assume magnitude melspectrum with correct sr/fmin/fmax as input
    mel = _amp_to_db(mel) - ref_level_db
    mel = np.clip((mel - min_level_db) / -min_level_db, 0, 1)
    return mel.T

def denormalize_from_VC(mel):
    mel = (np.clip(mel, 0, 1) * -min_level_db) + min_level_db
    mel = _db_to_amp(mel + ref_level_db)
    return mel.T

def apply_autoVC(mel, embed_in, embed_out):
    # assume normalized mel spectrogram as input (normalized to db scale)
    # assume numpy input for both mel spect and embedding
    mel, len_pad = pad_seq(mel)
    
    mel       = torch.from_numpy(      mel[np.newaxis, ...]).to(device)
    embed_in  = torch.from_numpy( embed_in[np.newaxis, ...]).to(device)
    embed_out = torch.from_numpy(embed_out[np.newaxis, ...]).to(device)
    
    with torch.no_grad():
        mel_no_PN, mel_yes_PN, _ = auto_VC(mel, embed_in, embed_out)
            
        if len_pad == 0:
            mel_no_PN  =  mel_no_PN[0, 0, :, :].cpu().numpy()
            mel_yes_PN = mel_yes_PN[0, 0, :, :].cpu().numpy()
        else:
            mel_no_PN  =  mel_no_PN[0, 0, :-len_pad, :].cpu().numpy()
            mel_yes_PN = mel_yes_PN[0, 0, :-len_pad, :].cpu().numpy()
    
    return mel_no_PN, mel_yes_PN





# embed_in = np.load("spectrograms/p225/p225_emb.npy")
# embed_out = np.load("spectrograms/p225/p225_emb.npy")
embed_in = spkr_to_embed["p225"]
embed_out = spkr_to_embed["p225"]

audio_in, _ = librosa.load("train_input/p225/p225_001.wav", sr=sr)

lin_in = np.abs(librosa.stft(audio_in, n_fft=n_fft, win_length=win_length, hop_length=hop_length))
mel_in = librosa.feature.melspectrogram(S=lin_in, sr=sr, n_fft=n_fft, fmin=fmin, fmax=fmax, n_mels=n_mels) 

mel_in = normalize_for_VC(mel_in)

mel_no_PN, mel_yes_PN = apply_autoVC(mel_in, embed_in, embed_out)

mel_out = denormalize_from_VC(mel_yes_PN)

lin_out = librosa.feature.inverse.mel_to_stft(mel_out, n_fft=n_fft, sr=sr, fmin=fmin, fmax=fmax) 
audio = librosa.griffinlim(lin_out, win_length=win_length, hop_length=hop_length)

sf.write("output/test.wav", audio, 16000)