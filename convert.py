import os

from data_converter import Converter
import torch
import pickle
from math import ceil
import numpy as np
from autovc.model_vc import Generator
from synthesis import build_model
from synthesis import wavegen
import soundfile as sf


def pad_seq(x, base=32):
    len_out = int(base * ceil(float(x.shape[0])/base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0,len_pad),(0,0)), 'constant'), len_pad


def inference(input_dir, output_dir, device):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    G = Generator(32,256,512,32).eval().to(device)
    g_checkpoint = torch.load('./networks/autovc.ckpt', map_location=device) # TODO: config
    G.load_state_dict(g_checkpoint['model'])
    
    print(input_dir)
    metadata = pickle.load(open(os.path.join(input_dir, 'metadata.pkl'), "rb"))
    
    spect_vc = []

    for sbmt_i in metadata:
        emb_org = torch.from_numpy(sbmt_i[1][np.newaxis, :]).to(device)
        
        
        count = 0
        for utterance in sbmt_i[2]: 
            #x_org = sbmt_i[2]
            x_org, len_pad = pad_seq(utterance)
            uttr_org = torch.from_numpy(x_org[np.newaxis, :, :]).to(device)
        

            for sbmt_j in metadata:
                        
                emb_trg = torch.from_numpy(sbmt_j[1][np.newaxis, :]).to(device)
                
                with torch.no_grad():
                    _, x_identic_psnt, _ = G(uttr_org, emb_org, emb_trg)
                    
                if len_pad == 0:
                    uttr_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
                else:
                    uttr_trg = x_identic_psnt[0, 0, :-len_pad, :].cpu().numpy()
                
                spect_vc.append( ('{}x{}_{}'.format(sbmt_i[0], sbmt_j[0], count), uttr_trg))
                
            count += 1

    with open(os.path.join(output_dir, 'results.pkl'), 'wb') as handle:
        pickle.dump(spect_vc, handle) 
    
    return spect_vc
    

def output_to_wav(output_data, device):
    model = build_model().to(device)
    checkpoint = torch.load("./networks/checkpoint_step001000000_ema.pth")
    model.load_state_dict(checkpoint["state_dict"])
    
    for spect in output_data:
        name = spect[0]
        c = spect[1]
        print(name)
        waveform = wavegen(model, c=c)   
        sf.write('results/'+name+'.wav', waveform, 16000)
    
    
           

# audio file directory
input_dir = './input' # TODO: to config file

# spectrogram directory
converted_data_dir = './convert_data' # TODO: to config file
output_file_dir = "./output"
device = "cuda" if torch.cuda.is_available() else "cpu"

converter = Converter(device)

converter.wav_to_input(input_dir, converted_data_dir)

inference(converted_data_dir, output_file_dir, device)


    

