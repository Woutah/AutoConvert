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

logging.basicConfig(level=logging.INFO) 
log = logging.getLogger(__name__)
    

# Parse arguments
parser = argparse.ArgumentParser(description='Transfer voices using pretrained AutoVC')
parser.add_argument("--source", default=None,
                    help="Source speaker folder")
parser.add_argument("--target", default=None,
                    help="Target speaker folder")
parser.add_argument("--source_wav", default=None,
                    help="Source speaker utterance")
args = parser.parse_args()


def pad_seq(x, base=32):
    len_out = int(base * ceil(float(x.shape[0])/base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0,len_pad),(0,0)), 'constant'), len_pad


def inference(output_dir, device, input_dir=None, input_data=None):
    network_path = os.path.join(Config.dir_paths["networks"], Config.pretrained_names["autovc"])   
    G = Generator(32,256,512,32).eval().to(device)
    g_checkpoint = torch.load(network_path, map_location=device) 
    G.load_state_dict(g_checkpoint['model'])
        
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)    
    
    if input_data is not None:
        metadata = input_data
    elif input_dir is not None:
        metadata = pickle.load(open(os.path.join(input_dir, Config.metadata_name), "rb"))
        log.debug(input_dir)
    
    print("Starting inference...")
    
    spect_vc = []
    for src_speaker in metadata["source"].values():
        
        emb_org = torch.from_numpy(src_speaker["emb"][np.newaxis, :]).to(device)
        
        
        count = 0
        for utterance_i in src_speaker["utterances"].keys(): 
            utterance = src_speaker["utterances"][utterance_i]
            #x_org = sbmt_i[2]
            x_org, len_pad = pad_seq(utterance)
            uttr_org = torch.from_numpy(x_org[np.newaxis, :, :]).to(device)
        

            for speaker_j in metadata["target"].keys():
                trg_speaker = metadata["target"][speaker_j]
                        
                emb_trg = torch.from_numpy(trg_speaker["emb"][np.newaxis, :]).to(device)
                
                with torch.no_grad():
                    _, x_identic_psnt, _ = G(uttr_org, emb_org, emb_trg)
                    
                if len_pad == 0:
                    uttr_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
                else:
                    uttr_trg = x_identic_psnt[0, 0, :-len_pad, :].cpu().numpy()
                
                spect_vc.append( ('{}x{}'.format(utterance_i, speaker_j), uttr_trg))
                
            count += 1

    with open(os.path.join(output_dir, 'results.pkl'), 'wb') as handle:
        pickle.dump(spect_vc, handle) 
    
    print("Created output spectograms...")
    
    return spect_vc
    


source_speaker = args.source if args.source is not None else "p225"
target_speaker = args.target if args.target is not None else "p225"
source_list = args.source_wav if args.source_wav is not None else ["p225_024"]

    

# directories
input_dir = Config.dir_paths["input"]
converted_data_dir = Config.dir_paths["metadata"]
output_file_dir = Config.dir_paths["output"]
metadata_name = Config.metadata_name

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    print(torch.cuda.get_device_name(0))


converter = Converter(device)

input_data = converter.wav_to_input(input_dir, source_speaker, target_speaker, source_list, converted_data_dir, metadata_name)

output_data = inference(output_file_dir, device, input_data=input_data)

converter.output_to_wav(output_data)


    

