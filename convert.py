"""
Convert audio files using a pretrained model
"""
import argparse
import logging
import os
import pickle
from math import ceil

import numpy as np
import torch
import torchaudio

from autovc.model_vc import Generator
from config import Config
from data_converter import Converter

logging.basicConfig(level=logging.INFO) 
log = logging.getLogger(__name__)
    

# Parse arguments
parser = argparse.ArgumentParser(description='Transfer voices using pretrained AutoVC')
parser.add_argument("--source", default=None,
                    help="Source speaker folder")
parser.add_argument("--target", default=None,
                    help="Target speaker folder")
parser.add_argument("--source_wav", nargs='+', default=None,
                    help="Source speaker utterance")
args = parser.parse_args()


def pad_seq(x, base=32):
    len_out = int(base * ceil(float(x.shape[0])/base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0,len_pad),(0,0)), 'constant'), len_pad


def inference(output_dir, device, input_dir=None, input_data=None):
    network_path = os.path.join(Config.dir_paths["networks"], Config.pretrained_names["autovc"])   
    
    # Define AutoVC model
    G = Generator(32,256,512,32).eval().to(device)
    g_checkpoint = torch.load(network_path, map_location=device) 
    G.load_state_dict(g_checkpoint['model'])
        
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)    
    
    # Load input data
    if input_data is not None:
        metadata = input_data
    elif input_dir is not None:
        metadata = pickle.load(open(os.path.join(input_dir, Config.metadata_name), "rb"))
        log.debug(input_dir)
    
    print("Starting inference...")
    
    spect_vc = []
    for src_speaker in metadata["source"].values(): 
        emb_org = torch.from_numpy(src_speaker["emb"][np.newaxis, :]).to(device) #Get source em
        
        for speaker_j in metadata["target"].keys():
            for utterance_i in src_speaker["utterances"].keys(): 
                uttr_total = np.empty(shape=(0,80)) #used to concat sub-spectrograms (2-sec parts)
                
                for sub_utterance in src_speaker["utterances"][utterance_i]: #iterate through sub-utterances (due to 2-sec limit spects. are split up if > 2 sec)
                    # utterance = src_speaker["utterances"][utterance_i] 
                    #x_org = sbmt_i[2]
                    x_org, len_pad = pad_seq(sub_utterance)
                    uttr_org = torch.from_numpy(x_org[np.newaxis, :, :]).to(device)
                

                    trg_speaker = metadata["target"][speaker_j]
                    emb_trg = torch.from_numpy(trg_speaker["emb"][np.newaxis, :]).to(device)
                    
                    with torch.no_grad():
                        _, x_identic_psnt, _ = G(uttr_org, emb_org, emb_trg)
                        
                    if len_pad == 0:
                        uttr_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
                    else:
                        uttr_trg = x_identic_psnt[0, 0, :-len_pad, :].cpu().numpy()

                    uttr_total = np.concatenate((uttr_total, uttr_trg), axis=0) #append the split-spectrograms to create new one
                    
                    # import matplotlib.pyplot as plt
                    # fig, ax = plt.subplots(2)
                    # ax[0].imshow(np.swapaxes(uttr_trg, 0, 1))
                    # ax[1].imshow(np.swapaxes(uttr_total, 0, 1))
                    # plt.show()
                        
                spect_vc.append( ('{}x{}'.format(utterance_i, speaker_j), uttr_total))


    with open(os.path.join(output_dir, 'results.pkl'), 'wb') as handle:
        pickle.dump(spect_vc, handle) 
    
    print("Created output spectrograms...")
    
    return spect_vc
    

source_speaker = args.source if args.source is not None else "p225"
target_speaker = args.target if args.target is not None else "Wouter"
source_list = args.source_wav if args.source_wav is not None else ["p225_024"]



# directories
input_dir = Config.dir_paths["input"]
converted_data_dir = Config.dir_paths["metadata"]
output_file_dir = Config.dir_paths["output"]
metadata_name = Config.metadata_name

if not os.path.isdir(os.path.join(input_dir, source_speaker)):
    print("Didn't find a {} folder in the {} folder".format(source_speaker, input_dir))
    exit(1)
    
if not os.path.isdir(os.path.join(input_dir, target_speaker)):
    print("Didn't find a {} folder in the {} folder".format(target_speaker, input_dir))
    exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

if device.type == "cuda":
    print(torch.cuda.get_device_name(0))

#data = torchaudio.datasets.VCTK_092(".", download=True)

converter = Converter(device)

input_data = converter.wav_to_input(input_dir, source_speaker, target_speaker, source_list, converted_data_dir, metadata_name)

output_data = inference(output_file_dir, device, input_data=input_data)

converter.output_to_wav(output_data)
