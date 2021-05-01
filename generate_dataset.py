""" 
Train an AutoVC model
"""
import os
import argparse
from autovc.solver_encoder import Solver
from autovc.data_loader import get_loader
from torch.backends import cudnn

import json
import torch
from data_converter import Converter
from data_converter_melgan import MelganConverter

from config import Config
import logging
logging.basicConfig(level=logging.INFO) 
log = logging.getLogger(__name__)
import datetime

def str2bool(v):
    return v.lower() in ('true')

        

if __name__ == '__main__':
    log.info(f"Now generating a dataset")
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

#python generate_dataset.py --spectrogram_type=melgan --input_dir=train_input --output_dir=train_melgan
    #=================Conversion method of spectrograms===========
    parser.add_argument("--spectrogram_type", type=str, choices=["standard", "melgan"], default="standard",
                            help="What converter to use, use 'melgan' to train the converter + model output on 24khz fft'd spectrograms used in the parallel melgan implementation")
    parser.add_argument('--input_dir', type=str, required=True, help=f".wav file dir of with structure:\n"
                                                        f"|-input_dir                 (the specified folder \n"
                                                        f"  |-speaker_name            (for every speaker in input data)\n"
                                                        f"    |-wavefile (.wav)       (for every utterance for this speaker)\n"
                                                        f"    |-...\n"
                                                        f"  |-...\n"

                                                        )#.wav file dir of with structure: `input_dir`/speaker_name/wavname1.wav  (etc.)
    parser.add_argument('--output_dir',  type=str, required=True, help=f"Where to store the preprocessed dataset folder (folder is created automatically) with structure:\n"
                                                        f"|-output_dir                (the specified folder)\n"
                                                        f"  |-metadata.pkl            (main metadata file)\n"
                                                        f"  |-speaker_folder          (for every speaker)\n"
                                                        f"    |-utterance1.npy        (for every utterance)\n"
                                                        f"    |-..."
                                                        f"    |-embedding.npy         (single (speaker) embedding per speaker)\n"
                                                        f"  |-...\n"
                                                    )

    
    args = parser.parse_args()


    cur_time = datetime.datetime.now().strftime("%Y%m%d_%H;%M")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    #==============================Dir settings==========================
    input_dir = args.input_dir
    output_dir = args.output_dir

    #===========================Converter (+output dir addition)================================
    if args.spectrogram_type == "standard":
        log.info("Using the default AutoVC spectrogram creator")
        converter = Converter(device)
    elif args.spectrogram_type == "melgan":
        log.info("Using a melgan spectrogram-converter for dataset generator")
        # Config.dir_paths["output"] = Config.dir_paths["output"] + "_melgan" #set output to _melgan folder
        converter = MelganConverter(device, Config.dir_paths["melgan_config_path"], Config.dir_paths["melgan_stats_path"]) #TODO: pyt in args?

    #===============================create metadata (if it does not exist already)====================
    if not os.path.exists(os.path.join(output_dir, Config.train_metadata_name)): #if metadata doesnt already exist
        _ = converter.generate_train_data(input_dir, output_dir, Config.train_metadata_name)
    else:
        log.warning(f" ATTENTION: metadata already exists at: {os.path.join(output_dir, Config.train_metadata_name)}, now exiting...")
        exit(0)
            
