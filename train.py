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

def main(args, device):
    # For fast training.
    cudnn.benchmark = True

    # Data loader.
    vcc_loader = get_loader(args.data_dir, args.batch_size, args.len_crop, crop_range=args.crop_range)
    
    solver = Solver(vcc_loader, args, device, args.start_learning_rate)

    solver.train()
        
    
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    # Model configuration.
    parser.add_argument('--lambda_cd', type=float, default=1, help='weight for hidden code loss')
    parser.add_argument('--dim_neck', type=int, default=32)
    parser.add_argument('--dim_emb', type=int, default=256)
    parser.add_argument('--dim_pre', type=int, default=512)
    parser.add_argument('--freq', type=int, default=32)
    
    # Training configuration.
    parser.add_argument('--model_path', type=str, default=None) #Load a pretrained model

    parser.add_argument('--data_dir', type=str, default='train', help=f"Folder in which train.pkl resides, as well as a folder with embeddings for each speaker, of format:"
                                                                        f"|-data_dir                  (the specified folder)\n"
                                                                        f"  |-train.pkl               (main metadata file)\n"
                                                                        f"  |-speaker_folder          (for every speaker)\n"
                                                                        f"    |-utterance1.npy        (for every utterance)\n"
                                                                        f"    |-embedding.npy         (single (speaker) embedding per speaker)\n"
                                                                    )
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/cur_datetime', help="By default, create a subfolder under ./checkpoints/yearmonthday_hour;minute_checkpoints")
    parser.add_argument('--batch_size', type=int, default=2, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=1000000, help='number of total iterations')
    parser.add_argument('--len_crop', type=int, default=128, help='dataloader output sequence length')
    parser.add_argument('--save_freq', type=int, default=10000)
    parser.add_argument('--crop_range', type=tuple, default=None, nargs=2)
    parser.add_argument("--start_learning_rate", type=float, default=0.0001, help="The start learning rate of the adam optimizer, defaults to 0.0001")

    #Conversion method of spectrograms
    parser.add_argument("--spectrogram_type", type=str, choices=["standard", "melgan"], default="standard",
                            help="What converter to use, use 'melgan' to train the converter + model output on 24khz fft'd spectrograms used in the parallel melgan implementation")
    
    # Miscellaneous.
    parser.add_argument('--log_step', type=int, default=10)
    
    args = parser.parse_args()
    #==========================
    cur_time = datetime.datetime.now().strftime("%Y%m%d_%H;%M")
    if args.checkpoint_dir == './checkpoints/cur_datetime':
        args.checkpoint_dir = f"./checkpoints/{cur_time}_checkpoints"

    if not os.path.isdir(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    


    with open(os.path.join(args.checkpoint_dir, 'train_args.txt'), 'w') as f: #Save train arguments
        json.dump(args.__dict__, f, indent=2)
    
    with open(os.path.join(args.checkpoint_dir, 'config.txt'), 'w') as f: #Save the used config as well
        json.dump(Config.__dict__, f, indent=2, default=str)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #=============================Run main training loop===========================
    main(args, device)