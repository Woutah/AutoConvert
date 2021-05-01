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

from config import Config
import logging
logging.basicConfig(level=logging.INFO) 
log = logging.getLogger(__name__)
import datetime

def str2bool(v):
    return v.lower() in ('true')

def main(config, device):
    # For fast training.
    cudnn.benchmark = True

    # Data loader.
    vcc_loader = get_loader(config.data_dir, config.batch_size, config.len_crop, crop_range=config.crop_range)
    
    solver = Solver(vcc_loader, config, device)

    solver.train()
        
    
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--lambda_cd', type=float, default=1, help='weight for hidden code loss')
    parser.add_argument('--dim_neck', type=int, default=32)
    parser.add_argument('--dim_emb', type=int, default=256)
    parser.add_argument('--dim_pre', type=int, default=512)
    parser.add_argument('--freq', type=int, default=32)
    
    # Training configuration.
    parser.add_argument('--model_path', type=str, default=None) #Load a pretrained model
    parser.add_argument('--input_dir', type=str, default='./train_input')#.wav file dir of with structure: `input_dir`/speaker_name/wavname1.wav  (etc.)
    parser.add_argument('--data_dir', type=str, default='./train')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--batch_size', type=int, default=2, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=1000000, help='number of total iterations')
    parser.add_argument('--len_crop', type=int, default=128, help='dataloader output sequence length')
    parser.add_argument('--save_freq', type=int, default=10000)
    parser.add_argument('--crop_range', type=tuple, default=None)
    
    # Miscellaneous.
    parser.add_argument('--log_step', type=int, default=10)
    
    config = parser.parse_args()
    
    if not os.path.isdir(config.checkpoint_dir):
        os.mkdir(config.checkpoint_dir)
    
    cur_time = datetime.datetime.now().strftime("%Y%m%d_%H;%M")


    with open(os.path.join(config.checkpoint_dir, cur_time+'_train_args.txt'), 'w') as f:
        json.dump(config.__dict__, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(os.path.join(config.data_dir, Config.train_metadata_name)):
        converter = Converter(device)
        
        input_dir = config.input_dir
        output_dir = config.data_dir
        output_file = Config.train_metadata_name
        _ = converter.generate_train_data(input_dir, output_dir, output_file)
            
    main(config, device)