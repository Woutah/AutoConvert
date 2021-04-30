from torch.utils import data
import torch
import numpy as np
import pickle 
import os    
       
from multiprocessing import Process, Manager   


class Utterances(data.Dataset):
    """Dataset class for the Utterances dataset."""

    def __init__(self, root_dir, len_crop, crop_range = None):
        """Initialize and preprocess the Utterances dataset.

        Args:
            root_dir (str): Root dir path with metadata (see data_converter.py - generate_train_data)
            len_crop (int): Spectrogram width for each sample (`len_crop`x80 mel spectrograms) - ignored when `take_avg_length` = True 
            crop_avg_length (bool): Whether to take the average spect width for cropsize instead of a custom value - ignored `len_crop`. Defaults to False.
        """
        self.root_dir = root_dir
        self.len_crop = len_crop
        self.step = 10

        #==========cropping method===============
        self.crop_range = crop_range
        #======================================

        metaname = os.path.join(self.root_dir, "train.pkl")
        meta = pickle.load(open(metaname, "rb"))
        
        """Load data using multiprocessing"""
        manager = Manager()
        meta = manager.list(meta)
        dataset = manager.list(len(meta)*[None])  
        processes = []
        for i in range(0, len(meta), self.step):
            p = Process(target=self.load_data, 
                        args=(meta[i:i+self.step],dataset,i))  
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
            
        self.train_dataset = list(dataset)
        self.num_tokens = len(self.train_dataset)
        
        print('Finished loading the dataset...')
        
        
    def load_data(self, submeta, dataset, idx_offset):  
        for k, sbmt in enumerate(submeta):    
            uttrs = len(sbmt)*[None]
            for j, tmp in enumerate(sbmt):
                if j < 2:  # fill in speaker id and embedding
                    uttrs[j] = tmp
                else: # load the mel-spectrograms
                    uttrs[j] = np.load(os.path.join(self.root_dir, tmp))
            dataset[idx_offset+k] = uttrs


        
        
    def __getitem__(self, index):
        # pick a random speaker
        dataset = self.train_dataset 
        list_uttrs = dataset[index]
        emb_org = list_uttrs[1]
        
        # pick random uttr with random crop
        a = np.random.randint(2, len(list_uttrs))
        tmp = list_uttrs[a]
        
        len_crop = self.len_crop
        if self.crop_range is not None: #if len_crop range is specified, use that instead
            len_crop = np.random.randint(*self.crop_range)


        if tmp.shape[0] < len_crop:
            len_pad = len_crop - tmp.shape[0]
            uttr = np.pad(tmp, ((0,len_pad),(0,0)), 'constant')
        elif tmp.shape[0] > len_crop:
            left = np.random.randint(tmp.shape[0]-len_crop)
            uttr = tmp[left:left+len_crop, :]
        else:
            uttr = tmp
        
        return uttr, emb_org
    

    def __len__(self):
        """Return the number of spkrs."""
        return self.num_tokens
    
    
    

def get_loader(root_dir, batch_size=16, len_crop=128, num_workers=0, crop_range = None):
    """Build and return a data loader."""
    
    dataset = Utterances(root_dir, len_crop, crop_range)
    
    worker_init_fn = lambda x: np.random.seed((torch.initial_seed()) % (2**32))
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  worker_init_fn=worker_init_fn)
    return data_loader






