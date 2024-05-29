from torch.utils.data import Dataset

import os
import glob
import pandas as pd

import pdb

class BaseDataset(Dataset):
    def __init__(self, data_path, label_path, video_path, n_frames, device='cuda', save=False, save_path=None):
        self.data_path = data_path
        self.label_path = label_path
        self.video_path = video_path
        self.n_frames = n_frames
        self.device = device
        self.save = save
        self.save_path = save_path
        if save_path:
            os.makedirs(save_path)
        
        # Create valid_idx
        self.valid_idx = []
        self.data_files = []
        video_names = os.listdir(self.video_path)
        self.label = pd.read_csv(label_path, header=None)
        
        next_start = 0
        for video_name in video_names:
            data_files = sorted(glob.glob(f'{data_path}/{video_name}*'))
            self.data_files.extend(data_files)
            self.valid_idx.extend(list(range(next_start, next_start+len(data_files)-n_frames+1)))
            next_start += len(data_files)

    def __len__(self):
        return len(self.valid_idx)
    
    # Override this function
    def __getitem__(self):
        pass