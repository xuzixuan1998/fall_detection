from torch.utils.data import Dataset

import os
import glob
import json
import pandas as pd

import pdb

class BaseDataset(Dataset):
    def __init__(self, data_path, label_path, video_path, image_path, n_frames, device='cuda', save=False, save_path=None):
        self.data_path = data_path
        self.label_path = label_path
        self.video_path = video_path
        self.image_path = image_path
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
        with open(label_path, 'r') as file:
            data = json.load(file)
            self.label = data

        next_start = 0
        for video_name in video_names:
            data_files = sorted(glob.glob(f'{video_path}/{video_name}/*.png'))
            self.data_files.extend(data_files)
            self.valid_idx.extend(list(range(next_start, next_start+len(data_files)-n_frames+1)))
            next_start += len(data_files)

    def __len__(self):
        return len(self.valid_idx)
    
    # Override this function
    def __getitem__(self):
        pass