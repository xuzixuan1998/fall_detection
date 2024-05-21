from torch.utils.data import Dataset

import glob
import pandas as pd

class BaseDataset(Dataset):
    def __init__(self, data_path, label_path, save_path, video_names, n_frames, wh=[640,480], device='cuda', header=None):
        self.data_path = data_path
        self.label_path = label_path
        self.save_path = save_path
        self.n_frames = n_frames
        self.wh = wh
        self.device = device

        # Create valid_idx
        self.valid_idx = []
        self.data_files = []
        self.label = pd.read_csv(label_path, header=header)
        
        next_start = 0
        for video_name in video_names:
            data_files = sorted(glob.glob(f'{data_path}/{video_name}*'))
            self.data_files.extend(data_files)
            self.valid_idx.extend(list(range(next_start, next_start+len(data_files)-n_frames+1)))
            next_start += len(data_files)

    def __len__(self):
        return len(self.valid_idx)
    
    # Override this function
    def __getitem__(self, idx):
        pass