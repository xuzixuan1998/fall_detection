from torch.utils.data import Dataset
import torch

from collections import Counter
import pandas as pd
import json
import glob
import pdb
import os

class URFallDataset(Dataset):
    def __init__(self, data_path, label_path, save_path, n_frames, wh=[640,480], device='cuda', header=None):
        self.data_path = data_path
        self.label_path = label_path
        self.save_path = save_path
        self.n_frames = n_frames
        self.wh = wh
        self.device = device

        # Use a step == n_frames range() to check if the first file 
        self.valid_idx = []
        
        self.label = pd.read_csv(label_path, header=header)

    def __len__(self):
        return len(self.valid_idx)
    
    def __getitem__(self, idx):
        # If the batch data is saved
        pt_file = f'{self.save_path}/batch_{str(idx)}.pt'
        if os.path.exists(pt_file):
            return torch.load(pt_file)
        
        # Otherwise we save it 
        name = []
        data = []
        label = [0] * 3
        labels = []

        # Load json file
        for i in range(self.n_frames):
            json_file = self.data_files[self.valid_idx[idx+i]]
            with open(json_file, 'r') as file:
                json_data = json.load(file)
            data.append(json_data[0]['keypoints'])

            # Query and store labels
            file_name = os.path.basename(json_file)[:-5]
            video_type, video_idx, _, _, frame_idx = file_name.split('-')

            # Store image paths
            video_prefix = f'{video_type}-{video_idx}'
            folder_path = f'{video_prefix}-cam0-rgb'
            image_name = f'{file_name}.png'
            name.append(f'data/URFall/images/{folder_path}/{image_name}')

            if video_type == 'adl':
                labels.append(-1)
                continue
            query_result = self.label[(self.label[0] == video_prefix) & (self.label[1] == int(frame_idx))].iloc[0]
            if query_result.iloc[-1] == 1:
                labels.append(-1)
            else:
                labels.append(query_result.iloc[2])

        # Majority vote
        label_counts = Counter(labels)
        label[label_counts.most_common(1)[0][0]+1] = 1

        # Normalization
        data = torch.tensor(data, dtype=torch.float32)
        normalized_data = data / torch.tensor(self.wh).view(1, 1, 2)

        # Save pt file
        data_dict = {'image_paths': name, 'data':normalized_data.view(self.n_frames,-1).to(self.device), 'label':torch.tensor(label, dtype=torch.float32).to(self.device)}
        torch.save(data_dict, pt_file)
        return data_dict

if __name__ == '__main__':
    data_path = 'data/URFall/keypoints/predictions/'
    label_path = 'data/URFall/annotation/urfall-cam0-falls.csv'
    n_frames = 5

    dataset = URFallDataset(data_path, label_path, '.', n_frames)
    for data in dataset:
        pass