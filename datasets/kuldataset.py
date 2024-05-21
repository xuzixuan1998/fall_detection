from basedataset import BaseDataset

import torch

from collections import Counter
import json
import pdb
import os

class URFallDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def __getitem__(self, idx):
        # If the batch data is saved
        pt_file = f'{self.save_path}/batch_{str(idx)}.pt'
        if os.path.exists(pt_file):
            return torch.load(pt_file)
        
        # Otherwise we save it 
        name = []
        data = []
        labels = []
        label = [0] * 3

        # Load json file
        idx = self.valid_idx[idx]
        for i in range(self.n_frames):
            json_file = self.data_files[idx+i]
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


            query_result = self.label[(self.label[0] == video_prefix) & (self.label[1] == int(frame_idx))].iloc[0]
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