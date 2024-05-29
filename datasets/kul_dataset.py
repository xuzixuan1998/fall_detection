from .base_dataset import BaseDataset

import torch

from collections import Counter
import json
import pdb
import os

class KULDataset(BaseDataset):

    def __getitem__(self, idx):
        # If the batch data is saved
        if self.save_path:
            pt_file = f'{self.save_path}/batch_{str(idx)}.pt'
            if os.path.exists(pt_file):
                return torch.load(pt_file)
                
        data = []
        paths = []
        labels = []

        # Load json file
        idx = self.valid_idx[idx]
        for i in range(self.n_frames):
            json_file = self.data_files[idx+i]
            with open(json_file, 'r') as file:
                json_data = json.load(file)

            # Normalization
            keypoints = torch.tensor(json_data[0]['keypoints'])
            bbox = torch.tensor(json_data[0]['bbox'])
            xy, wh = bbox[:,:2], bbox[:,2:]-bbox[:,:2]
            data.append((keypoints-xy)/wh)

            # Query and store labels
            file_name = os.path.basename(json_file)[:-5]
            video_name, cam_name, frame_idx = file_name.split('_')

            # Store image paths
            image_name = f'{file_name}.png'
            paths.append(f'{self.video_path}/{video_name}_{cam_name}/{image_name}')

            # Search label
            query_result = self.label[(self.label[0] == video_name) & (self.label[1] == int(frame_idx))].iloc[0]
            labels.append(query_result.iloc[2])

        # Majority vote
        label = [0] * 3
        label_counts = Counter(labels)
        label[label_counts.most_common(1)[0][0]+1] = 1

        # Save pt file
        data_dict = {'image_paths': paths, 'data':torch.stack(data).view(self.n_frames,-1).to(self.device), 'label':torch.tensor(label, dtype=torch.float32).to(self.device)}
        if self.save:
            torch.save(data_dict, pt_file)
        return data_dict
    
if __name__ == '__main__':
    data_path = '../data/KUL/keypoints/predictions/'
    label_path = '../data/KUL/annotation/annotations.csv'
    video_path = '../data/KUL/images'
    n_frames = 5

    dataset = KULDataset(data_path, label_path, video_path, n_frames)
    for data in dataset:
        pass 