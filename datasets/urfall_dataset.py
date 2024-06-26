from .base_dataset import BaseDataset

import torch

import json
import pdb
import os

class URFallDataset(BaseDataset):

    def __getitem__(self, idx):
        # If the batch data is saved
        if self.save:
            pt_file = f'{self.save_path}/batch_{str(idx)}.pt'
            if os.path.exists(pt_file):
                return torch.load(pt_file)
        
        # Load n frames
        data = []
        paths = []
        labels = []

        # Load json file
        start_idx = self.valid_idx[idx]
        for i in range(self.n_frames):
            json_file = self.data_files[start_idx+i]
            with open(json_file, 'r') as file:
                json_data = json.load(file)

            # Normalization
            keypoints = torch.tensor(json_data[0]['keypoints'])
            bbox = torch.tensor(json_data[0]['bbox'])
            xy, wh = bbox[:,:2], bbox[:,2:]-bbox[:,:2]
            data.append((keypoints-xy)/wh)
            # data.append(keypoints/torch.tensor([320, 240]))

            # Query and store labels
            file_name = os.path.basename(json_file)[:-5]
            video_type, video_idx, cam_idx, frame_idx = file_name.split('-')

            # Store image paths
            video_name = f'{video_type}-{video_idx}'
            folder_path = f'{video_name}-{cam_idx}'
            image_name = f'{file_name}.png'
            paths.append(f'{self.image_path}/{image_name}')

            query_result = self.label[(self.label[0] == video_name) & (self.label[1] == int(frame_idx))].iloc[0]
            label = query_result.iloc[2]
            labels.append(label)
        
        # Label clip
        label = [0] * 2
        if labels[0] == -1 and labels[-1] == 1:
            label[1] = 1
        else:
            label[0] = 1

        # keypoints difference
        data = torch.stack(data).view(self.n_frames,-1).to(self.device)
        label = torch.tensor(label, dtype=torch.float32).to(self.device)
        flag = (labels[0] == 0 or labels[-1] == 0)

        # Save pt file
        data_dict = {'image_paths': paths, 'data':data, 'label':label, 'flag':flag}
        if self.save:
            torch.save(data_dict, pt_file)
        return data_dict

if __name__ == '__main__':
    data_path = '../data/URFall/keypoints/predictions/'
    label_path = '../data/URFall/annotation/urfall-cam0-falls.csv'
    video_path = '../data/URFall/images'
    n_frames = 5

    dataset = URFallDataset(data_path, label_path, video_path, n_frames)
    for data in dataset:
        pass