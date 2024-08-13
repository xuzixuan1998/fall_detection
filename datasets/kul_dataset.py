from .base_dataset import BaseDataset
from .transform import KeypointTransform

import torch

import shutil
import json
import pdb
import os

class KULDataset(BaseDataset):

    def __getitem__(self, idx):
        # If the batch data is saved, also do data augmentation
        if self.save_path:
            pt_file = f'{self.save_path}/batch_{str(idx)}.pt'
            if os.path.exists(pt_file):
                data = torch.load(pt_file)
                return data
                
        data = []
        paths = []
        labels = []

        # Load json file
        start_idx = self.valid_idx[idx]
        for i in range(self.n_frames):
            image_file = self.data_files[start_idx+i]

            json_file = os.path.join(self.data_path, os.path.basename(image_file).replace('.png', '.json'))
            with open(json_file, 'r') as file:
                json_data = json.load(file)

            # Load keypoints
            keypoints = torch.tensor(json_data[0]['keypoints']).t()         
            data.append(keypoints)

            # Normalization
            # bbox = torch.tensor(json_data[0]['bbox'])
            # xy, wh = bbox[:,:2], bbox[:,2:]-bbox[:,:2]
            # data.append((keypoints-xy.t())/wh.t())

            # Query and store labels
            image_name = os.path.basename(image_file)
            video_name, cam_name, _ = image_name.split('_')
            video_name = f'{video_name}_{cam_name}'

            # Store image paths
            paths.append(os.path.join(self.image_path, image_name))

            # Search label
            labels.append(self.label[video_name][image_name])

        # Label clip
        label = [0] * 2
        if labels[0] == False and labels[-1] == True:
            label[1] = 1
        else:
            label[0] = 1

        # Initiation 
        data = torch.stack(data) / self.wh

        # Remove keypoints 
        remove_idx = [15,16]
        indices = torch.tensor([i for i in range(data.shape[-1]) if i not in remove_idx])
        data = torch.index_select(data, dim=-1, index=indices)

        # data = torch.stack(data)
        label = torch.tensor(label, dtype=torch.float32).to(self.device)
        flag = (labels[0] == None or labels[-1] == None)

        # Save pt file
        data_dict = {'image_paths': paths, 'keypoints':data, 'label':label, 'flag':flag}
        if self.save:
            torch.save(data_dict, pt_file)
        return data_dict
    
    def get_indices_by_camera(self, camera_ids):
        indices = [] 
        for i, idx in enumerate(self.valid_idx):
            json_file = self.data_files[idx]
            _, cam_name, _ = os.path.basename(json_file).split('_')
            if cam_name.lower() in camera_ids:
                indices.append(i)
        return indices
    
if __name__ == '__main__':
    data_path = 'data/KUL/keypoints/predictions/'
    label_path = 'data/KUL_easy/annotation/annotations.json'
    video_path = 'data/KUL_easy/images'
    image_path = 'data/KUL/keypoints/visualizations/'
    n_frames = 5

    # Transformation
    transform = KeypointTransform(
    rotation_range=(-15, 15),
    scale_range=(0.8, 1.2),
    translation_range=(-0.1,0.1),
    shear_range=(-10,10),
    flip_prob=0.5,
    wh = torch.tensor([800,480],dtype=torch.float32)
)

    dataset = KULDataset(data_path, label_path, video_path, image_path, n_frames, transform=transform)
    for data in dataset:
        pass 