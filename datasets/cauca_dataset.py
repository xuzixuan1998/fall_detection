from .base_dataset import BaseDataset
from .transform import KeypointTransform

import os
import json
import glob
import torch
import shutil

import pdb

class CAUCADataset(BaseDataset):
    def __init__(self, data_path, label_path, video_path, image_path, n_frames, wh, fps=5, device='cuda', save=False, save_path=None):

        self.data_path = data_path
        self.label_path = label_path
        self.video_path = video_path
        self.image_path = image_path
        self.n_frames = n_frames
        self.wh = torch.tensor(wh, dtype=torch.float32).view(1,2,1)
        self.device = device
        self.save = save
        self.save_path = save_path
        step = 20//fps

        if save_path:
            os.makedirs(save_path)
        
        # Create valid_idx
        self.valid_idx = []
        self.data_files = []
        self.label = {}
        subject_names = os.listdir(self.video_path)

        for subject_name in subject_names:
            # Load annotations
            with open(os.path.join(label_path, f'{subject_name}.json'), 'r') as file:
                data = json.load(file)
                self.label[subject_name] = data

            video_names = os.listdir(os.path.join(self.video_path, subject_name))
            next_start = 0
            for video_name in video_names:
                data_files = sorted(glob.glob(f'{video_path}/{subject_name}/{video_name}/*.png'))[::step]
                self.data_files.extend(data_files)
                self.valid_idx.extend(list(range(next_start, next_start+len(data_files)-n_frames+1)))
                next_start += len(data_files)

    def __getitem__(self, idx):
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
            image_path = self.data_files[start_idx+i]
            subject_name, video_name, image_name = image_path.split('/')[-3:]
    
            json_name = image_name.replace('.png', '.json')
            json_file = os.path.join(self.data_path, json_name)
            with open(json_file, 'r') as file:
                json_data = json.load(file)
            
            # Add keypoints
            keypoints = torch.tensor(json_data[0]['keypoints']).t()
            data.append(keypoints)

            # Normaliztion
            # bbox = torch.tensor(json_data[0]['bbox'])
            # xy, wh = bbox[:,:2], bbox[:,2:]-bbox[:,:2]
            # data.append((keypoints-xy.t())/wh.t())

            # Add image paths
            paths.append(os.path.join(self.image_path, image_name))

            # Add label
            labels.append(self.label[subject_name][video_name][image_name])
        
        # Label clip
        label = [0] * 2
        if labels[0] == False and labels[-1] == True:
            label[1] = 1
        else:
            label[0] = 1

        # keypoints difference
        data = torch.stack(data) / self.wh
        label = torch.tensor(label, dtype=torch.float32).to(self.device)
        flag = (labels[0] == None or labels[-1] == None)

        # Save pt file
        data_dict = {'image_paths': paths, 'keypoints':data, 'label':label, 'flag':flag}
        if self.save:
            torch.save(data_dict, pt_file)
        return data_dict
             
if __name__ == '__main__':
    data_path = 'data/CAUCAFall/keypoints/predictions/'
    label_path = 'data/CAUCAFall/annotation/'
    video_path = 'data/CAUCAFall/images'
    image_path = 'data/CAUCAFall/keypoints/visualizations/'
    n_frames = 5

    # Transformation
    transform = KeypointTransform(
    rotation_range=(-15, 15),
    scale_range=(0.8, 1.2),
    translation_range=(-0.1,0.1),
    shear_range=(-10,10),
    flip_prob=0.5,
    wh = torch.tensor([720,480],dtype=torch.float32)
)

    dataset = CAUCADataset(data_path, label_path, video_path, image_path, n_frames, wh=[720,480], transform=transform)

    for data in dataset:
        pass 