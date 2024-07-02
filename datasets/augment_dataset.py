from torch.utils.data import Dataset

import pdb
class AugmentDataset(Dataset):
    def __init__(self, subset, transform=None, device='cpu'):
        self.subset = subset
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        data = self.subset[idx]
        if self.transform:
            data['keypoints'] = self.transform(data['keypoints'])

        n_frames = data['keypoints'].size(0)
        data['keypoints'] = data['keypoints'].reshape(n_frames, -1).to(self.device)
        return data