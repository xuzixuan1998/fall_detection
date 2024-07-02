import torch
import numpy as np

import pdb

class KeypointTransform:
    def __init__(self, rotation_range, scale_range, translation_range, shear_range, flip_prob):
        self.rotation_range = rotation_range  # degrees
        self.scale_range = scale_range
        self.translation_range = translation_range
        self.shear_range = shear_range        
        self.flip_prob = flip_prob
 
        self.center = torch.tensor([0.5, 0.5], dtype=torch.float32).view(1,2,1)

    def rotate(self, keypoints, angle):
        theta = np.radians(angle)
        R = torch.tensor([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ]).to(torch.float32)
        rotated_keypoints = torch.einsum('ij,ajk->aik', R, keypoints - self.center) + self.center
 
        return rotated_keypoints
    
    def scale(self, keypoints, scale_factor):
        scaled_keypoints = (keypoints - self.center) * scale_factor + self.center

        return scaled_keypoints
    
    def translate(self, keypoints, translation):
        translation = translation.view(1,2,1)
        translated_keypoints = keypoints + translation
        
        return translated_keypoints
    
    def shear(self, keypoints, angle):
        theta = np.radians(angle)
        S = torch.tensor([
                [1, np.tan(theta)],
                [0, 1],
            ]).to(torch.float32)
        sheared_keypoints = torch.einsum('ij,ajk->aik', S, keypoints - self.center) + self.center
        
        return sheared_keypoints

    def __call__(self, keypoints):
        # Affine Transformation        
        original_keypoints = keypoints.clone()
        rotation_angle = np.random.uniform(*self.rotation_range)
        keypoints = self.rotate(keypoints, rotation_angle)
        if torch.any(keypoints < 0) or torch.any(keypoints > 1):
            keypoints = original_keypoints

        original_keypoints = keypoints.clone()
        scale_factor = np.random.uniform(*self.scale_range)
        keypoints = self.scale(keypoints, scale_factor)
        if torch.any(keypoints < 0) or torch.any(keypoints > 1):
            keypoints = original_keypoints

        original_keypoints = keypoints.clone()
        translation = torch.tensor([np.random.uniform(*self.translation_range), np.random.uniform(*self.translation_range)])
        keypoints = self.translate(keypoints, translation)
        if torch.any(keypoints < 0) or torch.any(keypoints > 1):
            keypoints = original_keypoints

        original_keypoints = keypoints.clone()
        shear_angle = np.random.uniform(*self.shear_range)
        keypoints = self.shear(keypoints, shear_angle)
        if torch.any(keypoints < 0) or torch.any(keypoints > 1):
            keypoints = original_keypoints

        # Horizontal Flip
        if np.random.rand() < self.flip_prob:
            keypoints[:, 0, :] = 1 - keypoints[:, 0, :]

        return keypoints