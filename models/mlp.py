import torch
import torch.nn as nn

import pdb
import time

from configs.mlpconfig import MLPConfig

class FallDetectionMLP(nn.Module):
    def __init__(self, n_keypoints=17, n_frames=5, n_classes=3, hidden_size=[256], dropout=0.5):
        super(FallDetectionMLP, self).__init__()
        # Embed each keypoints
        # Define MLP
        self.linear = nn.Sequential()
        in_dim = n_keypoints * n_frames * 2
        for i, out_dim in enumerate(hidden_size):
            self.linear.add_module(f'linear_{i}', nn.Linear(in_dim, out_dim))
            self.linear.add_module(f'bn_{i}', nn.BatchNorm1d(out_dim))
            self.linear.add_module(f'relu_{i}', nn.ReLU())
            self.linear.add_module(f'dropout_{i}', nn.Dropout(dropout))
            in_dim = out_dim
        self.linear.add_module(f'layer_out', nn.Linear(in_dim, n_classes))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(x.shape[0],-1)  
        x = self.linear(x)  
        return self.softmax(x)
    
if __name__ == '__main__':
    config = MLPConfig()
    model = FallDetectionMLP(n_frames=config.n_frames, hidden_size=config.hidden_size)

    model.eval()
    data = torch.randn(1, 5, 34)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Inference on {device}!')
    model.to(device)
    input_data = data.to(device)

    # Warm up the GPU
    for _ in range(10):
        with torch.no_grad():
            _ = model(input_data)

    # Measure inference time
    num_runs = 100
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_data)

    end_time = time.time()

    # Calculate average inference time
    total_time = end_time - start_time
    average_time_per_inference = total_time / num_runs
    fps = 1 / average_time_per_inference
    print(f'Average inference time: {average_time_per_inference:.6f} seconds. FPS: {fps}')