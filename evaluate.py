from tqdm import tqdm 

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from models import FallDetectionMLP
from datasets import CAUCADataset
from configs import EvaluateConfig
from utility import collate_fn

def evaluate(model, loader):
    model.eval() 

    y_pred = []
    y_true = []
    metrics = {}

    with torch.no_grad():  # Disable gradient computation
        for batch in tqdm(loader, desc='Evaluating', unit='batch'):
            outputs = model(batch['data'])

            pred = torch.argmax(outputs, dim=1)
            true = torch.argmax(batch['label'], dim=1)
            y_pred.extend(pred.cpu().numpy())
            y_true.extend(true.cpu().numpy())

    # Compute some metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0,1], average=None)
    for i in range(len(precision)):
        metrics[f'precision/{i}'] = precision[i]
        metrics[f'recall/{i}'] = recall[i]
        metrics[f'f1/{i}'] = f1[i]
    
    return metrics, cm    

if __name__ == '__main__':
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

    # Config
    config = EvaluateConfig()
    
    # Model
    model = FallDetectionMLP(n_frames=config.n_frames, n_classes=config.n_classes, hidden_size=config.hidden_size).to(device)
    model.load_state_dict(torch.load(config.model_path))
    
    # Dataset
    dataset = CAUCADataset(config.data_path, config.label_path, config.video_path, config.image_path, config.n_frames)

    # DataLoader
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

    # Evaluate
    metrics, cm = evaluate(model, loader)

    print(metrics)
    print(cm)