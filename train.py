import os
import logging
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from configs import MLPConfig
from models import FallDetectionMLP
from models.loss import FocalLoss
from datasets import URFallDataset, KULDataset
from utility import split_train_val_dataloader, save_images

import pdb

def train_loop(model, train_loader, optimizer, criterion):
    model.train()  # Set the model to training mode
    total_loss = 0

    for batch in train_loader:
        # Forward pass
        outputs = model(batch['data'])
        loss = criterion(outputs, batch['label'])

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def evaluate(model, val_loader, criterion, name='val', save_image=False, output_dir=None):
    model.eval()  # Set the model to evaluation mode

    y_pred = []
    y_true = []
    metrics = {}
    total_loss = 0.
    with torch.no_grad():  # Disable gradient computation
        for batch in val_loader:
            outputs = model(batch['data'])
            loss = criterion(outputs, batch['label'])
            total_loss += loss.item()

            pred = torch.argmax(outputs, dim=1) - 1
            true = torch.argmax(batch['label'], dim=1) - 1
            y_pred.extend(pred.cpu().numpy())
            y_true.extend(true.cpu().numpy())

            # Print wrongly classified images
            if save_image:
                save_images(batch['image_paths'], true, pred, output_dir=f'{output_dir}/{name}')

    # Compute some metrics
    metrics['dataset'] = name
    metrics['loss'] = total_loss / len(val_loader)
    metrics['accuracy'] = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[-1, 0, 1], average=None)
    for i in range(len(precision)):
        metrics[f'precision/{i-1}'] = precision[i]
        metrics[f'recall/{i-1}'] = recall[i]
        metrics[f'f1/{i-1}'] = f1[i]
    
    return metrics

def train(model, train_loader, val_loader, test_loader, config, device):
    # TensorBoard setup
    os.makedirs(config.tensorboard_dir)
    writer = SummaryWriter(log_dir=config.tensorboard_dir)
    best_val_loss = float('inf')

    # Configure the logger
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(config.log_path),
                            logging.StreamHandler()
                        ])

    logger = logging.getLogger(__name__)

    # Print hyperparamers
    hyperparameters = {
    'learning_rate': config.learning_rate,
    'batch_size': config.batch_size,
    'epochs': config.n_epochs,
    'n_frames': config.n_frames,
    'optimizer': config.optimizer,
    'loss function': config.loss_function,
    'gamma': config.gamma,
    'alpha': config.alpha,
    'scheduler': config.scheduler
    }
    logger.info(hyperparameters)
    logger.info(model)

    # Loss function and optimizer
    model.to(device)
    if config.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    if config.loss_function == 'ce':
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(config.alpha).to(device))
    elif config.loss_function == 'focal':
        criterion = FocalLoss(gamma=config.gamma, alpha=config.alpha)
    if config.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=40, threshold=1e-7)

    logger.info('=======================================Start training=======================================')
    for epoch in range(config.n_epochs):
        train_loss = train_loop(model, train_loader, optimizer, criterion)
        val_metrics = evaluate(model, val_loader, criterion)

        # Adjust learning rate
        scheduler.step(val_metrics['loss'])

        # Logging to TensorBoard
        writer.add_scalar('lr', scheduler.get_last_lr()[-1], epoch)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)

        for i in range(3):
            writer.add_scalar(f'Precision/{i-1}', val_metrics[f'precision/{i-1}'], epoch)
            writer.add_scalar(f'Recall/{i-1}', val_metrics[f'recall/{i-1}'], epoch)
            writer.add_scalar(f'F1/{i-1}', val_metrics[f'f1/{i-1}'], epoch)

        logger.info(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_metrics['loss']:.4f}")

        # Save the best model and metrics
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save(model.state_dict(), config.model_path)
            logger.info(f"Saved better model!")

    logger.info('=======================================End training=======================================')

    # Save train/val/test set metrics to CSV
    model.load_state_dict(torch.load(config.model_path))
    train_metrics = evaluate(model, train_loader, criterion, name='train')
    val_metrics = evaluate(model, val_loader, criterion, name='val', save_image=True, output_dir=config.output_dir)
    test_metrics = evaluate(model, test_loader, criterion, name='test', save_image=True, output_dir=config.output_dir)
    df = pd.DataFrame([train_metrics, val_metrics, test_metrics])
    df.to_csv(config.csv_path, index=False)

    writer.close()

if __name__ == '__main__':
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

    # Create save_dir
    config = MLPConfig()
    
    # Some customized components
    dataset = KULDataset(config.data_path, config.label_path, config.video_path, config.n_frames, device=device, save=True, save_path=config.cache_dir)
    model = FallDetectionMLP(n_frames=config.n_frames, hidden_size=config.hidden_size)

    # Split training data and create dataloader
    train_loader, val_loader = split_train_val_dataloader(dataset, config.train_split, config.val_split, batch_size=config.batch_size, seed=config.seed)

    # Create test loader
    test_dataset = URFallDataset(config.test_data_path, config.test_label_path, config.test_video_path, config.n_frames, device=device)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # Train
    train(model, train_loader, val_loader, test_loader, config, device)