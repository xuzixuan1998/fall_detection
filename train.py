import os
import logging
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from config import MLPConfig
from model import FallDetectionMLPV2
from datasets import URFallDataset
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

    metrics = {}
    y_pred = []
    y_true = []
    cnt, total_loss = 0, 0
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
                n_wrong = torch.sum(pred!=true)
                save_images(batch['image_paths'], true, pred, output_dir, cnt)
                cnt += n_wrong

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

def train(model, train_loader, val_loader, config, device):
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
    'scheduler': config.scheduler
    }
    logger.info(hyperparameters)
    logger.info(model)

    # Loss function and optimizer
    model.to(device)
    if config.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    if config.loss_function == 'ce':
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 1., 1.]).to(device))
    if config.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=50, threshold=1e-7)

    # TensorBoard setup
    writer = SummaryWriter(log_dir=config.tensorboard_dir)
    best_val_loss = float('inf')

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
    # Save train/val set metrics to CSV
    model.load_state_dict(torch.load(config.model_path))
    train_metrics = evaluate(model, train_loader, criterion, name='train')
    val_metrics = evaluate(model, val_loader, criterion, save_image=True, output_dir=f'{config.output_dir}/val')
    df = pd.DataFrame([train_metrics, val_metrics])
    df.to_csv(config.csv_path, index=False)

    writer.close()

if __name__ == '__main__':
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

    # Create save_dir
    config = MLPConfig()
    os.makedirs(config.tensorboard_dir)
    os.makedirs(config.output_dir)
    os.makedirs(config.cache_dir)

    # Some customized components
    dataset = URFallDataset(config.data_path, config.label_path, config.cache_dir, config.video_names, config.n_frames, device=device)
    model = FallDetectionMLPV2(n_frames=config.n_frames, hidden_size=config.hidden_size)

    # Split data and create dataloader
    if config.test_split:
        train_loader, val_loader, test_loader = split_train_val_dataloader(dataset, config.train_split, config.val_split, config.test_split, batch_size=config.batch_size, seed=config.seed)
    else:
        train_loader, val_loader = split_train_val_dataloader(dataset, config.train_split, config.val_split, config.test_split, batch_size=config.batch_size, seed=config.seed)

    # Train
    train(model, train_loader, val_loader, config, device)

    # Evaluate 
    