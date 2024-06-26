import os
from PIL import Image, ImageDraw, ImageFont

import torch
from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt
import seaborn as sns

import pdb
def split_train_val_dataloader(dataset, train_split, val_split, batch_size=32, seed=42):
    
    torch.manual_seed(seed)
    # Compute size of each split
    total_size = len(dataset)
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)

    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader
    
def save_images(image_paths, label, prediction, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, frame in enumerate(image_paths):
        for _, path in enumerate(frame):
            if label[i] == prediction[i]:
                break
            img = Image.open(path).convert('RGB')

            draw = ImageDraw.Draw(img)
            font = ImageFont.load_default()

            text = f"Label: {label[i]} Prediction: {prediction[i]}"
            draw.text((16, 16), text, font=font, fill="white")

            img.save(os.path.join(output_dir, os.path.basename(path)))

def plot_confusion_matrix(cm, class_names):
    figure = plt.figure(figsize=(16,16))
    sns.heatmap(cm, annot=True, cmap=plt.cm.Blues, fmt='g', cbar=False, xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')
    return figure

def collate_fn(batch):
    filtered_batch = [sample for sample in batch if not sample['flag']]
    data_batch = torch.stack([sample['data'] for sample in filtered_batch])
    label_batch = torch.stack([sample['label'] for sample in filtered_batch])
    paths_batch = [sample['image_paths'] for sample in filtered_batch]
    return {'image_paths': paths_batch, 'data':data_batch, 'label':label_batch}

if __name__ == '__main__':
    pass