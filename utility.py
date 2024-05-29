import os
from PIL import Image, ImageDraw, ImageFont

import torch
from torch.utils.data import DataLoader, random_split

def split_train_val_dataloader(dataset, train_split, val_split, batch_size=32, seed=42):
    torch.manual_seed(seed)

    # Compute size of each split
    total_size = len(dataset)
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)

    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
    
def save_images(image_paths, label, prediction, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for j, frame in enumerate(image_paths):
        for i, path in enumerate(frame):
            if label[i] == prediction[i]:
                continue
            img = Image.open(path).convert('RGB')

            draw = ImageDraw.Draw(img)
            font = ImageFont.load_default()

            text = f"Label: {label[i]} Prediction: {prediction[i]}"
            draw.text((16, 16), text, font=font, fill="white")

            img.save(os.path.join(output_dir, os.path.basename(path)))

if __name__ == '__main__':
    pass