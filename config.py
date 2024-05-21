import os
from datetime import datetime

class MLPConfig():
    # Save Path
    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    save_dir = f'runs/model_{start_time}'
    tensorboard_dir = f'{save_dir}/tensorboard'
    output_dir = f'{save_dir}/output'
    log_path = f'{save_dir}/training.log'
    model_path = f'{save_dir}/best_model.pth'
    csv_path = f'{save_dir}/results.csv'
    cache_dir = f'{save_dir}/cache'

    # Dataset
    seed = 42
    data_path = 'data/URFall/keypoints/predictions/'
    label_path = 'data/URFall/annotation/urfall-cam0-falls.csv'
    video_names = os.listdir('data/URFall/images')
    n_frames = 5
    train_split = 0.7
    val_split = 0.3
    test_split = None

    # Model
    hidden_size = [256, ]

    # Training
    loss_function = 'ce'
    optimizer = 'adam'
    scheduler = 'plateau'
    learning_rate = 1e-3
    batch_size = 128
    n_epochs = 500
