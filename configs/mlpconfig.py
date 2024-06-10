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

    # Train Dataset
    seed = 42
    data_path = 'data/KUL/keypoints/predictions/'
    label_path = 'data/KUL/annotation/annotations.csv'
    video_path = 'data/KUL/images'
    n_frames = 5
    train_split = 0.7
    val_split = 0.3

    # Test Dataset
    test_data_path = 'data/URFall/keypoints/predictions/'
    test_label_path = 'data/URFall/annotation/annotations.csv'
    test_video_path = 'data/URFall/images'

    # Model
    hidden_size = [256, 512]

    # Training
    # loss_function = 'ce'
    loss_function = 'focal'
    gamma = 1
    alpha = [0.2, 0.6, 0.2]

    optimizer = 'adam'
    scheduler = 'plateau'
    learning_rate = 1e-3
    batch_size = 512
    n_epochs = 500
