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
    

    # Train Dataset
    seed = 42
    kul_data_path = 'data/KUL/keypoints/predictions'
    kul_label_path = 'data/KUL_easy/annotation/annotations.json'
    kul_video_path = 'data/KUL_easy/images'
    kul_image_path = 'data/KUL/keypoints/visualizations'
    kul_cache_dir = f'{save_dir}/cache/kul'

    cauca_data_path = 'data/CAUCAFall/keypoints/predictions'
    cauca_label_path = 'data/CAUCAFall/annotation'
    cauca_video_path = 'data/CAUCAFall/images'
    cauca_image_path = 'data/CAUCAFall/keypoints/visualizations'
    cauca_cache_dir = f'{save_dir}/cache/cauca'

    train_cam = ['cam1', 'cam2', 'cam3']
    test_cam = ['cam4', 'cam5']
    n_frames = 10
    train_split = 0.7
    val_split = 0.3

    # Transformation
    rotation_range=(-20, 20)
    scale_range=(0.8, 1.2)
    translation_range=(-0.1, 0.1)
    shear_range=(-10, 10)
    flip_prob=0.5
    kul_wh = [800,480]
    cauca_wh = [720,480]

    # Test Dataset
    # test_data_path = 'data/URFall/keypoints/predictions'
    # test_label_path = 'data/URFall/annotation/annotations.csv'
    # test_video_path = 'data/URFall/images'
    # test_image_path = 'data/URFall/keypoints/visualizations'

    # Model
    n_keypoints = 15
    n_classes = 2
    hidden_size = [256]
    dropout = 0.5

    # Training
    loss_function = 'focal'
    gamma = 4
    # alpha = [0.3, 0.6]
    alpha = None

    optimizer = 'adam'
    scheduler = 'plateau'
    learning_rate = 1e-4
    batch_size = 64
    n_epochs = 200
