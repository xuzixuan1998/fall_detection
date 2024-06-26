class EvaluateConfig():
    # Pretrained model
    model_path = '/home/ms/david/fall_detection/runs/model_2024-06-19_11-58-57/best_model.pth'

    # Dataset
    data_path = 'data/CAUCAFall/keypoints/predictions'
    label_path = 'data/CAUCAFall/annotation'
    video_path = 'data/CAUCAFall/images'
    image_path = 'data/CAUCAFall/keypoints/visualizations'
    n_frames = 10

    # Model
    n_classes = 2
    hidden_size = [2048]

    # Batch size
    batch_size = 1024