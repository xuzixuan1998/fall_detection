from mmpose.apis import MMPoseInferencer
import glob
import os

# config
model_cfg = '../mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py'
ckpt = 'mmpose/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth'
device = 'cuda'

# init model
inferencer = MMPoseInferencer(
    pose2d=model_cfg,
    pose2d_weights=ckpt
)

# load images
video_name = 'data/URFall/images/*/*.png'
images = glob.glob(video_name)

# inference on a single image
out_dir = f'data/URFall/keypoints'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
result_generator = inferencer(images, out_dir=out_dir, device=device)
for result in result_generator:
    pass
