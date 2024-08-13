from mmpose.apis import inference_topdown, init_model
import time
import glob
import os

# config
model_cfg = '../mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py'
ckpt = 'mmpose/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth'
device = 'cuda'

# init model
inferencer = init_model(
    model_cfg,
    ckpt,
    device=device
)

# load images
video_name = 'data/CAUCAFall/*/*/*.png'
images = glob.glob(video_name)

# inference on a single image
out_dir = f'data/CAUCAFall/keypoints'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
result_generator = inferencer(images, out_dir=out_dir, device=device)
for result in result_generator:
    pass

# # inference time
# warm_up = 50
# test = 100
# img_path = 'data/KUL/images/ADL1-2_Cam1/ADL1-2_Cam1_008.png'

# for _ in range(warm_up):
#     batch_results = inference_topdown(inferencer, img_path)

# start = time.time()
# for _ in range(test):
#     batch_results = inference_topdown(inferencer, img_path)
# end = time.time()
# print(f'The average time is {(end-start)*1e3/test:.4f}')