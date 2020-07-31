from mmdet.apis import init_detector, inference_detector, show_result_pyplot,show_result
import mmcv
import tensorwatch as tw
import torch


config_file = './configs/centernet_mobile_1x_spoil.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = './data/work_dirs/centernet_dla_spoil/epoch_140.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')
img = './demo/1.jpg'
result = inference_detector(model, img)
res = show_result(img, result, model.CLASSES, score_thr=0.4, show =False, out_file='out.png')
