# model settings
model = dict(
    type='CenterNet',
    pretrained='',
    backbone=dict(
        type='RegMobileNetV2'),
    rpn_head=dict(
        type='CtdetHead', heads=dict(hm=1, wh=2, reg=2), head_conv=64))
cudnn_benchmark = True

train_cfg = dict(a=10)

_valid_ids = [
    1,
]
img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True)

test_cfg = dict(
    num_classes=1,
    valid_ids={i + 1: v
               for i, v in enumerate(_valid_ids)},
    img_norm_cfg=img_norm_cfg,
    debug=0)

import numpy as np
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='CtdetTrainTransforms',
            flip_ratio=0.5,
            size_divisor=31,
            keep_ratio=False,
            img_scale=(416,736),
            img_norm_cfg=img_norm_cfg,
            max_objs = 128,
            num_classes = 1,
            _data_rng = np.random.RandomState(123),
            _eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                     dtype=np.float32),
            _eig_vec = np.array([[-0.58752847, -0.69563484, 0.41340352],
                                      [-0.5832747, 0.00994535, -0.81221408],
                                      [-0.56089297, 0.71832671, 0.41158938]],
                                     dtype=np.float32)
                                    )
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1, 1),
        flip=False,
        transforms=[
            dict(type='CtdetTestTransforms',
                size_divisor=31,
                keep_ratio=False,
                input_res=(704,1248),
                img_norm_cfg=img_norm_cfg)
        ])
]


dataset_type = 'SpoilDataset'
data_root = '/data2/spoil_data/'
data = dict(
    imgs_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations.npy',
        img_prefix=data_root + '/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations.npy',
        img_prefix=data_root + '/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations.npy',
        img_prefix=data_root + '/',
        pipeline=test_pipeline))

# optimizer
optimizer = dict(type='Adam', lr=2.5e-4)
# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
optimizer_config = {}
# learning policy
lr_config = dict(
    policy='step',
    # warmup='linear',
    # warmup_iters=500,
    # warmup_ratio=1.0 / 3,
    step=[18, 24])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=2,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 28
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'data/work_dirs/centernet_dla_spoil'
load_from = None
resume_from = None
workflow = [('train', 1)]
