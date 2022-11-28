<<<<<<< HEAD
import cv2
# dataset settings
dataset_type = 'CocoDataset'
data_root='/opt/ml/dataset/' 
classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing") ## class 정의
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
=======
# dataset settings
dataset_type = 'CocoDataset'
data_root = '/opt/ml/dataset/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    # dict(type='PhotoMetricDistortion', brightness_delta=32,
    #              contrast_range=(0.5, 1.5),
    #              saturation_range=(0.5, 1.5),
    #              hue_delta=18),
>>>>>>> 1cfaeb01051a0c98f4822478d1467f1d29334eaf
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
<<<<<<< HEAD
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),]

=======
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
>>>>>>> 1cfaeb01051a0c98f4822478d1467f1d29334eaf
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
<<<<<<< HEAD
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
=======
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
>>>>>>> 1cfaeb01051a0c98f4822478d1467f1d29334eaf
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
<<<<<<< HEAD
data = dict(
    samples_per_gpu=2, # gpu 하나 당 올라갈 이미지 수
    workers_per_gpu=2, # gpu 하나 당 cpu코어 수
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'custom/train_2.json',
=======
classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file='/opt/ml/custom_dataset/train_2.json',
>>>>>>> 1cfaeb01051a0c98f4822478d1467f1d29334eaf
        img_prefix=data_root,
        classes = classes,
        pipeline=train_pipeline),
    val=dict(
<<<<<<< HEAD
    type=dataset_type,
        ann_file=data_root + 'custom/valid_2.json',
        img_prefix=data_root,
        classes = classes,
=======
        type=dataset_type,
        ann_file='/opt/ml/custom_dataset/valid_2.json',
        img_prefix=data_root,
         classes = classes,
>>>>>>> 1cfaeb01051a0c98f4822478d1467f1d29334eaf
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test.json',
        img_prefix=data_root,
<<<<<<< HEAD
        classes = classes,
=======
         classes = classes,
>>>>>>> 1cfaeb01051a0c98f4822478d1467f1d29334eaf
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
