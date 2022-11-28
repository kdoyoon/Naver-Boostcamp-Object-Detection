# dataset settings
dataset_type = 'CocoDataset'
data_root='/opt/ml/dataset/' 
classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing") ## class 정의
img_scale=(1024,1024)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[[
                dict(
                    type='Resize',
                    img_scale=[(1024, 1024), (512, 512), (256, 256), (128, 128)],
                    multiscale_mode='value',
                    keep_ratio=True),
                    dict(type='BrightnessTransform', prob=0.8, level=10),

                ],
                [
                    dict(type='Shear',prob=0.5,level=10),
                    dict(type='EqualizeTransform',prob=0.5)

                ],
                [
                dict(

                    type='Resize',
                    img_scale=[(1024, 1024), (512, 512), (256, 256)],
                    multiscale_mode='value',
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(512, 512),
                    allow_negative_crop=True),
                dict(
                    type='Resize',
                    img_scale=[(1024, 1024), (512, 512), (256, 256), (128, 128)],
                    multiscale_mode='value',
                    override=True,
                    keep_ratio=True)
                ]]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# train_dataset = dict(
#     #_delete_ = True, # remove unnecessary Settings
#     type='MultiImageMixDataset',
#     dataset=dict(
#         type=dataset_type,
#         ann_file=data_root + 'custom/train.json',
#         img_prefix=data_root,
#         classes= classes,
#         pipeline=[
#             dict(type='LoadImageFromFile'),
#             dict(type='LoadAnnotations', with_bbox=True)
#         ],
#         filter_empty_gt=False,
#     ),
#     pipeline=train_pipeline
#     )

data = dict(
    samples_per_gpu=2, # gpu 하나 당 올라갈 이미지 수
    workers_per_gpu=2, # gpu 하나 당 cpu코어 수
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'custom/train.json',
        img_prefix=data_root,
        classes = classes,
        pipeline=train_pipeline),
    val=dict(
    type=dataset_type,
        ann_file=data_root + 'custom/valid.json',
        img_prefix=data_root,
        classes = classes,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test.json',
        img_prefix=data_root,
        classes = classes,
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
