# optimizer
optimizer = dict(type='Adam', lr=0.0001, weight_decay=0.005)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    min_lr_ratio=1e-6)
runner = dict(type='EpochBasedRunner', max_epochs=24)
