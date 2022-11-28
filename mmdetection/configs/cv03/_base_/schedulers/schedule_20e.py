# optimizer
optimizer = dict(type='Adam', lr=0.0001, weight_decay=0.05)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-6)
runner = dict(type='EpochBasedRunner', max_epochs=20)
