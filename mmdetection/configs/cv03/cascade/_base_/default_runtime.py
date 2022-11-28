import wandb

wandb.login()
<<<<<<< HEAD:mmdetection/configs/cv03/cascade/_base_/default_runtime.py

=======
# wandb.init(project="aistages_level2_competition", entity="level2_3")
>>>>>>> b2855873ead6b8dd8a18aa2961c05423c430bfdd:mmdetection/configs/cv03/_base_/default_runtime.py
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),  
        dict(type='WandbLoggerHook',interval=10,
            init_kwargs=dict(
                project='Object_Detection',
                entity = 'aitech4_cv3',
                name = "cascade_swin-B_adamW_tta"),)
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
<<<<<<< HEAD:mmdetection/configs/cv03/cascade/_base_/default_runtime.py

=======
#workflow = [('train', 1), ('val', 1)]
>>>>>>> b2855873ead6b8dd8a18aa2961c05423c430bfdd:mmdetection/configs/cv03/_base_/default_runtime.py
# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=True, base_batch_size=16)
