optimizer = dict(
    type='SGD', 
    lr=0.01/4., #0.0,#0.01*(1/8.), 
    momentum=0.9, 
    weight_decay=0.0001,
    grad_clip=dict(
        max_norm=35, 
        norm_type=2))

scheduler = dict(
    type='StepLR',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    milestones=[7, 10])


logger = dict(
    type="RunLogger")

max_epoch = 12
eval_interval = 12
checkpoint_interval = 1
log_interval = 50