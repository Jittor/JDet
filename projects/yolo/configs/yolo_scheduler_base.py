scheduler=dict(
    type='CosineAnnealingLRGroup',
    min_lr_ratio=0.2, # hyp[lrf]
    warmup_init_lr_pg=[0., 0., 0.1], #[pg0, pg1, pg2]
    warmup_ratio = 0.,
    warmup_initial_momentum = 0.8, #hyp[warmup_momentum]
    warmup = 'linear',
    warmup_iters= 1000 # 3 epochs 
)