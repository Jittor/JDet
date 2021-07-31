_base_ = 'retinanet.py'
work_dir = "./exp/retinanet_21"
pretrained_weights="weights/yx_init_pretrained.pk_jt.pk"
max_epoch = 30

scheduler = dict(
    milestones= [27])

optimizer = dict(
    lr=3 * 5e-4
)

# flip_test=['H', 'V', 'HV']
checkpoint_interval = 30
dataset = dict(
    val=dict(
        dataset_dir="/home/cxjyxx_me/workspace/JAD/datasets/test/processed_DOTA/trainval_600_150_1.0-1.5",
        annotations_file=None,
        images_dir=None,
    ),
    train=dict(
        dataset_dir="/home/cxjyxx_me/workspace/JAD/datasets/test/processed_DOTA/trainval_600_150_1.0-1.5",
        annotations_file=None,
        images_dir=None,
    ),
    test = dict(
      images_dir= "/home/cxjyxx_me/workspace/JAD/datasets/test/processed_DOTA/test_600_150_1.0/images/")
)
resume_path="/home/cxjyxx_me/workspace/JAD/JDet/projects/retinanet/work_dirs/retinanet_21/checkpoints/ckpt_30.pkl"