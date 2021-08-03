_base_ = 'retinanet_r50v1d_fpn_fair.py'

scheduler = dict(
    warmup_iters= 0)

dataset = dict(
    dataset_type="FAIRDataset",
    val=None,
    train=None,
    test = dict(
        images_dir= "/dataset/processed/test_600_150_1.0/images/"
    )
)
pretrained_weights=None
resume_path="/home/cxjyxx_me/workspace/JAD/JDet/projects/retinanet/work_dirs/retinanet_46/checkpoints/ckpt_30.pkl"