import argparse
import os
import jittor as jt
from jdet.runner import Runner 
from jdet.config import init_cfg
import models

def main():
    parser = argparse.ArgumentParser(description="Jittor Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--task",
        default="train",
        help="train,val,test",
        type=str,
    )
    parser.add_argument(
        "--use_cuda",
        action='store_true'
    )
    
    args = parser.parse_args()

    if args.use_cuda:
        jt.flags.use_cuda=1

    assert args.task in ["train","val","test"],f"{args.task} not support, please choose [train,val,test]"
    
    if args.config_file:
        init_cfg(args.config_file)
    
    runner = Runner()

    if args.task == "train":
        # runner.model.load("weights/RetinaNet_DOTA_2x_20200915_DOTA_702000model.pk_jt.pk")
        runner.run()
    elif args.task == "val":
        runner.val()
    elif args.task == "test":
        runner.model.load("weights/RetinaNet_DOTA_2x_20200915_DOTA_702000model.pk_jt.pk")
        # runner.load("exps/jittor_train/retinanet/checkpoints/ckpt_26.pkl")
        runner.test()

if __name__ == "__main__":
    main()