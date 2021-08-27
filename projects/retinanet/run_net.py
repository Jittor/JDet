import jittor as jt
jt.set_global_seed(233 + jt.rank * 2591) # TODO:random seed
import argparse
import os
from jdet.runner import Runner 
from jdet.config import init_cfg, get_cfg
import models
import jdet

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
        "--no_cuda",
        action='store_true'
    )
    
    args = parser.parse_args()

    if not args.no_cuda:
        jt.flags.use_cuda=1

    assert args.task in ["train","val","test","time"],f"{args.task} not support, please choose [train,val,test]"
    
    if args.config_file:
        init_cfg(args.config_file)
    
    cfg = get_cfg()

    runner = Runner()

    if args.task == "train":
        runner.run()
    elif args.task == "val":
        runner.val()
    elif args.task == "test":
        runner.test()
    elif args.task == 'time':
        runner.test_time()

if __name__ == "__main__":
    main()