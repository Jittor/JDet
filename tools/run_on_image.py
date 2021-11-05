import argparse
import os

from jdet.runner import Runner 
from jdet.config import init_cfg
from jdet.utils.general import list_images
        

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
        "--images",
        default="",
        help="img_dir,images,image",
        type=str,
    )

    parser.add_argument(
        "--save_dir",
        default="",
        type=str,
    )

    args = parser.parse_args()

    images = list_images(args.images)
    assert len(images)==0, f"theres is not images"

    if args.config_file:
        init_cfg(args.config_file)
    
    runner = Runner()
    runner.run_on_images(images,args.save_dir)

if __name__ == "__main__":
    main()
