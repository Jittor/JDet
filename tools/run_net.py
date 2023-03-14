import argparse
import jittor as jt
jt.flags.use_cuda_managed_allocator=1
from jdet.runner import Runner 
from jdet.config import init_cfg

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

    parser.add_argument(
        "--save_dir",
        default=".",
        type=str,
    )
    
    args = parser.parse_args()

    if not args.no_cuda:
        jt.flags.use_cuda=1

    assert args.task in ["train","val","test","vis_test"],f"{args.task} not support, please choose [train,val,test,vis_test]"
    
    if args.config_file:
        init_cfg(args.config_file)

    runner = Runner()
    # import pickle
    # state_dict = pickle.load(open("/mnt/disk/flowey/remote/JDet-debug/weights/state_dict.pkl", "rb"))
    # tensor_stat_dict = dict()
    # for k, v in state_dict.items():
    #     tensor_stat_dict[k] = jt.array(v)
    # runner.model.load_state_dict(tensor_stat_dict)

    if args.task == "train":
        runner.run()
    elif args.task == "val":
        runner.val()
    elif args.task == "test":
        runner.test()
    elif args.task == "vis_test":
        runner.run_on_images(args.save_dir)

if __name__ == "__main__":
    main()
