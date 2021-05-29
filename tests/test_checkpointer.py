import unittest

from jdet.utils.checkpointer import Checkpointer


class TestCheckpointer(unittest.TestCase):
    def test(self):
        checkpointer = Checkpointer()
        pth_file = "/home/lxl/workspace/JDet/weights/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
        checkpointer._load_torch(pth_file)



if __name__ == "__main__":
    unittest.main()