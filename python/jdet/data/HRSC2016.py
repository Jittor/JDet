from jdet.data.dota import DOTADataset
from jdet.utils.registry import DATASETS
from jdet.config.constant import HRSC2016_CLASSES, get_classes_by_name
from jdet.utils.general import check_dir
from tqdm import tqdm
from PIL import Image
import os.path as osp
import xml.etree.ElementTree as ET
import numpy as np

def list_from_file(filename, prefix='', offset=0, max_num=0):
    """Load a text file and parse the content as a list of strings.

    Args:
        filename (str): Filename.
        prefix (str): The prefix to be inserted to the begining of each item.
        offset (int): The offset of lines.
        max_num (int): The maximum number of lines to be read,
            zeros and negatives mean no limitation.

    Returns:
        list[str]: A list of strings.
    """
    cnt = 0
    item_list = []
    with open(filename, 'r') as f:
        for _ in range(offset):
            f.readline()
        for line in f:
            if max_num > 0 and cnt >= max_num:
                break
            item_list.append(prefix + line.rstrip('\n'))
            cnt += 1
    return item_list


@DATASETS.register_module()
class HRSC2016Dataset(DOTADataset):
    def __init__(self,*arg,**kwargs):
        super().__init__(*arg,**kwargs)
        self.CLASSES = HRSC2016_CLASSES
