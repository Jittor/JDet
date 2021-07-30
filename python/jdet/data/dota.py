from jdet.models.boxes.box_ops import rotated_box_to_poly_single
from jdet.utils.registry import DATASETS
from jdet.config.constant import DOTA1_CLASSES
from jdet.models.boxes.box_ops import rotated_box_to_poly_single
from jdet.data.custom import CustomDataset
import os

@DATASETS.register_module()
class DOTADataset(CustomDataset):
    CLASSES = DOTA1_CLASSES

    def parse_result(self,results,save_path):
        check_dir(save_path)
        data = {}
        for (dets,labels),img_name in results:
            img_name = os.path.splitext(img_name)[0]
            for det,label in zip(dets,labels):
                bbox = det[:5]
                score = det[5]
                classname = self.CLASSES[label-1]
                bbox = rotated_box_to_poly_single(bbox)
                temp_txt = '{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n'.format(
                            img_name, score, bbox[0], bbox[1], bbox[2], bbox[3], bbox[4],
                            bbox[5], bbox[6], bbox[7])
                if classname not in data:
                    data[classname] = []
                data[classname].append(temp_txt)
        for classname,lines in data.items():
            f_out = open(os.path.join(save_path, classname + '.txt'), 'w')
            f_out.writelines(lines)
            f_out.close()

    def evaluate(self,results,work_dir,epoch,logger=None):
        save_path = os.path.join(work_dir,f"detections/val_{epoch}")
        self.parse_result(results,save_path)

        
        
    