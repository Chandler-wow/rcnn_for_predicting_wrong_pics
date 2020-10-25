# 训练的配置文件
from pprint import pprint
import torch as t 

class Config:
    VOC_BBOX_LABEL_NAMES = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor') 
    voc_bbox_label_length = 20
    num_workers = 4
    test_num_workers = 4
    lr = 1e-3  # 微调时的学习率
    svm_tresh = 0.3 
    LR_tresh = 0.7
    nms_thresh = 0.6
    iou_thresh = 0.6
    epoch = 10  # 训练轮数
    ss_num_bboxes = 2000
    bef_categories = 1000  # AlexNet最初最后一层的神经元个数
    aft_categories = 21  # AlexNet修改的最后一层的神经元个数
    
    device = 'cpu'
    pic_path = './pic/'
    voc_data_dir = './VOCdevkit/VOC2007/'
    model_path = './model/checkpoints/'
    load_native_alexnet = model_path+'alexnet-owt-4df8aa71.pth'  # 加载的预训练模型
    load_trained_path = model_path+'AlexNet.pth'
    
    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')
    
    def _state_dict(self):
       return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
               if not k.startswith('_')}

opt = Config()
    