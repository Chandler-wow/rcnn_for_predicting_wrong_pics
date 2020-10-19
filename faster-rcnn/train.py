# python train.py ppp2 --env='fasterrcnn' --plot-every=100 --load-path='./fasterrcnn_12211511_0.701052458187_torchvision_pretrain.pth'
from __future__ import  absolute_import
import os
import cv2
import ipdb
import matplotlib
from tqdm import tqdm
import numpy as np
from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize,inverse_preprocess
from model import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc
from model.utils.bbox_tools import *

# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
# import resource

# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')


def eval(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    return result


def train(**kwargs):
    opt._parse(kwargs)

    dataset = Dataset(opt)
    print('load data')
    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=True, \
                                  # pin_memory=True,
                                  num_workers=opt.num_workers)
    testset = TestDataset(opt)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       pin_memory=True
                                       )
    faster_rcnn = FasterRCNNVGG16()
    print('model construct completed')
    
    trainer = FasterRCNNTrainer(faster_rcnn)
    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)
    trainer.vis.text(dataset.db.label_names, win='labels')
    best_map = 0
    lr_ = opt.lr
    # ipdb.set_trace()
    for epoch in range(opt.epoch):
        trainer.reset_meters()
        for ii, (img, bbox_, label_, scale,_) in tqdm(enumerate(dataloader)):
            scale = at.scalar(scale)
            img, bbox, label = img.float(), bbox_, label_
            trainer.train_step(img, bbox, label, scale)
            
            if (ii + 1) % opt.plot_every == 0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                # plot loss
                trainer.vis.plot_many(trainer.get_meter_data())

                # plot groud truth bboxes
                ori_img_ = inverse_normalize(at.tonumpy(img[0]))
                gt_img = visdom_bbox(ori_img_,
                                     at.tonumpy(bbox_[0]),
                                     at.tonumpy(label_[0]),
                                     None,None,'green')
                trainer.vis.img('gt_img', gt_img)

                _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)
                # plot predicti bboxes
                pred_img = visdom_bbox(ori_img_,
                                       at.tonumpy(_bboxes[0]),
                                       at.tonumpy(_labels[0]).reshape(-1),
                                       at.tonumpy(_scores[0]))
                trainer.vis.img('pred_img', pred_img)

                # rpn confusion matrix(meter)
                trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
                # roi confusion matrix
                trainer.vis.img('roi_cm', at.totensor(trainer.roi_cm.conf, False).float())
        eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)
        trainer.vis.plot('test_map', eval_result['map'])
        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
        log_info = 'lr:{}, map:{},loss:{}'.format(str(lr_),
                                                  str(eval_result['map']),
                                                  str(trainer.get_meter_data()))
        trainer.vis.log(log_info)

        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            best_path = trainer.save(best_map=best_map)
        if epoch == 9:
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay

        if epoch == 13: 
            break

def ppp(**kwargs):
    opt._parse(kwargs)
    device = opt.device
    
    dataset = Dataset(opt,split = 'test')
    dataloader = data_.DataLoader(dataset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       pin_memory=True
                                       )
    faster_rcnn = FasterRCNNVGG16()
    faster_rcnn.to(device)
    print('model construct completed')
    
    trainer = FasterRCNNTrainer(faster_rcnn)
    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)
    trainer.vis.text(dataset.db.label_names, win='labels')
    all_count = 0
    count = 0
    zhen_duo = 0
    # ipdb.set_trace()
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (img, bbox_, label_, scale,difficult) in tqdm(enumerate(dataloader)):
        all_count += 1
        img, bbox, label = img.float().to(device), bbox_.to(device), label_.to(device)
        ori_img_ = inverse_normalize(at.tonumpy(img[0]))
        _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)
        
        
        thresh = 0.2
        gp_ = bbox_iou(_bboxes[0],bbox.numpy()[0])
        if _labels[0].size == 0 :
            continue
        gp = gp_.max(axis = 1)
        gp_ind = gp_.argmax(axis = 1)
        
        gt_bboxes += list(bbox_.numpy())
        gt_labels += list(label_.numpy())
        gt_difficults += list(difficult.numpy())
        pred_bboxes += _bboxes
        pred_labels += _labels
        pred_scores += _scores
        
        do = False
        # 如果预测框的最大IoU存在小于阈值的，检测错误
        # 如果预测框对应的那个真实框分类错误
        if sum(gp < thresh) > 0 or sum(at.tonumpy(_labels[0]).reshape(-1) != at.tonumpy(label_[0])[gp_ind]) > 0 : 
            do = True
        if do :
            if at.tonumpy(_labels[0]).reshape(-1).shape[0] < at.tonumpy(label_[0]).shape[0]:
                zhen_duo += 1
        
            gt_img = visdom_bbox(ori_img_,
                                at.tonumpy(bbox_[0]),
                                at.tonumpy(label_[0]),
                                None,None,'green')
            trainer.vis.img('img_{}_gt'.format(ii+1), gt_img)
    
            # plot predicti bboxes
            pred_img = visdom_bbox(ori_img_,
                                at.tonumpy(_bboxes[0]),
                                at.tonumpy(_labels[0]).reshape(-1),
                                at.tonumpy(_scores[0]))
            trainer.vis.img('img_{}_pred'.format(ii+1), pred_img)
            count += 1
    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    print(result)
    print('count:{}\t all:{}\t per:{:.2f}'.format(count,all_count,count/all_count))
    print('zhenduo:{}\t count:{}\t per:{:.2f}'.format(zhen_duo,count,zhen_duo/count))

if __name__ == '__main__':
    import fire

    fire.Fire()
