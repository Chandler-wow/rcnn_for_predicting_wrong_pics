import sys
# sys.path.append('..')
import os
import random
import ipdb 
import numpy as np
import xml.etree.ElementTree as ET
import torch as t
import pandas as pd 
from tqdm import tqdm
from PIL import Image
from model.SelectiveSearch import selective_search as ss
from utils.bbox_tools import bbox_iou,bbox2loc
from utils.supplement import warp_img,cut_img
from .BaseDataset import BaseDataset
from utils.Config import opt

class TrainCNNDataset(BaseDataset):
    def __init__(self, data_dir, split='trainval',
                 use_difficult=False, return_difficult=False,
                 ):
        super(TrainCNNDataset,self).__init__(data_dir, split='trainval',use_difficult=False, return_difficult=False)
        
    def __getitem__(self,i):
        '''
        找到每一个图片的当中的正样本和负样本bbox，然后将这些bbox换成相应位置的图片和标签一起传递给CNN训练
        这里我修改了一个地方，本来是每个Batch都是32个正样本，96个负样本的，但是因为正样本过少，甚至出现了
        正样本为0的现象，所以除非正样本全为0，否则所有的负样本数量是正样本的3倍，但是着就造成了每次训练的
        图片数量是不相同的。
        '''
        img,gt_bbox,gt_label,_ = self.get_example(i)
        bbox = ss(img.transpose((1,2,0)))
        cate_label = np.apply_along_axis(lambda x:self.identify_pos_neg(x,gt_bbox,gt_label),axis = 1,arr = bbox)
        pos_ind = np.where(cate_label[:,0])[0]
        neg_ind = np.where(cate_label[:,0] == False)[0]
        if len(neg_ind) > 96:
            np.random.shuffle(neg_ind)
            if len(pos_ind) != 0:
                neg_ind = neg_ind[:int(len(pos_ind)*96/32)]
            else:
                neg_ind = neg_ind[:96]
                
        neg_bbox = bbox[neg_ind]
        neg_label = cate_label[neg_ind,1]
        neg_img = cut_img(neg_bbox,img)
        list_neg_img = list(neg_img)
        neg_sample = np.array([(list_neg_img[i],neg_label[i]) for i in range(neg_img.shape[0])])
        
        if len(pos_ind) != 0:
            pos_bbox = bbox[pos_ind]
            pos_label = cate_label[pos_ind,1]
            pos_img = cut_img(pos_bbox,img)
            list_pos_img = list(pos_img)
            pos_sample = np.array([(list_pos_img[i],pos_label[i]) for i in range(pos_img.shape[0])])
            sample = np.vstack((pos_sample,neg_sample))
            np.random.shuffle(sample)
        else:
            sample = neg_sample
            
        batch_imgs = np.apply_along_axis(lambda x:x[0][:,:,:],axis = 1,arr = sample)
        batch_labels = sample[:,1]
        return t.from_numpy(batch_imgs.astype('float32')),t.from_numpy(batch_labels.astype('float32'))
    
    def trim(self,l,n):
        length = len(l)
        redundant = length%n
        if redundant != 0:
            return l[:-redundant]
        else:
            return l
        
    def identify_pos_neg(self,bbox,gts,gtl):
        pos = True
        neg = False
        bg = bbox_iou(bbox.reshape((-1,4)),gts)
        if (bg > 0.5).any():
            return pos,gtl[np.argmax(bg)]
        else:
            return neg,20

class NormalDataset(BaseDataset):
    def __init__(self, data_dir, split='trainval',
                 use_difficult=False, return_difficult=False,
                 ):
        super(NormalDataset,self).__init__(data_dir, split='trainval',use_difficult=False, return_difficult=False)
    
    def __getitem__(self,i):
        ''' 测试集使用的dataset不需要其他改变'''
        img, gt_bbox, gt_label, difficult = self.get_example(i)
        bbox = ss(img.transpose((1,2,0)))
        bbox_imgs = cut_img(bbox,img)
        bbox_imgs = t.from_numpy(bbox_imgs.astype('float32'))
        return img,bbox_imgs,bbox,gt_bbox,gt_label,difficult
        
class TrainSVMAndLRDataset(BaseDataset):
    '''
    SVM和LR的训练dataset应该分开，这样结构更加明显。
    但是因为内存开销太大了，并且两个dataset的很多步骤以及SVM和LR的训练步骤是相似的，所以合在了一起。
    '''
    def __init__(self, data_dir, split='trainval',
                 use_difficult=False, return_difficult=False,
                 ):
        super(TrainSVMAndLRDataset,self).__init__(data_dir, split='trainval',use_difficult=False, return_difficult=False)
    
    def __getitem__(self,i):
        ''' SVM和LR的区别就在于mask的不同 '''
        # ground truth为正例，需要选择负例
        img, gt_bbox, gt_label, difficult = self.get_example(i)
        bbox = ss(img.transpose((1,2,0)))
        iou = bbox_iou(bbox,gt_bbox)
        svm_mask = np.where((iou < opt.svm_tresh).all(axis = 1))[0]
        svm_bbox = bbox[svm_mask]
        svm_iou = iou[svm_mask]
        # SVM 操作
        # 获得所有负例
        neg_label = np.apply_along_axis(lambda x:gt_label[self.get_hardest_neg(x)],axis = 1,arr = svm_iou)
        neg_img = cut_img(svm_bbox,img)
        # 获得所有正例
        pos_label = gt_label
        pos_imgs = cut_img(gt_bbox,img)
        # 整合
        sample_img = np.vstack((neg_img,pos_imgs))
        sample_label = np.hstack((neg_label,pos_label))
        sample_pos_neg_label = np.hstack((np.zeros((neg_label.shape[0],)),np.ones((pos_label.shape[0]))))
        # LR操作
        lr_mask = np.where((iou > opt.LR_tresh).any())[0]
        lr_bbox = bbox[lr_mask]
        lr_iou = iou[lr_mask]
        
        bbox_imgs = cut_img(lr_bbox,img)
        ind = lr_iou.argmax(axis = 1)
        labels = gt_label[ind]
        locs = bbox2loc(lr_bbox.astype('float32'),gt_bbox[ind])
        return t.from_numpy(sample_img.astype('float32')),sample_label,sample_pos_neg_label,t.from_numpy(bbox_imgs.astype('float32')),locs,labels
    
    def get_hardest_neg(self,a):
        # 选择一个最难学习的负例
        if sum(a == 0) == 0:
            return a.argmin()
        elif sum(a == 0) == 1:
            return np.where(a == 0)[0][0]
        else:
            return np.random.choice(np.where(a == 0)[0])