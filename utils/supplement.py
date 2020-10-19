import numpy as np 
from skimage import transform as sktsf
from PIL import Image,ImageDraw
from utils.bbox_tools import bbox_iou
from utils.Config import opt

def warp_img(img):
    '''各向同性缩放目标框的图片至227x227，满足AlexNey的要求'''
    C,H,W = img.shape
    newimg = np.zeros((C,H+32,W+32))
    newimg[:,16:-16,16:-16] = img
    newimg = sktsf.resize(newimg, (C, 227, 227), mode='reflect',anti_aliasing=False)
    return newimg

def wrong_2_draw(img,bboxes,labels,gt_bboxes,gt_labels,ind):
    ''' 判断这次预测是否正确，不正确则画出两张分别带有ground truth和pred bbox的图片 '''
    pg = bbox_iou(bboxes,gt_bboxes)
    pg_ind = pg.argmax(axis = 1)
    
    if sum(pg.max(axis = 1) < 0.2) > 0 or sum(labels != gt_labels[pg_ind]) > 0:
        draw_(img,bboxes,ind)
        draw_(img,gt_bboxes,ind,outline = (0,255,0))
        
def draw_(img,bboxes,ind,outline = (255,0,0)):
    ''' 画出两张分别带有ground truth和pred bbox的图片 '''
    image = Image.fromarray(img.transpose((1,2,0)))
    draw = ImageDraw.Draw(image)
    for bbox in bboxes:
        ymin,xmin,ymax,xmax = bbox
        draw.rectangle((xmin,ymin,xmax,ymax),outline = outline)
    image.save(opt.pic_path+'img_{}_pred.png'.format(ind))

def cut_img(bbox,img):
    ''' 将img上的bbox框住的图片选出来 '''
    def f(ox):
        ymin,xmin,ymax,xmax = ox.astype('int32')
        return warp_img(img[:,xmin:xmax,ymin:ymax])
    if bbox.shape[0] == 0:
        return np.array([]).reshape((-1,3,3,227))
    return np.apply_along_axis(lambda x:f(x),axis = 1,arr = bbox)
    
def maki_all(seq,sett):
    ''' 提前测试test函数的工具函数，让svm和lr模型都训练一次后退出，这样子每个模型都具有权重 '''
    for i in np.unique(seq):
        if i == 20:
            continue
        l = opt.VOC_BBOX_LABEL_NAMES[i]
        sett.discard(l)
    return sett,True if sett == set() else False