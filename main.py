import numpy as np 
import fire
import sys
import torch as t 
import matplotlib.pyplot as plt 
import pandas as pd 
import ipdb
import cv2
import warnings
from dataset.VOCDataset import TrainCNNDataset,TrainSVMAndLRDataset,NormalDataset
from model.AlexNet import AlexNet
from model.SVM import SVMs
from model.LinearRegressor import LRs
from utils.Config import opt
from utils.eval_tool import eval_detection_voc
from utils.supplement import wrong_2_draw,maki_all
from utils.bbox_tools import loc2bbox
from torchnet import meter
from tqdm import tqdm 
from torchvision.ops import nms
warnings.filterwarnings('ignore')


def nograd(f):
    def new_f(*args,**kwargs):
        with t.no_grad():
           return f(*args,**kwargs)
    return new_f
    
def train_CNN(**kwargs):
    '''
    训练CNN提取特征图
    '''
    # opt._parse(kwargs)
    device = opt.device
    
    model = AlexNet()
    print('Loading the pre-trained CNN model...')
    model.load(opt.load_native_alexnet)
    
    in_features = model.classifier[6].in_features
    model.classifier[6] = t.nn.Linear(in_features,21)
    model.to(device)
    
    # ipdb.set_trace()
    trainset = TrainCNNDataset(opt.voc_data_dir)
    train_dataloader = t.utils.data.DataLoader(trainset,batch_size = 1,shuffle = True)
    criterion = t.nn.CrossEntropyLoss()
    optimizer = t.optim.Adam(model.parameters(),lr = opt.lr,)
    loss_meter = meter.AverageValueMeter()
    loss_list = []
    
    print('Start traing CNN...')
    for epoch in range(opt.epoch):
        loss_meter.reset()
        all_neg_counts = 0
        counts = 0
        for ii,(imgs,labels) in tqdm(enumerate(train_dataloader)):
            inputs = imgs[0].to(device)
            targets = labels.squeeze().to(device)
            
            if (labels.numpy().squeeze() == 20).all():
                all_neg_counts += 1
            counts += 1
            
            optimizer.zero_grad()
            score = model(inputs)
            # RuntimeError: "log_softmax_lastdim_kernel_impl" not implemented for 'Long'
            loss = criterion(score,targets.long())
            loss.backward()
            optimizer.step()
            
            loss_meter.add(loss.item())
        
        loss_list.append(loss_meter.value()[0])
        model.save(opt.model_path+'AlexNet.pth')
        print('Epoch {} all_neg_percent : {}'.format(epoch+1,all_neg_counts/counts*1.0))
    print('Plot loss picture...')
    plt.plot(np.arange(1,len(loss_list)+1),loss_list,'r^-')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(opt.pic_path+'CNN_train.png')
    print('Train CNN is Done!')
    
@nograd
def train_SVMs_and_LRs(**kwargs):
    # opt._parse(kwargs)
    
    device = opt.device
    model = AlexNet()
    in_features = model.classifier[6].in_features
    model.classifier[6] = t.nn.Linear(in_features,21)
    print('Loading total model...')
    if opt.load_trained_path != None:
        model.load(opt.load_trained_path)
    svms = SVMs(method = 'train')
    lrs = LRs(method = 'train')
        
    model.to(device)
    dataset = TrainSVMAndLRDataset(opt.voc_data_dir)
    dataloader = t.utils.data.DataLoader(dataset,batch_size = 1,shuffle = False)  
    # ipdb.set_trace()
    
    print('Obtaining all features...')
    features_svm = []
    features_lr = []
    svm_labels = []
    pos_neg = []
    locs = []
    lr_labels = []
    
    for ii,(sample_imgs,sample_labels,sample_pos_neg_labels,bbox_imgs,locs_,labels_) in tqdm(enumerate(dataloader)):
        inputs_ = sample_imgs.to(device)[0]
        features_ = model.get_features(inputs_).cpu().detach().numpy()
        features_svm.extend(list(features_))
        svm_labels.extend(sample_labels.numpy())
        pos_neg.extend(sample_pos_neg_labels.numpy())
        
        if bbox_imgs.shape[1] != 0:
            _inputs = bbox_imgs.to(device)[0]
            _features = model.get_features(_inputs).cpu().detach().numpy()
            features_lr.extend(list(_features))
            lr_labels.extend(list(labels_.numpy()))
            locs.extend(locs_.numpy())
        
    svm_labels = np.hstack(svm_labels)
    pos_neg = np.hstack(pos_neg)
    lr_labels = np.hstack(lr_labels)
    # locs = np.vstack(locs)

    df_svm = pd.DataFrame()
    df_svm['features'] = features_svm
    df_svm['labels'] = svm_labels
    df_svm['pos_neg'] = pos_neg
    
    df_lr = pd.DataFrame()
    df_lr['features'] = features_lr
    df_lr['labels'] = lr_labels
    df_lr['locs'] = locs

    print('Sort out SVM and LR training data of different categories...')
    svm_data_dict = dict()
    svm_label_dict = dict()
    lr_data_dict = dict()
    lr_loc_dict = dict()
    for i in range(opt.voc_bbox_label_length):
        name = opt.VOC_BBOX_LABEL_NAMES[i]
        if (df_svm.loc[:,'labels'] == i).any(): 
            svm_data_dict[name] = np.vstack(df_svm.loc[(df_svm.loc[:,'labels'] == i),'features'].values)
            svm_label_dict[name] = df_svm.loc[(df_svm.loc[:,'labels'] == i),'pos_neg'].values
        if (df_lr.loc[:,'labels'] == i).any(): 
            lr_data_dict[name] = np.vstack(df_lr.loc[(df_lr.loc[:,'labels'] == i),'features'].values)
            lr_loc_dict[name] = np.vstack(df_lr.loc[(df_lr.loc[:,'labels'] == i),'locs'].values)
        
    print('Training SVM and LR...')
    old = None
    for cat in tqdm(opt.VOC_BBOX_LABEL_NAMES):
        if cat in svm_data_dict.keys():
            svms.train_one(cat,svm_data_dict[cat],svm_label_dict[cat])
        if cat in lr_data_dict.keys():
            lrs.train_one(cat,lr_data_dict[cat],lr_loc_dict[cat])
    
    print('Saving model...')
    svms.save()
    lrs.save()
    print('Train SVMs and LRs is Done!')
    
    
def train(**kwargs):
    opt._parse(kwargs)
    train_CNN()
    train_SVMs_and_LRs()

def test(**kwargs):
    opt._parse(kwargs)
    device = opt.device
    
    cnn = AlexNet()
    in_features = cnn.classifier[6].in_features
    cnn.classifier[6] = t.nn.Linear(in_features,21)
    cnn.load(opt.load_trained_path)
    cnn.to(device)
    svms = SVMs(method = 'load')
    lrs = LRs(method = 'load')
    
    dataset = NormalDataset(opt.voc_data_dir,split = 'test')
    dataloader = t.utils.data.DataLoader(dataset,batch_size = 1,num_workers = 0,shuffle = False)
    ipdb.set_trace()
    pred_bbox_last_, pred_label_last_, pred_score_last_, gt_bbox_, gt_label_ = [],[],[],[],[]
    for ii,(img,bbox_imgs,bboxes,gt_bbox,gt_label,_) in tqdm(enumerate(dataloader)):
    # for ii in tqdm(range(len(dataset))):
        # img,bbox_imgs,bboxes,gt_bbox,gt_label,_ = dataset.getitem(ii)
        img = img[0].numpy()
        bboxes = bboxes[0].numpy()
        gt_bbox = gt_bbox[0].numpy()
        gt_label = gt_label[0].numpy()
        inputs = bbox_imgs.to(device)[0]
        features = cnn.get_features(inputs).cpu().detach().numpy()
        '''
        imOut = cv2.UMat(img.numpy().transpose((1,2,0))*255).get()
        for i, rect in enumerate(bboxes):
            ymin, xmin, ymax, xmax = rect
            cv2.rectangle(imOut, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imwrite('./pic/1.jpg',imOut)
        
        pred_label,pred_score,pred_bg = svms.predict(features)
        mask = np.where(pred_bg == False)[0]
        
        counts += 1
        if mask.size == 0:
            null_counts += 1
        if ii == 10:
            print("null per:{}".format(1.0*null_counts/counts))
            break
        
        pred_label = pred_label[mask]
        pred_score = pred_score[mask]
        pred_bboxes_unregress = bboxes[mask]  # np.ndarray
        bbox_features = features[mask]
        '''
        pred_label,pred_score = svms.predict(features)
        pred_bboxes_unregress = bboxes
        bbox_features = features
        # mark
        # 获取每个种类的bboxes和score，为了nms
        pred_bboxes_2_nms = dict()
        pred_score_2_nms = dict()
        bbox_features_2_nms = dict()
        for lab in np.unique(pred_label):
            lab_mask = np.where(pred_label == lab)[0]
            pred_bboxes_2_nms[opt.VOC_BBOX_LABEL_NAMES[lab]] = pred_bboxes_unregress[lab_mask]
            pred_score_2_nms[opt.VOC_BBOX_LABEL_NAMES[lab]] = pred_score[lab_mask]
            bbox_features_2_nms[opt.VOC_BBOX_LABEL_NAMES[lab]] = bbox_features[lab_mask]
            
        # nms & regression
        pred_bbox_last = []
        pred_label_last = []
        pred_score_last = []
        for cat,bbox_nms in pred_bboxes_2_nms.items():
            mask = np.where(pred_score_2_nms[cat] > opt.nms_thresh)[0]
            if mask.size == 0:
                continue
            else:
                bbox_nms = bbox_nms[mask]
                pre_score_2_nms[cat] =  pre_score_2_nms[cat][mask]
                features_2_nms = features_2_nms[mask]
            keep_mask = nms(t.from_numpy(bbox_nms[:,[1,0,3,2]]).float(),t.from_numpy(pred_score_2_nms[cat]).float(),opt.iou_thresh)
            loc = lrs.predict(cat,bbox_features_2_nms[cat][keep_mask])
            pred_bbox_cat = loc2bbox(bbox_nms[keep_mask],loc)
            
            pred_score_last.append(pred_score_2_nms[cat][keep_mask])
            pred_bbox_last.append(pred_bbox_cat)
            pred_label_last.extend([cat]*pred_bbox_cat.shape[0])
        if len(pred_label_last) > 0:
            wrong_2_draw(img,np.vstack(pred_bbox_last),np.array(pred_label_last),gt_bbox,gt_label)
            
            pred_bbox_last_ += np.vstack(pred_bbox_last)
            pred_label_last_ += np.array(pred_label_last)
            pred_score_last_ += np.hstack(pred_Score_last)
            gt_bbox_ += gt_bbox
            gt_label_ += gt_label
    # evaluation
    res = eval_detection_voc(pred_bbox_last_, pred_label_last_, pred_score_last_, gt_bbox_, gt_label_)
    print(res)
    pd.DataFrame(res).to_excel('./res.xlsx')
    
if __name__ == '__main__':
    fire.Fire()