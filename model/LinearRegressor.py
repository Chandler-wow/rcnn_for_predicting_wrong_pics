import joblib
import os 
from sklearn.neural_network import MLPRegressor
from utils.Config import opt

class LRs:
    '''
    LR 模型：因为输出是4个元素并且是线性回归器，所以使用了只有输入输出层并且不加激活函数的神经网络代替
    '''
    def __init__(self,method = 'train'):
        if method == 'train':
            self.lrs = self.initialize()
        elif method == 'load':
            self.lrs = self.load()
            
    def initialize(self):
        lrs = dict()
        for cat in opt.VOC_BBOX_LABEL_NAMES:
            lrs[cat] = MLPRegressor(hidden_layer_sizes = (4,),activation = 'identity',learning_rate = 'adaptive')
        return lrs
        
    def train_one(self,cat,X,y):
        self.lrs[cat].fit(X,y)
        
    def predict(self,cat,X):
        return self.lrs[cat].predict(X)
    
    def save(self):
        for cat,lr in self.lrs.items():
            joblib.dump(lr,opt.model_path+'LR_'+cat+'.model')
    
    def load(self):
        lrs = dict()
        for cat in sopt.VOC_BBOX_LABEL_NAMES:
            filename = opt.model_path+'LR_'+cat+'.model'
            if os.path.exists(filename):
                lrs[cat] = joblib.load(filename)
        return lrs