import joblib
import os 
from sklearn.svm import SVC
from utils.Config import opt

class SVMs:
    def __init__(self,method = 'train'):
        if method == 'train':
            self.svms = self.initialize()
        elif method == 'load':
            self.svms = self.load()
    
    def initialize(self):
        svms = dict()
        for cat in opt.VOC_BBOX_LABEL_NAMES:
            svms[cat] = SVC(kernel = 'linear',probability = True)
        return svms
    
    def train_one(self,cat,X,y):
        self.svms[cat].fit(X,y)
    
    def predict(self,X):
        labels = []
        for cat,svm in self.svms:
            label = svm.predict(X)
            prob = svm.predict_proba(X)
            label_ = []
            for ii,l in enumerate(label):
                if l == 0:
                    label_.append(l)
                else:
                    label_.append(prob[ii,1])
            labels.append(np.array(label_))
        labels = np.vstack(labels).T
        bg = np.where((labels == 0).all(axis = 1))[0]
        score = labels.max(axis = 1)
        return labels.argmax(axis = 1),score,bg
                
            
    def save(self):
        for cat,svm in self.svms.items():
            joblib.dump(svm,opt.model_path+'SVM_'+cat+'.model')
    
    def load(self):
        svms = dict()
        for cat in sopt.VOC_BBOX_LABEL_NAMES:
            filename = opt.model_path+'SVM_'+cat+'.model'
            if os.path.exists(filename):
                svms[cat] = joblib.load()
        return svms