from sklearn.metrics import matthews_corrcoef
from sklearn import metrics
import numpy as np
from scipy import stats
import pandas as pd
def com_auc(test_l,pred):
    ## 真实值，预测值
    fpr, tpr, thresholds = metrics.roc_curve(test_l, pred, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc,fpr,tpr,thresholds
def com_measure(y,py,name='model_name',heatmap=False):
    AUC,_,_,_=com_auc(y,py) 
    precision1, recall1,_ = metrics.precision_recall_curve(y,py)
    PRC = metrics.auc(recall1, precision1)
    py=np.where(py>0.5,1,0)
    cm = metrics.confusion_matrix(y, py)
    Pre=metrics.precision_score(y,py,average='macro')
    Recall=metrics.recall_score(y,py,average='macro')
    F1_score=metrics.f1_score(y,py,average='macro')
    ACC=(cm[0,0]+cm[1,1])/(np.sum(np.sum(cm,axis=0),axis=0))
#    _,p_value=stats.ttest_ind(y, py)
    if heatmap:    
        sn.heatmap(cm,annot=True)
    FP = cm[1,0] 
    TN = cm[1,1]
    Specificity=TN/(TN+FP)
#    MCC=(TP*TN-FP*FN)/(np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
    MCC=matthews_corrcoef(y,py)
#    return np.array([Pre,Recall,F1_score,ACC,Specificity,MCC,TP/len(y),FN/len(y),FP/len(y),TN/len(y),p_value,_AUC,PRC]).T
    return Pre,Recall,F1_score,ACC,Specificity,MCC,AUC,PRC
###############################################################################

if __name__=='__main__':
    dataf=np.loadtxt('E:/Motif/deeplearning/DL/Codedata/1.deepbind/1_PARCLIP_AGO1234_hg19/Seq/CNN/test.txt')
    Pre,Recall,F1_score,ACC,Specificity,MCC,AUC,PRC=com_measure(dataf.T[:,0],dataf.T[:,1])
    data=pd.DataFrame(np.array([Pre,Recall,F1_score,ACC,Specificity,MCC,AUC,PRC]))
    print(ACC,Specificity,MCC,AUC,PRC)
#    data.to_excel('data.xlsx')
    