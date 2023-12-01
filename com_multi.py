import numpy as np
import pandas as pd 
from measure_code import com_measure
import os
import shutil
import argparse
import matplotlib.pyplot as plt
import numpy as np
parser = argparse.ArgumentParser(description="Process ATAC-seq data.")
parser.add_argument('--hash', default='fbd7c1da229c4007bf2d7b8a1ba1cf03',type=str,help='The The hash of the task')
args = parser.parse_args()
name=args.hash
path='./save/'+name+'/output/'
files=os.listdir(path)
#########################################
colname=['Precision','Recall','F1_score','ACC','Specificity','MCC','AUC','PRC']
values=[]
data=dict()
data['data']=[]
data['Precision']=[]
data['Recall']=[]
data['F1_score']=[]
data['ACC']=[]
data['Specificity']=[]
data['MCC']=[]
data['AUC']=[]
data['PRC']=[]
data['PRC']=[]
data['Area']=[]
########################################
def Areas(data):
    aa=0
    for i in range(len(data)-1):
        aa+=data[i]*data[i+1]*np.sin(np.pi/180*45)/2
    return aa
########################################
for i in range(len(files)):
    if 'val' not in files[i]:
        print(files[i])
        try:
            dessoCNN=np.loadtxt(path+files[i])
            cres=com_measure(dessoCNN[0,:],dessoCNN[1,:])
            data['data'].append(files[i].split('.')[0])
            data['Precision'].append(cres[0])
            data['Recall'].append(cres[1])
            data['F1_score'].append(cres[2])
            data['ACC'].append(cres[3])
            data['Specificity'].append(cres[4])
            data['MCC'].append(cres[5])
            data['AUC'].append(cres[6])
            data['PRC'].append(cres[7])
            data['Area'].append(Areas(cres))
            print(data['Area'][0])
            values.append(cres[0])
            values.append(cres[1])
            values.append(cres[1])
            values.append(cres[3])
            values.append(cres[4])
            values.append(cres[5])
            values.append(cres[6])
            values.append(cres[7])
        except:
             print(path+files[i])
#            shutil.rmtree(path+files[i])
#############################################
data=pd.DataFrame(data)
data=data[['data','Precision','Recall','F1_score','ACC','Specificity','MCC','AUC','PRC','Area']]
#data=data.mean(axis=0)
data.columns=['data','Precision','Recall','F1_score','ACC','Specificity','MCC','AUC','PRC','Area']
data.to_excel('./download/'+name+'/'+name+'_area.xlsx',index=None)


# 计算八个顶点的角度
angles = np.linspace(0, 2 * np.pi, len(colname), endpoint=False).tolist()

# 创建一个圆形图
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
# 定义标签到圆心的距离
label_distance = [1.3,1.15,1.1,1.15,1.35,1.15,1.1,1.15]  # 可以根据需要调整
# 添加八边形的边
ax.fill(angles, values, 'b', alpha=0.1)
# 绘制刻度和标签
for angle, label,distance in zip(angles, colname,label_distance):
    ax.text(angle, distance, label, ha='center', va='center',fontsize=20)
    ax.plot([angle, angle], [0, 1], color='gray')
# 设置极坐标的刻度范围
ax.set_ylim(0, 1)  # 调整极坐标的半径范围

# 隐藏极坐标的刻度标签
ax.set_xticklabels([])

# # 添加属性和值标签
# ax.set_xticks(angles)
# ax.set_xticklabels(colname)

# # 设置刻度标签的字体大小
# ax.tick_params(axis='x', labelsize=20)

# # 添加属性值标签
# for i, val in enumerate(values):
#     ax.annotate(f'{attributes[i]}: {val}', xy=(angles[i], val), fontsize=12, ha='center')

# 设置图的标题
plt.title('AEMR: '+str(round(data['Area'][0],2)), size=24, y=1.1)
plt.savefig('./download/'+name+'/aemr.png', dpi=300, bbox_inches='tight',transparent=True)