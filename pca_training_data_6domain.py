import numpy as np
from PIL import Image
import scipy.misc
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import os
from glob import glob
import time
import shutil
from PIL import Image
import cv2
from imutils import build_montages
import torch.nn as nn
import torchvision.models as models
from sklearn.decomposition import PCA
from torchvision import transforms
from sklearn import manifold
from sklearn.cluster import KMeans

#统计文件目录下的文件个数
def get_path_num(file_dir):
    count=0
    for files in os.listdir(file_dir):
        count=count+1
    return count

#get all files in the path in a list
def get_figure_matrix_vector(local_path):
    #get the number of the local figures
    local_num=get_path_num(local_path)
    im_list=[]
    for root, dirs, files in os.walk(local_path):
            for i in range(local_num):
                files[i]=root+files[i]
    for i in range(local_num):
        current_im_local = Image.open(files[i])
        #resize the fig to get smaller pixels
        current_im_local = current_im_local.resize( (224,224), Image.BICUBIC )
        #rgb figure -> gray figure
        image_gray = current_im_local.convert('L')
        #PIL.image -> np.array
        image_gray = np.asarray(image_gray, np.float32)
        # image_gray = image_gray.transpose((2, 0, 1))
        #matrix -> vector
        #image_gray=image_gray.reshape(1,-1)
        #save all the vector in a list
        im_list.append(image_gray)
    return im_list

training_data_path=r'./domain_data/data/'
training_label_path=r'./domain_data/label/'

training_data_list=get_figure_matrix_vector(training_data_path)
training_label_list=[]

len_training_data=len(training_data_list)

print("starting pca")
x1_eigen=[]
x2_eigen=[]
x1_ratio=[]
x2_ratio=[]
ax = plt.figure()
ax = plt.subplot(111)
pca = PCA(n_components=2, svd_solver='full')

for i in range(len_training_data):
    pca=pca.fit(training_data_list[i])
    x1_eigen.append(pca.explained_variance_[0])
    x2_eigen.append(pca.explained_variance_[1])
    x1_ratio.append(pca.explained_variance_ratio_[0])
    x2_ratio.append(pca.explained_variance_ratio_[1])

print(np.mean(x1_ratio))
print(np.mean(x2_ratio))

dir={}
ax.scatter(x1_eigen[:],x2_eigen[:])
for i in range(len_training_data):
    ax.text(x1_eigen[i],x2_eigen[i],str(i))

#print(pca.explained_variance_ratio_)
# print("finish pca")
# print(all_images)
print("start kmeans")
x_kmeans=[]
from sklearn.datasets import make_classification
for i in range(len(x1_eigen)):
    x=np.zeros(2)
    x[0]=x1_eigen[i]
    x[1]=x2_eigen[i]
    x_kmeans.append(x)
data=np.array(x_kmeans)
y_pred=KMeans(n_clusters=6).fit_predict(data)
ax.scatter(data[:,0],data[:,1],c=y_pred[:])

mean_x1=np.mean(data[:,0])
mean_x2=np.mean(data[:,1])
var_x1=np.var(data[:,0])
var_x2=np.var(data[:,1])

data[:,0]=(data[:,0]-mean_x1)/np.sqrt(var_x1)
data[:,1]=(data[:,1]-mean_x2)/np.sqrt(var_x2)

import joblib
# save data. y_pred
joblib.dump(data, 'data.pkl')
joblib.dump(y_pred, 'y_pred.pkl') 

dic_fig_class={}
#use dictionary to save the fig with its class
for i in range(len_training_data):
    dic_fig_class[i]=y_pred[i]

dic_final={0:[],1:[],2:[],3:[],4:[],5:[]}

for key,value in dic_fig_class.items():
    if value == 0:
        dic_final[value].append(key)
    if value == 1:
        dic_final[value].append(key)
    if value == 2:
        dic_final[value].append(key)
    if value == 3:
        dic_final[value].append(key)
    if value == 4:
        dic_final[value].append(key)
    if value == 5:
        dic_final[value].append(key)
#print(dic_final)

#create a dir to store the training figures and its label after 领域迁移
training_figure_dir=r'./data_after_pca/domain0/data'
if os.path.exists(training_figure_dir)==0:
    os.makedirs(training_figure_dir)
training_figure_dir=r'./data_after_pca/domain0/label'
if os.path.exists(training_figure_dir)==0:
    os.makedirs(training_figure_dir)

training_label_dir=r'./data_after_pca/domain1/data'
if os.path.exists(training_label_dir)==0:
    os.makedirs(training_label_dir)
training_label_dir=r'./data_after_pca/domain1/label'
if os.path.exists(training_label_dir)==0:
    os.makedirs(training_label_dir)

training_label_dir=r'./data_after_pca/domain2/data'
if os.path.exists(training_label_dir)==0:
    os.makedirs(training_label_dir)
training_label_dir=r'./data_after_pca/domain2/label'
if os.path.exists(training_label_dir)==0:
    os.makedirs(training_label_dir)

training_figure_dir=r'./data_after_pca/domain3/data'
if os.path.exists(training_figure_dir)==0:
    os.makedirs(training_figure_dir)
training_figure_dir=r'./data_after_pca/domain3/label'
if os.path.exists(training_figure_dir)==0:
    os.makedirs(training_figure_dir)

training_label_dir=r'./data_after_pca/domain4/data'
if os.path.exists(training_label_dir)==0:
    os.makedirs(training_label_dir)
training_figure_dir=r'./data_after_pca/domain4/label'
if os.path.exists(training_figure_dir)==0:
    os.makedirs(training_figure_dir)

training_label_dir=r'./data_after_pca/domain5/data'
if os.path.exists(training_label_dir)==0:
    os.makedirs(training_label_dir)
training_figure_dir=r'./data_after_pca/domain5/label'
if os.path.exists(training_figure_dir)==0:
    os.makedirs(training_figure_dir)

#read all figures and labels into lists
data_path_list=[]
label_path_list=[]

# for root, dirs, files in os.walk(training_data_path):
#         for i in range(len_training_data):
#             data_path_list.append(root+files[i])
# for root, dirs, files in os.walk(training_label_path):
#         for i in range(len_training_data):
#             label_path_list.append(root+files[i])

# for key,value in dic_fig_class.items():
#     if value == 0:
#         for i in dic_final[0]:
#             shutil.copy(data_path_list[i], r'./data_after_pca/domain0/data/'+str(i)+'.bmp')
#             shutil.copy(label_path_list[i], r'./data_after_pca/domain0/label/'+str(i)+'.bmp')
#     if value == 1:
#         for i in dic_final[1]:
#             shutil.copy(data_path_list[i], r'./data_after_pca/domain1/data/'+str(i)+'.bmp')
#             shutil.copy(label_path_list[i], r'./data_after_pca/domain1/label/'+str(i)+'.bmp')
#     if value == 2:
#         for i in dic_final[2]:
#             shutil.copy(data_path_list[i], r'./data_after_pca/domain2/data/'+str(i)+'.bmp')
#             shutil.copy(label_path_list[i], r'./data_after_pca/domain2/label/'+str(i)+'.bmp')
#     if value == 3:
#         for i in dic_final[3]:
#             shutil.copy(data_path_list[i], r'./data_after_pca/domain3/data/'+str(i)+'.bmp')
#             shutil.copy(label_path_list[i], r'./data_after_pca/domain3/label/'+str(i)+'.bmp')
#     if value == 4:
#         for i in dic_final[4]:
#             shutil.copy(data_path_list[i], r'./data_after_pca/domain4/data/'+str(i)+'.bmp')
#             shutil.copy(label_path_list[i], r'./data_after_pca/domain4/label/'+str(i)+'.bmp')
#     if value == 5:
#         for i in dic_final[5]:
#             shutil.copy(data_path_list[i], r'./data_after_pca/domain5/data/'+str(i)+'.bmp')
#             shutil.copy(label_path_list[i], r'./data_after_pca/domain5/label/'+str(i)+'.bmp')

plt.show()