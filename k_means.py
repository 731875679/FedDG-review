import os
from scipy import *
from pylab import *
from PIL import Image
from scipy.cluster.vq import *
 
#统计文件目录下的文件个数
def get_path_num(file_dir):
    count=0
    for files in os.listdir(file_dir):
        count=count+1
    return count

#get all files in the path in a list
def get_figure_list(local_path):
    #get the number of the local figures
    local_num=get_path_num(local_path)

    im_list=[]

    for root, dirs, files in os.walk(local_path):
            for i in range(local_num):
                files[i]=root+files[i]

    for i in range(local_num):
        current_im_local = Image.open(files[i])
        #resize the fig to get smaller pixels
        current_im_local = current_im_local.resize( (384,384), Image.BICUBIC )
        #PIL.image -> np.array
        current_im_local = np.asarray(current_im_local, np.float32)
        im_list.append(current_im_local.transpose((2, 0, 1)))
    return im_list
# Uses sparse pca codepath.

#get all domains we need to interpolate
target_path=[r"Domain1/data/",r"Domain2/data/",r"Domain3/data/"]

imlist_target=[]
for path in target_path:
    imlist_target_domain = get_figure_list(path)
    imlist_target.append(imlist_target_domain)

imnbr = len(imlist_target)

#############K-means-鸢尾花聚类############
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
#from sklearn import datasets
from sklearn.datasets import load_iris

# iris = load_iris()
X = imlist_target ##表示我们只取特征空间中的后两个维度

estimator = KMeans(n_clusters=3)#构造聚类器
estimator.fit(X)#聚类
label_pred = estimator.labels_ #获取聚类标签
#绘制k-means结果
x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]

plt.scatter(x0[:, 0], x0[:, 1], c = "red", marker='o', label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c = "green", marker='*', label='label1')
plt.scatter(x2[:, 0], x2[:, 1], c = "blue", marker='+', label='label2')

plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc=2)

