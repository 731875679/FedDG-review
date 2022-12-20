import os
import numpy as np
from sklearn.cluster import KMeans
import cv2
from imutils import build_montages
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from sklearn.decomposition import PCA
from torchvision import transforms
import matplotlib.pyplot as plt

def draw_image(image):
    plt.imshow(image, cmap='gray')
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    plt.xticks([])
    plt.yticks([])
    
    return 0

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
        image_gray = current_im_local.convert('L')
        #PIL.image -> np.array
        image_gray = np.asarray(image_gray, np.float32)
        # image_gray = image_gray.transpose((2, 0, 1))
        #current_im_local=current_im_local.reshape(1,-1)
        im_list.append(image_gray)
    
    return im_list

image_list_client=[]
images_path = [r'./Domain1/data/',r'./Domain2/data/',r'./Domain3/data/']
local_num=np.zeros(3)

figure_vector=get_figure_matrix_vector(images_path[0])
image_list_client.append(get_figure_matrix_vector(images_path[0]))
image_list_client.append(get_figure_matrix_vector(images_path[1]))
image_list_client.append(get_figure_matrix_vector(images_path[2]))

#image_output=np.clip(image_list_client1 / 255, 0, 1)
# draw_image(image_output)
# plt.show()
len_domain=len(images_path)
len_fig_in_domain=len(figure_vector)

print("starting pca")
x1_eigen=[]
x2_eigen=[]
x1_ratio=[]
x2_ratio=[]

pca = PCA(n_components=2, svd_solver='full')
ax = plt.figure()
ax = plt.subplot(111)

for i in range(len_domain):
    for j in range(len_fig_in_domain):
        pca=pca.fit(image_list_client[i][j])
        x1_eigen.append(pca.explained_variance_[0])
        x2_eigen.append(pca.explained_variance_[1])
        x1_ratio=[pca.explained_variance_ratio_[0]]
        x2_ratio=[pca.explained_variance_ratio_[1]]

print(np.mean(x1_ratio))
print(np.mean(x2_ratio))

for i in range(len(x1_eigen)):
    ax.text(x1_eigen[i],x2_eigen[i],str(int(i/10)+1)+'_'+str(i%10+1))

#print(pca.explained_variance_ratio_)
# print("finish pca")
# print(all_images)
print("starting kmeans")

x_kmeans=[]
from sklearn.datasets import make_classification
for i in range(len(x1_eigen)):
    x=np.zeros(2)
    x[0]=x1_eigen[i]
    x[1]=x2_eigen[i]
    x_kmeans.append(x)

data=np.array(x_kmeans)
#start kmeans
y_pred=KMeans(n_clusters=3).fit_predict(data)

#数据归一化 get mean and variance
mean_x=np.mean(data[:,0])
mean_y=np.mean(data[:,1])
var_x=np.var(data[:,0])
var_y=np.var(data[:,1])

# print(y_pred)
#plot the points after Normalization
ax.scatter((data[:,0]-mean_x)/np.sqrt(var_x),(data[:,1]-mean_y)/np.sqrt(var_y),c=y_pred[:])
plt.grid()
plt.show()