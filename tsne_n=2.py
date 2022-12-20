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
from sklearn import manifold

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
        image_gray=image_gray.reshape(1,-1)
        im_list.append(image_gray)
    
    return im_list

image_list_client=[]
images_path = [r'./Domain1/data/',r'./Domain2/data/',r'./Domain3/data/']
local_num=np.zeros(3)
len_each_domain=len(get_figure_matrix_vector(images_path[0]))

image_list_client.append(get_figure_matrix_vector(images_path[0]))
image_list_client.append(get_figure_matrix_vector(images_path[1]))
image_list_client.append(get_figure_matrix_vector(images_path[2]))

X_tsne=[]
for i in range(len(images_path)):
    for j in range(len_each_domain):
        X_tsne.append(image_list_client[i][j][0]) 

#image_output=np.clip(image_list_client1 / 255, 0, 1)
# draw_image(image_output)
# plt.show()

# print("starting tsne")
ax=plt.figure()
ax = plt.subplot(111)

target=[]
for i in range(3):
    for j in range(10):
        if i==0:
            target.append('black')
        elif i==1:
            target.append('blue')
        elif i==2:
            target.append('red')

tsne = manifold.TSNE(n_components=3, init='pca', random_state=501)
X_after_tsne = tsne.fit_transform(X_tsne)

for i in range(30):
    ax.scatter(X_after_tsne[i, 0], X_after_tsne[i, 1], c=target[i])

plt.legend()
plt.xticks([])
plt.yticks([])
plt.show()
