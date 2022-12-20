import numpy as np
from PIL import Image
import scipy.misc
import matplotlib.pyplot as plt
import cv2
import numpy as np
import SimpleITK as sitk
import os
import numpy as np
from glob import glob
import time
import shutil
from PIL import Image
import math
#统计文件目录下的文件个数
def get_path_num(file_dir):
    count=0
    for files in os.listdir(file_dir):
        count=count+1
    return count

def extract_amp_spectrum(trg_img):

    fft_trg_np = np.fft.fft2( trg_img, axes=(-2, -1) )
    #计算复杂参数的角度。复数由“ x + yi”表示，其中x和y是实数。角度由公式计算tan^{-1}(x/y).
    amp_target, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)*180/np.pi

    return amp_target,pha_trg

#the interpolation algorithm
def amp_spectrum_swap( amp_local, amp_target, L=0.1 , ratio=0):
    
    a_local = np.fft.fftshift( amp_local, axes=(-2, -1) )
    a_trg = np.fft.fftshift( amp_target, axes=(-2, -1) )

    _, h, w = a_local.shape
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)

    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1

    a_local[:,h1:h2,w1:w2] = a_local[:,h1:h2,w1:w2] * ratio + a_trg[:,h1:h2,w1:w2] * (1- ratio)
    a_local = np.fft.ifftshift( a_local, axes=(-2, -1) )

    return a_local

def draw_image(image):
    plt.imshow(image, cmap='gray')
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    plt.xticks([])
    plt.yticks([])
    
    return 0

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

# local_data_path=r"Training_data/data/"
# local_label_path=r"Training_data/label/"

# #get the number of the local figures
# local_num=get_path_num(local_data_path)


# #get all lists all we need
# local_data_list=get_figure_list(local_data_path)
# local_label_list=get_figure_list(local_label_path)

#get all domains we need to interpolate
target_path=[r"data_after_pca/domain0/data/",r"data_after_pca/domain1/data/",r"data_after_pca/domain2/data/"
,r"data_after_pca/domain3/data/",r"data_after_pca/domain4/data/",r"data_after_pca/domain5/data/"]

#get the number of each domain
target_num=1

im_target_list=[]

#parameters we can change
#select 1 fig from each client
#use for naming the output figures
figure_name=0

domain_fig_num=target_num
for i in range(len(target_path)):
    for root, dirs, files in os.walk(target_path[i]):
            for j in range(target_num):
                #use 10 figures in each domain to int
                figure_path=root+files[j]
                im_target_list.append(figure_path)

#create a dir to store the figures
spectrum_fig_path=r'./Spectrum/'
if os.path.exists(spectrum_fig_path)==0:
    os.makedirs(spectrum_fig_path)

print(im_target_list)

for client_idx,im_trg in enumerate(im_target_list):

    img = cv2.imread(im_trg)		#读取通道顺序为B、G、R
    b,g,r = cv2.split(img)	
    img = cv2.merge([r,g,b])
    plt.figure(figsize=(40, 20)); 
    plt.subplot(141),plt.imshow(img),plt.title('picture')

    #根据公式转成灰度图
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #显示灰度图
    plt.subplot(142),plt.imshow(img,'gray'),plt.title('original')
    
    #进行傅立叶变换，并显示结果
    fft2 = np.fft.fft2(img)
    #将图像变换的原点移动到频域矩形的中心，并显示效果
    shift2center = np.fft.fftshift(fft2)
        
    #频谱相位图
    Picture_Phase_Specture = np.angle(fft2)*180/np.pi
    plt.subplot(143),plt.imshow(Picture_Phase_Specture,'gray'),plt.title('Phase_Specture')

    #频谱幅度图
    Picture_Amp_Specture = np.log(1 + np.abs(shift2center))/np.max(np.log(1 + np.abs(shift2center)))
    plt.subplot(144),plt.imshow(Picture_Amp_Specture,'gray'),plt.title('Amp_Specture')

plt.show()
