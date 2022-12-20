import numpy as np
from PIL import Image
import scipy.misc
import matplotlib.pyplot as plt

import numpy as np
import SimpleITK as sitk
import os
import numpy as np
from glob import glob
import time
import shutil
from PIL import Image

#统计文件目录下的文件个数
def get_path_num(file_dir):
    count=0
    for files in os.listdir(file_dir):
        count=count+1
    return count

def extract_amp_spectrum(trg_img):

    fft_trg_np = np.fft.fft2( trg_img, axes=(-2, -1) )
    #计算复杂参数的角度。复数由“ x + yi”表示，其中x和y是实数。角度由公式计算tan^{-1}(x/y).
    amp_target, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

    return amp_target

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

def freq_space_interpolation( local_img, amp_target, L=0 , ratio=0):

    local_img_np = local_img 

    # get fft of local sample
    fft_local_np = np.fft.fft2( local_img_np, axes=(-2, -1) )

    # extract amplitude and phase of local sample
    amp_local, pha_local = np.abs(fft_local_np), np.angle(fft_local_np)

    # swap the amplitude part of local image with target amplitude spectrum
    amp_local_ = amp_spectrum_swap( amp_local, amp_target, L=L , ratio=ratio)

    # get transformed image via inverse fft
    fft_local_ = amp_local_ * np.exp( 1j * pha_local )
    local_in_trg = np.fft.ifft2( fft_local_, axes=(-2, -1) )
    local_in_trg = np.real(local_in_trg)

    #figures after interpolation 
    return local_in_trg

def draw_image(image):
    plt.imshow(image, cmap='gray')
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    plt.xticks([])
    plt.yticks([])
    
    return 0

def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )

    return data

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

local_data_path=r"Training_data/data/"
local_label_path=r"Training_data/label/"

#get the number of the local figures
local_num=get_path_num(local_data_path)


#get all lists all we need
local_data_list=get_figure_list(local_data_path)
local_label_list=get_figure_list(local_label_path)

#get all domains we need to interpolate
target_path=[r"Domain1/data/",r"Domain2/data/",r"Domain3/data/"]

#get the number of each domain
target_num=get_path_num(target_path[0])

im_target_list=[]

#parameters we can change
#select 1 fig from n figures
n=10
#use for naming the output figures
figure_name=0
#int value list
int_list=[0.2,0.5,0.8]

domain_fig_num=target_num/n
for i in range(len(target_path)):
    for root, dirs, files in os.walk(target_path[i]):
            for j in range(target_num):
                #use 10 figures in each domain to int
                if j%n==0:
                    figure_path=root+files[j]
                    im_target_list.append(Image.open(figure_path))

#create a dir to store the figures
spectrum_fig_path=r'./Spectrum/'
if os.path.exists(spectrum_fig_path)==0:
    os.makedirs(spectrum_fig_path)

for client_idx,im_trg in enumerate(im_target_list):
    
    im_trg = im_trg.resize( (384,384), Image.BICUBIC )
    im_trg = np.asarray(im_trg, np.float32)
    im_trg = im_trg.transpose((2, 0, 1))

    L = 0.003    
    #visualize local data, target data, amplitude spectrum of target data
    plt.figure(figsize=(12,8))
    plt.subplot(1,2,1)
    draw_image((im_trg / 255).transpose((1, 2, 0)))
    plt.xlabel("Target Image (figure {})".format(client_idx+1), fontsize=12)
    
    # amplitude spectrum of target data
    amp_target = extract_amp_spectrum(im_trg)
    amp_target_shift = np.fft.fftshift( amp_target, axes=(-2, -1) )
    spectrum_fig=np.clip((np.log(amp_target_shift)/ np.max(np.log(amp_target_shift))).transpose((1, 2, 0)), 0, 1)
    plt.subplot(1,2,2)
    draw_image(np.clip((np.log(amp_target_shift)/ np.max(np.log(amp_target_shift))).transpose((1, 2, 0)), 0, 1))
    plt.xlabel("Target Amp (figure {})".format(client_idx+1), fontsize=12)
    fig_path=spectrum_fig_path+str(client_idx+1)+'.png'
    plt.savefig(fig_path)
