from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import cv2
import os

#统计文件目录下的文件个数
def get_path_num(file_dir):
    count=0
    for files in os.listdir(file_dir):
        count=count+1
    return count

img0 = cv2.imread(r'calculate/origin/0.bmp')
img0 = cv2.resize(img0, (384,384))
img_con = cv2.imread(r'calculate/origin/1.bmp')
img_con = cv2.resize(img_con, (384,384))

p0 = compare_psnr(img0, img_con)
s0 = compare_ssim(img0, img_con, multichannel=True)  # 对于多通道图像(RGB、HSV等)关键词multichannel要设置为True
m0 = compare_mse(img0, img_con)
print(p0,s0,m0)

domain_path=r'calculate/data/'
dom_len=get_path_num(domain_path)
fig_list=[]
for root, dirs, files in os.walk(domain_path):
    for i in range(dom_len):
        files[i]=files[i]
        fig_list.append(root+files[i])

l = len(fig_list)

p=[]
s=[]
m=[]
for i in range(l):
    p0 = compare_psnr(img0, cv2.imread(fig_list[i]))
    s0 = compare_ssim(img0, cv2.imread(fig_list[i]), multichannel=True)  # 对于多通道图像(RGB、HSV等)关键词multichannel要设置为True
    m0 = compare_mse(img0, cv2.imread(fig_list[i]))
    p.append(p0)
    s.append(s0)
    m.append(m0)

import numpy as np
print(np.mean(p))
print(np.mean(m))
print(np.mean(s))

import matplotlib.pyplot as plt

x=np.arange(1,len(p)+1)

# plot中参数的含义分别是横轴值，纵轴值，线的形状，颜色，透明度,线的宽度和标签
plt.figure(figsize=(12,5))
plt.title('PSNR',fontsize=20)
plt.scatter(x, p)

plt.figure(figsize=(12,5))
plt.title('SSIM',fontsize=20)
plt.scatter(x, s)

plt.figure(figsize=(12,5))
plt.title('MSE',fontsize=20)
plt.scatter(x, m)

plt.legend(loc="upper right")
plt.show()

