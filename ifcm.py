import sys
import copy
import math
import numpy as np
from PIL import Image
from collections import namedtuple
from operator import attrgetter
import random
import matplotlib.pyplot as plt

def clustering(centroids,num_centroids):
    num_data=height*width
    cond=0
    e=0.01
    loop=0
    m=2
    lmb=2
    alp = 350
    while cond==0:
        loop=loop+1
        v = np.zeros(num_centroids)
        tmp=np.zeros((num_centroids,num_data))
        i_tmp = np.zeros((num_centroids, num_data))

        for i in range(num_centroids):
            distki = abs(pixels - centroids[i])
            for j in range(num_centroids):
                distkj = abs(pixels - centroids[j])
                tmp[i]=tmp[i]+((distki/distkj)**(2/(m-1)))
            tmp[i]=1/tmp[i]

        for i in range(num_centroids):
            i_tmp[i]=1-((1-tmp[i])**alp)**(1/alp)

        for i in range(num_centroids):
            num=np.sum((i_tmp[i]**m)*pixels)
            den=np.sum(i_tmp[i]**m)
            v[i]=num/den

        if np.average(abs(centroids-v))<e or loop==150:
            print("Final centroids are : ",v)
            cond=cond+1
        else:
            print(v)
            centroids=np.copy(v)
    print(loop)
    db(v,num_centroids)
    d(v,num_centroids)
    draw_img(v, num_centroids)
    return

def db(centroids,num_centroids):
    num_data = height * width
    m = 2
    lmb = 2
    alp = 350
    max_c = np.zeros(num_centroids)
    tmp = np.zeros((num_centroids, num_data))
    i_tmp = np.zeros((num_centroids, num_data))

    for i in range(num_centroids):
        distki = abs(pixels - centroids[i])
        for j in range(num_centroids):
            distkj = abs(pixels - centroids[j])
            tmp[i] = tmp[i] + ((distki / distkj) ** (2 / (m - 1)))
        tmp[i] = 1 / tmp[i]

    for i in range(num_centroids):
        i_tmp[i]=1-((1-tmp[i])**alp)**(1/alp)

    for i in range(num_centroids):
        for j in range(num_centroids):
            if i != j:
                dist = abs(centroids[i] - centroids[j])
                i_spr = np.sum((i_tmp[i] ** m) * ((pixels - centroids[i]) ** 2)) / np.sum(i_tmp[i] ** m)
                j_spr = np.sum((i_tmp[j] ** m) * ((pixels - centroids[j]) ** 2)) / np.sum(i_tmp[j] ** m)
                final = (i_spr + j_spr) / dist
                if final > max_c[i]:
                    max_c[i] = final
    db_val = np.sum(max_c) / num_centroids
    print("DB index value is : ", db_val)
    return

def d(centroids,num_centroids):
    num_data = height * width
    m = 2
    alp = 350
    lmb = 2
    final = 0
    tmp = np.zeros((num_centroids, num_data))
    i_tmp = np.zeros((num_centroids, num_data))

    for i in range(num_centroids):
        distki = abs(pixels - centroids[i])
        for j in range(num_centroids):
            distkj = abs(pixels - centroids[j])
            tmp[i] = tmp[i] + ((distki / distkj) ** (2 / (m - 1)))
        tmp[i] = 1 / tmp[i]

    for i in range(num_centroids):
        i_tmp[i]=1-((1-tmp[i])**alp)**(1/alp)

    for i in range(num_centroids):
        i_spr = np.sum((i_tmp[i] ** m) * ((pixels - centroids[i]) ** 2)) / np.sum(i_tmp[i] ** m)
        if i_spr > final:
            final = i_spr
    i_spr = final
    d_val = sys.maxsize
    for i in range(num_centroids):
        for j in range(num_centroids):
            if i != j:
                dist = abs(centroids[i] - centroids[j])
                final = dist / i_spr
                if final < d_val:
                    d_val = final
    print("Dunn index value is : ", d_val)
    return

def draw_img(centroids,num_centroids):
    num_data = height * width
    m = 2
    tmp = np.zeros((num_centroids, num_data))
    for i in range(num_centroids):
        distki = abs(pixels - centroids[i])
        for j in range(num_centroids):
            distkj = abs(pixels - centroids[j])
            tmp[i] = tmp[i] + ((distki / distkj) ** (2 / (m - 1)))
        tmp[i] = 1 / tmp[i]
    img=np.zeros((num_centroids,height,width))
    for k in range(num_centroids):
        z=0
        for i in range(height):
            for j in range(width):
                img[k][i][j]=tmp[k][z]
                z=z+1
    fig = plt.figure()
    for i in range(num_centroids):
        ax = fig.add_subplot(2,2,i+1)
        ax.imshow(img[i])
    plt.show()
    return

im=Image.open("C:/Users/User/Desktop/Capture.jpg")
pixels = list(im.getdata())
width,height=im.size
for i in range(len(pixels)):
   pixels[i]=int(round(sum(pixels[i]) / float(len(pixels[i]))))
pixels=np.array(pixels)
c=4
r=np.zeros(c)
for i in range(c):
    r[i]=random.uniform(0,255)
clustering(r,c)
