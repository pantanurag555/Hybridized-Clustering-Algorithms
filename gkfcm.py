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
    sd=np.std(pixels)
    m=2
    while cond==0:
        loop=loop+1
        v = np.zeros(num_centroids)
        tmp=np.zeros((num_centroids,num_data))
        for i in range(num_centroids):
            distki = 2*(1-np.exp(-((pixels - centroids[i])**2)/(sd**2)))
            for j in range(num_centroids):
                distkj = 2*(1-np.exp(-((pixels - centroids[j])**2)/(sd**2)))
                tmp[i]=tmp[i]+((distki/distkj)**(2/(m-1)))
            tmp[i]=1/tmp[i]
        for i in range(num_centroids):
            num=np.sum((tmp[i]**m)*pixels)
            den=np.sum(tmp[i]**m)
            v[i]=num/den
        if np.average(abs(centroids-v))<e or loop==150:
            print("Final centroids are : ",v)
            cond=cond+1
        else:
            print(v)
            centroids=np.copy(v)
    db(v, num_centroids)
    d(v, num_centroids)
    print(loop)
    draw_img(v, num_centroids)
    return

def db(centroids,num_centroids):
    num_data = height * width
    m=2
    sd = np.std(pixels)
    max_c=np.zeros(num_centroids)
    tmp = np.zeros((num_centroids,num_data))

    for i in range(num_centroids):
        distki = 2*(1-np.exp(-((pixels - centroids[i])**2)/(sd**2)))
        for j in range(num_centroids):
            distkj = 2*(1-np.exp(-((pixels - centroids[j])**2)/(sd**2)))
            tmp[i] = tmp[i] + ((distki / distkj) ** (2 / (m - 1)))
        tmp[i] = 1 / tmp[i]

    for i in range(num_centroids):
        for j in range(num_centroids):
            if i!=j:
                dist=abs(centroids[i]-centroids[j])
                distki = 2*(1-np.exp(-((pixels - centroids[i])**2)/(sd**2)))
                i_spr = np.sum((tmp[i]**m)*(distki**2))/np.sum(tmp[i]**m)
                distkj = 2*(1-np.exp(-((pixels - centroids[j])**2)/(sd**2)))
                j_spr = np.sum((tmp[j]**m)*(distkj**2))/np.sum(tmp[j]**m)
                final=(i_spr+j_spr)/dist
                if final>max_c[i]:
                    max_c[i]=final
    db_val=np.sum(max_c)/num_centroids
    print("DB index value is : ",db_val)
    return

def d(centroids,num_centroids):
    num_data=height*width
    m=2
    sd = np.std(pixels)
    final=0
    tmp = np.zeros((num_centroids, num_data))

    for i in range(num_centroids):
        distki = 2*(1-np.exp(-((pixels - centroids[i])**2)/(sd**2)))
        for j in range(num_centroids):
            distkj = 2*(1-np.exp(-((pixels - centroids[j])**2)/(sd**2)))
            tmp[i] = tmp[i] + ((distki / distkj) ** (2 / (m - 1)))
        tmp[i] = 1 / tmp[i]

    for i in range(num_centroids):
        distki = 2*(1-np.exp(-((pixels - centroids[i])**2)/(sd**2)))
        i_spr = np.sum((tmp[i] ** m) * (distki ** 2)) / np.sum(tmp[i] ** m)
        if i_spr>final:
            final=i_spr
    i_spr=final
    d_val=sys.maxsize
    for i in range(num_centroids):
        for j in range(num_centroids):
            if i!=j:
                dist=2*(1-np.exp(-((centroids[i] - centroids[j])**2)/(sd**2)))
                final=dist/i_spr
                if final<d_val:
                    d_val=final
    print("Dunn index value is : ",d_val)
    return

def draw_img(centroids,num_centroids):
    num_data = height * width
    m = 2
    sd = np.std(pixels)
    tmp = np.zeros((num_centroids, num_data))
    for i in range(num_centroids):
        distki = 2*(1-np.exp(-((pixels - centroids[i])**2)/(sd**2)))
        for j in range(num_centroids):
            distkj = 2*(1-np.exp(-((pixels - centroids[j])**2)/(sd**2)))
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