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
    wup = 0.1
    wlow = 0.9
    m=2
    print(centroids)
    while cond==0:
        loop=loop+1
        mem = np.zeros((num_centroids, num_data))
        b_mem = np.zeros((num_centroids, num_data))
        v = np.zeros(num_centroids)
        tmp = np.zeros((num_centroids,num_data))
        f_max = np.zeros(num_data)
        s_max = np.zeros(num_data)
        f_max_n = np.zeros(num_data)
        s_max_n = np.zeros(num_data)
        diff = np.zeros(num_data)

        for i in range(num_centroids):
            distki = abs(pixels - centroids[i])
            for j in range(num_centroids):
                distkj = abs(pixels - centroids[j])
                tmp[i]=tmp[i]+((distki/distkj)**(2/(m-1)))
            tmp[i]=1/tmp[i]

        for i in range(num_data):
            f_max_n[i] = s_max_n[i] = -1
            for j in range(num_centroids):
                if(tmp[j][i]>f_max[i]):
                    s_max[i] = f_max[i]
                    s_max_n[i] = f_max_n[i]
                    f_max[i] = tmp[j][i]
                    f_max_n[i] = j
                elif(tmp[j][i]>=s_max[i]):
                    s_max[i] = tmp[j][i]
                    s_max_n[i] = j
            diff[i] = f_max[i]-s_max[i]
        threshold = np.sum(diff) / num_data

        for i in range(num_data):
            if diff[i] <= threshold:
                b_mem[int(f_max_n[i])][i] = tmp[int(f_max_n[i])][i]
                b_mem[int(s_max_n[i])][i] = tmp[int(s_max_n[i])][i]
            else:
                mem[int(f_max_n[i])][i] = 1

        for i in range(num_centroids):
            if(np.sum(b_mem[i])==0):
                v[i] = np.sum(pixels*mem[i]) / np.sum(mem[i])
            elif(np.sum(mem[i])==0):
                v[i] = np.sum((b_mem**m)*pixels)/np.sum(b_mem[i]**m)
            else:
                v[i] = wlow * (np.sum(pixels*mem[i]) / np.sum(mem[i])) + wup * (np.sum((b_mem[i]**m)*pixels)/np.sum(b_mem[i]**m))

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
    max_c=np.zeros(num_centroids)

    s_v = np.zeros(num_centroids)
    mem = np.zeros((num_centroids, num_data))
    b_mem = np.zeros((num_centroids, num_data))
    t_b_mem = np.zeros((num_centroids, num_data))
    tmp = np.zeros((num_centroids, num_data))
    f_max = np.zeros(num_data)
    s_max = np.zeros(num_data)
    f_max_n = np.zeros(num_data)
    s_max_n = np.zeros(num_data)
    diff = np.zeros(num_data)
    wup = 0.1
    wlow = 0.9

    f = lambda a: (abs(a) + a) / 2

    for i in range(num_centroids):
        distki = abs(pixels - centroids[i])
        for j in range(num_centroids):
            distkj = abs(pixels - centroids[j])
            tmp[i] = tmp[i] + ((distki / distkj) ** (2 / (m - 1)))
        tmp[i] = 1 / tmp[i]

    for i in range(num_data):
        f_max_n[i] = s_max_n[i] = -1
        for j in range(num_centroids):
            if (tmp[j][i] > f_max[i]):
                s_max[i] = f_max[i]
                s_max_n[i] = f_max_n[i]
                f_max[i] = tmp[j][i]
                f_max_n[i] = j
            elif (tmp[j][i] >= s_max[i]):
                s_max[i] = tmp[j][i]
                s_max_n[i] = j
        diff[i] = f_max[i] - s_max[i]
    threshold = np.sum(diff) / num_data

    for i in range(num_data):
        if diff[i] <= threshold:
            t_b_mem[int(f_max_n[i])][i] = tmp[int(f_max_n[i])][i]
            b_mem[int(f_max_n[i])][i] = 1
            t_b_mem[int(s_max_n[i])][i] = tmp[int(s_max_n[i])][i]
            b_mem[int(s_max_n[i])][i] = 1
        else:
            mem[int(f_max_n[i])][i] = 1

    for i in range(num_centroids):
        if (np.sum(b_mem[i]) == 0):
            s_v[i] = np.sum(f((pixels * mem[i])-centroids[i])**2) / np.sum(mem[i])
        elif (np.sum(mem[i]) == 0):
            s_v[i] = np.sum((t_b_mem[i]**m)*(f((pixels * b_mem[i])-centroids[i])**2)) / np.sum(t_b_mem[i]**m)
        else:
            s_v[i] = wlow * (np.sum(f((pixels * mem[i])-centroids[i])**2) / np.sum(mem[i])) + wup * (np.sum((t_b_mem[i]**m)*(f((pixels * b_mem[i])-centroids[i])**2)) / np.sum(t_b_mem[i]**m))

    for i in range(num_centroids):
        for j in range(num_centroids):
            if i!=j:
                dist=abs(centroids[i]-centroids[j])
                tmp=(s_v[i]+s_v[j])/dist
                if tmp>max_c[i]:
                    max_c[i]=tmp
    db_val=np.sum(max_c)/num_centroids
    print("DB index value is : ",db_val)
    return

def d(centroids,num_centroids):
    num_data = height*width
    m = 2
    m_s_v = 0

    s_v = np.zeros(num_centroids)
    mem = np.zeros((num_centroids, num_data))
    b_mem = np.zeros((num_centroids, num_data))
    t_b_mem = np.zeros((num_centroids, num_data))
    tmp = np.zeros((num_centroids, num_data))
    f_max = np.zeros(num_data)
    s_max = np.zeros(num_data)
    f_max_n = np.zeros(num_data)
    s_max_n = np.zeros(num_data)
    diff = np.zeros(num_data)
    wup = 0.1
    wlow = 0.9

    f = lambda a: (abs(a) + a) / 2

    for i in range(num_centroids):
        distki = abs(pixels - centroids[i])
        for j in range(num_centroids):
            distkj = abs(pixels - centroids[j])
            tmp[i] = tmp[i] + ((distki / distkj) ** (2 / (m - 1)))
        tmp[i] = 1 / tmp[i]

    for i in range(num_data):
        f_max_n[i] = s_max_n[i] = -1
        for j in range(num_centroids):
            if (tmp[j][i] > f_max[i]):
                s_max[i] = f_max[i]
                s_max_n[i] = f_max_n[i]
                f_max[i] = tmp[j][i]
                f_max_n[i] = j
            elif (tmp[j][i] >= s_max[i]):
                s_max[i] = tmp[j][i]
                s_max_n[i] = j
        diff[i] = f_max[i] - s_max[i]
    threshold = np.sum(diff) / num_data

    for i in range(num_data):
        if diff[i] <= threshold:
            t_b_mem[int(f_max_n[i])][i] = tmp[int(f_max_n[i])][i]
            b_mem[int(f_max_n[i])][i] = 1
            t_b_mem[int(s_max_n[i])][i] = tmp[int(s_max_n[i])][i]
            b_mem[int(s_max_n[i])][i] = 1
        else:
            mem[int(f_max_n[i])][i] = 1

    for i in range(num_centroids):
        if (np.sum(b_mem[i]) == 0):
            s_v[i] = np.sum(f((pixels * mem[i])-centroids[i])**2) / np.sum(mem[i])
        elif (np.sum(mem[i]) == 0):
            s_v[i] = np.sum((t_b_mem[i]**m)*(f((pixels * b_mem[i])-centroids[i])**2)) / np.sum(t_b_mem[i]**m)
        else:
            s_v[i] = wlow * (np.sum(f((pixels * mem[i])-centroids[i])**2) / np.sum(mem[i])) + wup * (np.sum((t_b_mem[i]**m)*(f((pixels * b_mem[i])-centroids[i])**2)) / np.sum(t_b_mem[i]**m))

    for i in range(num_centroids):
        if s_v[i]>m_s_v:
            m_s_v = s_v[i]
    d_val=sys.maxsize

    for i in range(num_centroids):
        for j in range(num_centroids):
            if i!=j:
                dist=abs(centroids[i]-centroids[j])
                final = dist/m_s_v
                if final<d_val:
                    d_val=final
    print("Dunn index value is : ",d_val)
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