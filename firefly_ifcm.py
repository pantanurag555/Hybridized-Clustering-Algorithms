import sys
import copy
import math
import numpy as np
from PIL import Image
from collections import namedtuple
from operator import attrgetter
import random
import matplotlib.pyplot as plt
firefly = namedtuple("firefly", "error intensity")

def ffa(numfireflies):
    numclusters=int(input("Enter no. of clusters: "))
    maxepochs=50
    ming=0.0
    maxg=255
    b0=0.0
    g=1.0
    a=0.2
    displayinterval=maxepochs/10
    besterror=sys.maxsize
    swarm=[]
    gs=np.empty((0))
    for i in range(numfireflies):
        tmp = np.empty((0))
        for j in range(numclusters):
            g=(maxg-ming)*random.uniform(0,1)+ming
            tmp=np.append(tmp,g)
            swarm.append(firefly(0.0,0.0))
        if i==0:
            gs=tmp
        else:
            gs=np.vstack([gs,tmp])
    err=ffa_clustering(gs,numclusters,numfireflies)
    print(gs)
    print(err)
    for i in range(numfireflies):
        swarm[i]=swarm[i]._replace(error=err[i])
        swarm[i]=swarm[i]._replace(intensity=1/(swarm[i].error+1))
        if swarm[i].error<besterror:
            besterror=swarm[i].error
            bestposition=np.copy(gs[i])
    epoch=0
    while epoch<maxepochs:
        if epoch%displayinterval==0 and epoch<maxepochs:
            print("epoch = ",epoch," error = ",besterror)
        key1 = 0
        key2 = 0
        key3 = 0
        for i in range(1, numfireflies):
            key1 = swarm[i].error
            key2 = swarm[i].intensity
            key3 = np.copy(gs[i])
            j = i - 1
            while j >= 0 and swarm[j].error > key1:
                swarm[j + 1] = swarm[j + 1]._replace(error=swarm[j].error)
                swarm[j + 1] = swarm[j + 1]._replace(intensity=swarm[j].intensity)
                gs[j + 1][:] = gs[j][:]
                j = j - 1
            swarm[j + 1] = swarm[j + 1]._replace(error=key1)
            swarm[j + 1] = swarm[j + 1]._replace(intensity=key2)
            gs[j + 1][:] = key3[:]
        for i in range(numfireflies):
            for j in range(1,numclusters):
                key1=gs[i][j]
                k=j-1
                while k>=0 and gs[i][k]>key1:
                    gs[i][k+1]=gs[i][k]
                    k=k-1
                gs[i][k+1]=key1
        print(gs)
        if swarm[0].error<besterror:
            besterror=swarm[0].error
            bestposition=gs[0]
        for i in range(numfireflies):
            key1 = swarm[i].error
            key2 = swarm[i].intensity
            key3 = np.copy(gs[i])
            if i==0:
                for k in range(numclusters):
                    gs[i][k]=gs[i][k]+random.uniform(0,1)
            else:
                 #r=np.sqrt(((gs[i]-gs[j])**2))
                 #beta=b0*math.exp(-g*np.sum(r)*np.sum(r))
                 beta=0.01
                 gs[i]=gs[i]+beta*(gs[0]-gs[i])
                 gs[i]=gs[i]+a*(random.uniform(0,1)-0.5)
                 for k in range(numclusters):
                    if gs[i][k]<ming or gs[i][k]>maxg:
                         gs[i][k]=(maxg-ming)*random.uniform(0,1)+ming
            err = ffa_clustering(gs,numclusters,numfireflies)
            swarm[i] = swarm[i]._replace(error=err[i])
            swarm[i] = swarm[i]._replace(intensity=1 / (swarm[i].error + 1))
            if key1<swarm[i].error:
                swarm[i] = swarm[i]._replace(error=key1)
                swarm[i] = swarm[i]._replace(intensity=key2)
                gs[i][:] = key3[:]
        for i in range(numfireflies):
            print(swarm[i].error)
        epoch=epoch+1
        print(epoch)
    key1 = 0
    key2 = 0
    key3 = 0
    for i in range(1, numfireflies):
        key1 = swarm[i].error
        key2 = swarm[i].intensity
        key3 = gs[i]
        j = i - 1
        while j >= 0 and swarm[j].error > key1:
            swarm[j + 1] = swarm[j + 1]._replace(error=swarm[j].error)
            swarm[j + 1] = swarm[j + 1]._replace(intensity=swarm[j].intensity)
            gs[j + 1] = gs[j]
            j = j - 1
        swarm[j + 1] = swarm[j + 1]._replace(error=key1)
        swarm[j + 1] = swarm[j + 1]._replace(intensity=key2)
        gs[j + 1] = key3
    for i in range(numfireflies):
        for j in range(1,numclusters):
            key1 = gs[i][j]
            k = j - 1
            while k >= 0 and gs[i][k] > key1:
                gs[i][k + 1] = gs[i][k]
                k = k - 1
            gs[i][k + 1] = key1
    print(gs)
    if swarm[0].error < besterror:
        besterror = swarm[0].error
        bestposition = gs[0]
    clustering(bestposition,numclusters)
    return

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

def ffa_clustering(centroids,numclusters,num_fireflies):
    num_data=height*width
    m=2
    sd = np.std(pixels)
    mem=np.zeros((num_fireflies,numclusters,num_data))
    for i in range(num_fireflies):
        tmp1=np.zeros(num_data)
        for k in range(numclusters):
            distki=2*(1-np.exp(-((pixels - centroids[i][k])**2)/(sd**2)))
            tmp2=np.zeros(num_data)
            for j in range(numclusters):
                distkj=2*(1-np.exp(-((pixels - centroids[i][j])**2)/(sd**2)))
                tmp2=tmp2+((distki/distkj)**(2/(m-1)))
            tmp2=1/tmp2
            if k==0:
                tmp1=tmp2
            else:
                tmp1=np.vstack([tmp1,tmp2])
        mem[i]=tmp1
    err=[]
    for i in range(num_fireflies):
        add=0
        for j in range(numclusters):
            dist=(2*(1-np.exp(-((pixels - centroids[i][j])**2)/(sd**2))))**2
            tmp=(mem[i][j]**m)*dist
            add=add+np.sum(tmp)
        err.append(add)
    return err

im=Image.open("C:/Users/User/Desktop/brain.tif")
pixels = list(im.getdata())
width,height=im.size
#for i in range(len(pixels)):
 #   pixels[i]=int(round(sum(pixels[i]) / float(len(pixels[i]))))
pixels=np.array(pixels)
ffa(15)