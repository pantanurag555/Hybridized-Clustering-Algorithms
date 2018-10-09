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

def ffa_clustering(centroids,numclusters,num_fireflies):
    num_data=height*width
    m=2
    mem=np.zeros((num_fireflies,numclusters,num_data))
    for i in range(num_fireflies):
        tmp1=np.zeros(num_data)
        for k in range(numclusters):
            distki=abs(pixels-centroids[i][k])
            tmp2=np.zeros(num_data)
            for j in range(numclusters):
                distkj=abs(pixels-centroids[i][j])
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
            dist=(pixels-centroids[i][j])**2
            tmp=(mem[i][j]**m)*dist
            add=add+np.sum(tmp)
        err.append(add)
    return err