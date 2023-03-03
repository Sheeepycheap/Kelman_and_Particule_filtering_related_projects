import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
import matplotlib.image as mpimg
from PIL import Image
import math
from scipy.stats import norm
from numpy.random import random
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
import os
from scipy.stats import expon 


def multinomial_resample(weights):
    """ This is the naive form of roulette sampling where we compute the
    cumulative sum of the weights and then use binary search to select the
    resampled point based on a uniformly distributed random number. Run time
    is O(n log n). You do not want to use this algorithm in practice; for some
    reason it is popular in blogs and online courses so I included it for
    reference.
   Parameters
   ----------
    weights : list-like of float
        list of weights as floats
    Returns
    -------
    indexes : ndarray of ints
        array of indexes into the weights defining the resample. i.e. the
        index of the zeroth resample is indexes[0], etc.
    """
    weights=weights.T
    cumulative_sum = np.cumsum(weights)
    #cumulative_sum[-1] = 1.  # avoid round-off errors: ensures sum is exactly one
    #print ( np.searchsorted(cumulative_sum, random(len(weights))))
    return np.searchsorted(cumulative_sum, random(len(weights)))


def lecture_image(tt) :

    SEQUENCE = "./images/"
    START = 1
    #charge le nom des images de la sÃ©quence
    filenames = os.listdir(SEQUENCE)
    T = len(filenames)
    #print(T)
    #print(filenames)
    #charge la premiere image dans â€™imâ€™
    
   # print(str(SEQUENCE)+str(filenames[tt]))
    #im = mpimg.imread(str(SEQUENCE)+str(filenames[tt]))
    im=Image.open((str(SEQUENCE)+str(filenames[tt])))
    #affiche â€™imâ€™
    #fig = figure;
    #set(fig,'Units','Normalized','Position',[0 0 1 1]);
    #set(gcf, 'DoubleBuffer', 'on') ;
    plt.imshow(im)
    
    #plt.show()

    
    return(im,filenames,T,SEQUENCE)

def selectionner_zone() :

    #lecture_image()
    print('Cliquer 4 points dans l image pour definir la zone a suivre.') ;
    zone = np.zeros([2,4])
 #   print(zone)
    compteur=0
    while(compteur != 4):
        res = plt.ginput(1)
        a=res[0]
        #print(type(a))
        zone[0,compteur] = a[0]
        zone[1,compteur] = a[1]   
        plt.plot(a[0],a[1],marker='X',color='red') 
        compteur = compteur+1 

    #print(zone)
    newzone = np.zeros([2,4]) ;
    newzone[0, :] = np.sort(zone[0, :]) 
    newzone[1, :] = np.sort(zone[1, :])
    
    zoneAT = np.zeros([4])
    zoneAT[0] = newzone[0,0]
    zoneAT[1] = newzone[1,0]
    zoneAT[2] = newzone[0,3]-newzone[0,0] 
    zoneAT[3] = newzone[1,3]-newzone[1,0] 
    #affichage du rectangle
    #print(zoneAT)
    xy=(zoneAT[0],zoneAT[1])
    rect=ptch.Rectangle(xy,zoneAT[2],zoneAT[3],linewidth=3,edgecolor='red',facecolor='None') 
    #plt.Rectangle(zoneAT[0:1],zoneAT[2],zoneAT[3])
    currentAxis = plt.gca()
    currentAxis.add_patch(rect)
    plt.show(block=False)
    return(zoneAT)


def rgb2ind(im,nb) :
    #nb = nombre de couleurs ou kmeans qui contient la carte de couleur de l'image de rÃ©fÃ©rence
    
    image=np.array(im,dtype=np.float64)/255
    w,h,d=original_shape=tuple(image.shape)
    image_array=np.reshape(image,(w*h,d))
    image_array_sample=shuffle(image_array,random_state=0)[:1000]
   # print(type(image_array))
    if type(nb)==int :
        kmeans=KMeans(n_clusters=nb,random_state=0).fit(image_array_sample)
    else :
        kmeans=nb
            
    labels=kmeans.predict(image_array)
    #print(labels)
    image=recreate_image(kmeans.cluster_centers_,labels,w,h)
    #print(image)
    return(Image.fromarray(image.astype('uint8')),kmeans)

def recreate_image(codebook,labels,w,h):
    d=codebook.shape[1]
    #image=np.zeros((w,h,d))
    image=np.zeros((w,h))
    label_idx=0
    for i in range(w):
        for j in range(h):
            #image[i][j]=codebook[labels[label_idx]]*255
            image[i][j]=labels[label_idx]
            #print(image[i][j])
            label_idx+=1

    return image



def calcul_histogramme(im,zoneAT,Nb):

    #print(zoneAT)
    box=(zoneAT[0],zoneAT[1],zoneAT[0]+zoneAT[2],zoneAT[1]+zoneAT[3])
    #print(box)
    littleim = im.crop(box)
##    plt.imshow(littleim)
##    plt.show()
    new_im,kmeans= rgb2ind(littleim,Nb)
    histogramme=np.asarray(new_im.histogram())
##  print(histogramme)
    histogramme=histogramme/np.sum(histogramme)
  #  print(new_im)
    return (new_im,kmeans,histogramme)

N=50
Nb=30
ecart_type=np.sqrt(600)
lambda_im=20
c1=20
c2=20
C=np.diag([c1,c2])  

[im,filenames,T,SEQUENCE]=lecture_image(tt=0)   
zoneAT=selectionner_zone() # zoneAT = [X1,X2,l,L] X1 = abscisse X2 = ordonnée
print(zoneAT)
l = zoneAT[2]
L = zoneAT[3]
new_im,kmeans,histo_ref=calcul_histogramme(im,zoneAT,Nb)
#print(histo_ref)
plt.imshow(im)
rect=ptch.Rectangle((zoneAT[0],zoneAT[1]),zoneAT[2],zoneAT[3],linewidth=3,edgecolor='red',facecolor='None') 
currentAxis = plt.gca()
currentAxis.add_patch(rect)
plt.pause(0.5)

def particules(zoneAT,N) :
    part = np.zeros((2,N))
    for i in range(N) :
        part[0][i] = np.random.normal(zoneAT[0],ecart_type)
        part[1][i] = np.random.normal(zoneAT[1],ecart_type)
    return part

# test = particules(zoneAT,T) 
# print(test)
# test = np.append(test,[zoneAT[2]])
# test = np.append(test,[zoneAT[3]])
# print(test)
# print(test[0:T])
# print(test[T:-2])

def distance(q_prime,Nb) :
    res = 0 
    for i in range(Nb) :
        res += np.sqrt(histo_ref[i]*q_prime[i])
    D = np.sqrt(1 - res)
    return D

print(distance(histo_ref,Nb))

def construire_particule(temp,N) :
    temp_abs = temp[0:N]
    temp_ordo = temp[N:-2]
    l = temp[-2]
    L = temp[-1]
    X_t = np.zeros((N,4))

    for i in range(N):
        X_t[i][0] = temp_abs[i]
        X_t[i][1] = temp_ordo[i]
        X_t[i][2] = l 
        X_t[i][3] = L
    
    return X_t

def histo_particules(X_t,i,imm,Nb) : 
    zone = X_t[i]
    histo = calcul_histogramme(imm,zone,Nb)[2]
    return histo

def filtrage_particulaire(zoneAT,N,T,Nb):
    X_abs = np.zeros((N,T))
    X_ordo = np.zeros((N,T))
    W_abs = np.zeros((N,T))
    W_ordo = np.zeros((N,T))
    X_est_abs = np.zeros((1,T)).T
    X_est_ordo = np.zeros((1,T)).T  
    X_est_abs[0] = zoneAT[0]
    X_est_ordo[0] = zoneAT[1]
    temp = particules(zoneAT,N)
    temp = np.append(temp,[zoneAT[2]])
    temp = np.append(temp,[zoneAT[3]])
    X_t = construire_particule(temp,N)
    X_t[0] = zoneAT
    q_prime_par_particule = []
    D = []
    for i in range(N) : 
        q_prime_i = histo_particules(X_t,i,im,Nb)
        q_prime_par_particule.append(q_prime_i)
        D.append(distance(q_prime_i,Nb))
    print(D)
    scale = [-1*lambda_im*(D[i]**2) for i in range(len(D))]
    


    # w_t_abs = expon.pdf(X_t[:,0],scale=scale)
    w_t_abs = np.exp(scale)
    w_t_abs = w_t_abs/(w_t_abs.sum())
    print('w_t_abs:')
    print(len(w_t_abs))
    print(w_t_abs)
    # w_t_ordo = expon.pdf(X_t[:,1],scale=scale)
    w_t_ordo = np.exp(scale)
    w_t_ordo = w_t_ordo/(w_t_ordo.sum())
    W_abs[:,1] = w_t_abs.flatten()
    W_ordo[:,1] = w_t_ordo.flatten()
    X_abs[:,1] = X_t[:,0]
    X_ordo[:,1] = X_t[:,1]
    
    X_est_abs[1] = (W_abs[:,1]*X_abs[:,1]).sum()
    X_est_ordo[1] = (W_ordo[:,1]*X_ordo[:,1]).sum()
    print(X_est_abs[1],X_est_ordo[1])
    for t in range(2,T) :
        print(t)
        new_image = lecture_image(tt=t)[0]
    
        A_abs = np.random.choice(range(N),N,p=w_t_abs.flatten())
        print(A_abs)
        reech_abs = np.array([X_t[:,0][A_abs]]).T
        
        A_ordo = np.random.choice(range(N),N,p=w_t_ordo.flatten())
        reech_ordo = np.array([X_t[:,1][A_ordo]]).T

        X_t[:,0] = (reech_abs + np.random.normal(0,c1,(N,1))).flatten()
        X_t[:,1] = (reech_ordo + np.random.normal(0,c2,(N,1))).flatten()
        q_prime_par_particule = []
        D = []
        for i in range(N) : 
            q_prime_i = histo_particules(X_t,i,new_image,Nb)
            q_prime_par_particule.append(q_prime_i)
            D.append(distance(q_prime_i,Nb))
        scale = [-1*lambda_im*(D[i]**2) for i in range(len(D))]

        # w_t_abs = expon.pdf(X_t[:,0],scale=scale)
        w_t_abs = np.exp(scale)
        w_t_abs = w_t_abs/(w_t_abs.sum())
        # w_t_ordo = expon.pdf(X_t[:,1],scale=scale)
        w_t_ordo = np.exp(scale)
        w_t_ordo = w_t_ordo/(w_t_ordo.sum())
        W_abs[:,t] = w_t_abs.flatten()
        W_ordo[:,t] = w_t_ordo.flatten()
        X_abs[:,t] = X_t[:,0]
        X_ordo[:,t] = X_t[:,1]

        X_est_abs[t] = (W_abs[:,t]*X_abs[:,t]).sum()
        X_est_ordo[t] = (W_ordo[:,t]*X_ordo[:,t]).sum()

        print(X_est_abs[t],X_est_ordo[t])

        plt.imshow(new_image)
        rect=ptch.Rectangle((X_est_abs[t],X_est_ordo[t]),l,L,linewidth=3,edgecolor='red',facecolor='None') 
        currentAxis = plt.gca()
        currentAxis.add_patch(rect)
        plt.pause(0.1)
        currentAxis.remove()

    # X_est_abs = (W_abs*X_abs).sum(0)
    # X_est_ordo = (W_ordo*X_ordo).sum(0)

    return (X_est_abs,X_est_ordo)

X_est_abs, X_est_ordo = filtrage_particulaire(zoneAT,N,T,Nb)
X_est = np.zeros((T,4))
X_est[:,0] = X_est_abs[0]
X_est[:,1] = X_est_abs[0]
X_est[:,2] = l 
X_est[:,3] = L 
#print(X_est)
    
    
    





