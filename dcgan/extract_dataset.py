#%%
import tarfile

import numpy as np
import tarfile
from PIL import Image
import glob
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
def usingPILandShrink(f,size): 
    im = Image.open(f)  
    im.draft('RGB',(size,size))
    return np.asarray(im)

def load_images(n,path,size):    
  dataset = []
  it = 0
  for filename in Path(path).glob("*.jpg"):
    if it <= n:
      try:
        im=usingPILandShrink(filename,size)
        im = im.astype('float32')
        im = (im-127.5)/127.5
        im = cv2.resize(im, dsize=(size,size), interpolation=cv2.INTER_CUBIC)
        dataset.append(im)
      except:
        continue
      it += 1
      if it % 1000==0:
          print(it)
    else:
      break
  return dataset

dataset = load_images(20000,"C:/SAMUEL/Centrale/Automatants/cat/cats",64)
dataset = np.asarray(dataset)
np.save("C:/SAMUEL/Centrale/Automatants/cat_dataset.npy",dataset)
#%%
dataset = np.load("C:/SAMUEL/Centrale/Automatants/cat_dataset.npy")
#%%
def plot(X,n):
    fig = plt.figure(figsize=(12,12))
    for i in range(n*n):
        plt.subplot(n,n,i+1)
        plt.axis('off')
        plt.imshow(X[i+25]*0.5+0.5)
    plt.show()
plot(dataset,5)
