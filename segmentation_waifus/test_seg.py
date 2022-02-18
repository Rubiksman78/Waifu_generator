from pathlib import Path
from PIL import Image
import numpy as np
import cv2
from Unet import *

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
                im = im.astype('uint8')
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

dataset = load_images(16,"D:/Datasets/dataset_92000_256",128)
test_dataset = np.asarray(dataset)
#np.save("unet_test_dataset.npy",test_dataset)
u_net.load_weights('u_net_weights.h5')
preds = u_net.predict(test_dataset)
for i in range(16):
    images = [test_dataset[i],create_mask(np.expand_dims(preds[i],axis=0))]
    figure = plt.figure(figsize=(5,5))
    for j in range(2):
        plt.subplot(1,2,j+1)
        plt.axis('off')
        plt.imshow(images[j])
plt.show()