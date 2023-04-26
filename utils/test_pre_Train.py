import os
import cv2
import random
import numpy as np
from os.path import join
from tqdm import tqdm
import matplotlib.pyplot as plt
def adjust_contrast_brightness(img, contrast: float = 1.0, brightness: float = 0.0):
    # Apply the contrast and brightness adjustments to the image
    return np.clip((img * contrast + brightness), 0, 1)

def adjust_illumination_map(illu_map: np.ndarray,
                            power: int = 8,
                            contrast: float = 1.1,
                            brightness: float = 0.0,
                            weight: float = 1.0):
    illu_map = np.power(illu_map, power)
    illu_map = adjust_contrast_brightness(illu_map, contrast=contrast, brightness=brightness)
    if weight == 1.0:
        illu_full = np.ones(illu_map.shape)
        illu_map = (1-weight)*illu_full + weight*illu_map
    return illu_map

def add_shadow(image_np: np.ndarray, illu_map: np.ndarray):
    image_shadow = image_np * illu_map
    return (image_shadow).astype("uint8")

def cla(image):
    iclip = max(0.1,min(2,random.gauss(1,0.5)))

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=iclip, tileGridSize=(8,8))
    cl = clahe.apply(l)
    equalized = cv2.merge((cl,a,b))
    equalized = cv2.cvtColor(equalized, cv2.COLOR_LAB2BGR)
    return equalized

if __name__ == '__main__':
    image_folder = "/home/data2/viton_isac/photo"
    illu_map_folder = "/home/data2/viton_isac/illu_map"
    p=16
    for i in tqdm(range(4000)):
        image = cv2.imread(join(image_folder, str(p)+".jpg"))
        image = cla(image)
        illu_map = np.load(join(illu_map_folder, str(p)+".npz"))["arr_0"]
        
        power = random.gauss(7, 1)       #12,  1 
        contrast = random.gauss(1.0, 0.05)#1 ,  0.1
        brightness = random.gauss(0, 0.1) #0 ,  0.2
        illu_map = adjust_illumination_map(illu_map, power=power, contrast=contrast, brightness=brightness)
        illu_map = 1 / (1 + np.exp(-(illu_map+10)**2))        

        image_aug = add_shadow(image, illu_map)
        image_aug = image_aug[:, :, ::-1]   
        plt.imsave('/home/isac/data/shadow_test/'+str(i)+".png",image_aug)
        if i > 9:break

