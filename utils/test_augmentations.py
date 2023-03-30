from layers import *
import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
trms = [ # commented out are not best augmentations
    #A.ChannelDropout(p=1), #####
    #A.ChannelShuffle(p=1),#####
    #A.ColorJitter(p=1),#####
    #A.Equalize(p=1),  #####
    #A.InvertImg(p=1), #####
    #A.RandomSnow(p=1),#####
    #A.Superpixels(p=1),####
    #A.ToSepia(p=1),####
    #A.RandomRain(p=1),#####
    #A.ZoomBlur(p=1),####
    #A.Solarize(p=1),####
    #A.HueSaturationValue(p=1), ####
    #A.CLAHE(p=1), 

    #A.FancyPCA(p=1),
    #A.GaussNoise(p=1),
    #A.GaussianBlur(p=1),
    #A.GlassBlur(p=1),#worse version of blur
    #A.ISONoise(p=1),
    #A.ImageCompression(p=1),
    #A.MedianBlur(p=1),
    #A.MotionBlur(p=1),
    #A.Blur(p=1), #like
    #A.RandomFog(p=1), #like but similar to blur
    #A.MultiplicativeNoise(p=1),
    #A.AdvancedBlur(p=1),#try once more 
    #A.Emboss(p=1), # test 1 more time
    #A.RandomGamma(p=1),
    #A.RGBShift(p=1),  ####

    #A.RandomShadow(p=1,shadow_dimension=10), # try out changes, standard one is bad
    #A.RandomSunFlare(p=1,src_radius=40),# try out changes, standard one is bad
    #A.RandomToneCurve(p=1),
    #A.RingingOvershoot(p=1),
    #A.UnsharpMask(p=1),
    #A.Posterize(p=1), #
    #A.Sharpen(p=1), # like
    #A.Spatter(p=1),#like
    #A.Downscale(p=1),# like
    #A.Defocus(p=1),# similar to blur
    #A.RandomBrightnessContrast(p=1), #like!
    #A.Normalize(p=1)
]
#A.PiecewiseAffine(p=1) looks weird




#-----------use these ones------------
'''
trms = [
    A.RandomShadow(p=0.2,shadow_roi=(0,0,1,1),num_shadows_upper=1,shadow_dimension=4),
    A.Defocus(p=0.2,radius=(2,5)),
    A.Downscale(p=0.2),
    A.Spatter(p=0.2,std=0.15),
    A.RandomSunFlare(p=0.2, src_radius=125, num_flare_circles_upper=20),
    A.Sharpen(p=0.2),
    A.RandomBrightnessContrast(p=0.5,brightness_limit=0.25, contrast_limit=0.25)
]
'''

#test specific augmentation
trms = []
for i in range(9):
    ap =   A.Spatter(p=1, gauss_sigma=0.9*i+0.1)
    trms.append(ap)

save_folder = '/home/isac/Domain_transfer_network/data/test_results'
save_path1= os.path.join(save_folder,"res1.png")
save_path2= os.path.join(save_folder,"res2.png")
st1 = "/home/isac/data/viton_hd/photo/0.jpg"
st2 = "/home/isac/data/f550k/photo/0.jpg"

image = io.imread(st1)
save_path = save_path1


photos =[]
for i,x in enumerate(trms):
    transform = A.Compose([x, A.ToFloat()]) 
    #transform = A.Compose([x, A.Normalize(), A.ToFloat()])    

    augmented = transform(image=image)
    a_img = augmented["image"]
    photos.append(a_img)
print(len(photos))
print(np.max(photos[-1]))

photos_rows = []
for i in range(3):
    a = i*3
    b = (i+1)*3
    ph_row = photos[a:b]
    ph_row = np.concatenate(ph_row,axis=1)
    photos_rows.append(ph_row)

photo = np.concatenate(photos_rows,axis=0)
plt.imsave(save_path, photo)

