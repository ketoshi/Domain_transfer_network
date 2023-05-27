import os
import torch
import torchvision
from skimage import io, transform
import matplotlib.pyplot as plt
from utils.dataset import CropAndPad, Outline, A_Norm, ToErodedMask,file_names_dataset, A_transforms
import albumentations as A
from torchvision import transforms
import numpy as np
from utils.utils import get_dataloader, generate_images, write_csv, combine_csv, compare_bg_images, concat_save_2imgs, get_model, tensor_to_saveable_img
import albumentations as A
from PIL import Image
model_path = "/home/isac/data/final/seg_gan/test38.pth"
save_folder = '/home/isac/data/use_model_output'


class OWN(object): # 
    def __call__(self, sample): 
        photo_img        = sample['photo']#uint8 format
        if np.max(photo_img)<2: photo_img = np.array(photo_img*255,dtype=np.uint8)

        trms = [
            A.Defocus(p=1,radius=(1,4)),    
            #A.Downscale(p=1, scale_min=0.75, scale_max=0.8,interpolation=cv2.INTER_NEAREST),
            #A.ISONoise(p=1),
            #A.Sharpen(p=1,alpha=(0.5,0.5)),
            #A.RandomBrightnessContrast(p=1,brightness_limit=0.4, contrast_limit=0.4),
        ]
        transform = A.Compose(trms)
        augmented = transform(image=photo_img)
        photo_img = augmented["image"]
        
        if np.max(photo_img)>2.00: photo_img=np.array(photo_img,dtype=np.float64)/255
        photo_img[photo_img<0]=0
        photo_img[photo_img>1]=1
        sample['photo'] = photo_img
        return sample




#To generate sample from validation set when training go to "use_model.py"

#use files to an img or or folder
photo_str = "/home/isac/data/zalando/photo/img17.jpg"
segmentation_str = "/home/isac/data/zalando/segmentation/img17.png"

#set max images or -1 for generating for all images
MAX_IMAGES = 4





if os.path.isdir(photo_str): 
    files = sorted(os.listdir(photo_str))
    photo_str = [os.path.join(photo_str, file) for file in files]
else: photo_str = [photo_str] 

if os.path.isdir(segmentation_str): 
    files = sorted(os.listdir(segmentation_str))
    segmentation_str = [os.path.join(segmentation_str, file) for file in files]
else: segmentation_str = [segmentation_str] 

if not os.path.isdir(save_folder): os.mkdir(save_folder) 
ix = len(os.listdir(save_folder))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_model(model_path, device)
model.eval()
trms = transforms.Compose( [ToErodedMask(0),CropAndPad(),OWN, A_Norm()] )

for photo_path, segmentation_path in zip(photo_str,segmentation_str):

    save_path = os.path.join(save_folder,"img"+str(ix)+".png")
    ix+=1
    input_image = io.imread(photo_path)
    segmentation_image = io.imread(segmentation_path)[:,:,:3]
    sample = {'photo':input_image, 'segmentation':segmentation_image}

    sample = trms(sample)
    input_image = sample['photo'].to(device)
    segmentation_image = sample['segmentation'].to(device)

    input_image_aug = torch.unsqueeze( input_image, 0)
    segmentation_image_aug = torch.unsqueeze( segmentation_image, 0)
    generated_images = model(input_image_aug, segmentation_image_aug)
    
    input_image_aug = (input_image_aug+1)/2 
    segmentation_image_aug = (segmentation_image_aug+1)/2 
    
    bg = torchvision.io.read_image("/home/isac/data/bg_test.jpg")
    bg = torch.unsqueeze(bg,0).to(device) 
    #generated_images = segmentation_image_aug*input_image_aug + (1-segmentation_image_aug)*bg #to copy paste background


    k=20
    #segmentation_image_aug, input_image_aug, generated_images = segmentation_image_aug[:,:,:,k:-k], input_image_aug[:,:,:,k:-k], generated_images[:,:,:,k:-k]
    grid0 = torchvision.utils.make_grid(segmentation_image_aug)
    grid1 = torchvision.utils.make_grid(input_image_aug)
    grid2 = torchvision.utils.make_grid(generated_images)
    grid_tot = torch.concat((grid0,grid1,grid2),dim=2)

    img_save = tensor_to_saveable_img(grid_tot)
    plt.imsave(save_path,img_save)
    if ix > MAX_IMAGES and MAX_IMAGES > 0: break
print('complete')
