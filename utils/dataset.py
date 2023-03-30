import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import albumentations as A
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
from skimage.transform import resize


def clip(imgs,num=255):
    for x in imgs: 
        img = imgs[x]
        img[img<0]=0
        img[img>num]=num
        imgs[x]=img
    return imgs

class domain_transfer_dataset(Dataset):
    def __init__(self, root_dir, transform=None, subset=-1, background_mode="train", specific_background_path="no"):
        csv_path = os.path.join(root_dir,'dataset.csv') 
        self.has_segmentation = os.path.isfile(csv_path)
        if not self.has_segmentation: csv_path = os.path.join(root_dir,'photo.csv') 
        
        self.csv = pd.read_csv(csv_path)
        if subset > 0: self.csv = self.csv.iloc[0:subset,:] 
        self.root_dir = root_dir
        self.transform = transform
        if background_mode == "train": 
            self.background_dir = os.path.join( os.path.dirname(root_dir), "background_train")
        elif background_mode == "255": 
            self.background_dir = os.path.join( os.path.dirname(root_dir), "background_255")
        else:   
            self.background_dir = os.path.join( os.path.dirname(root_dir), "background_evaluation")
        self.number_of_backgrounds = len( os.listdir(self.background_dir) )
        self.specific_background_path = specific_background_path

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx): 
            idx = idx.tolist() 
        
        photo_name        = os.path.join(self.root_dir, 'photo',        self.csv.iloc[idx, 0])
        if self.has_segmentation:
            segmentation_name = os.path.join(self.root_dir, 'segmentation', self.csv.iloc[idx, 1])
        else: segmentation_name = photo_name
        bg_index          = (idx + np.random.randint(0, 200) ) % self.number_of_backgrounds #before it was just = rand(0,len(bg)) which gave really bad results for 255 images
        background_name   = os.path.join(self.background_dir, str(bg_index)+'.jpg')
        if self.specific_background_path != "no": background_name = self.specific_background_path

        photo_img  = io.imread(photo_name)[:,:,0:3]
        target_img = io.imread(photo_name)
        background_img = io.imread(background_name) 
        segmentation_img = io.imread(segmentation_name)[:,:,0:3]
        
        sample = {'photo': photo_img, 'target': target_img, 'segmentation': segmentation_img, 'background': background_img}

        if self.transform:
            sample = self.transform(sample)
        return sample

class RandomCrop(object):

    def __init__(self, output_size=(512,384)):
        self.output_size = output_size

    def __call__(self, sample):
        photo_img = sample['photo']
        h, w = photo_img.shape[:2]
        new_h, new_w = self.output_size

        if new_h >= h: 
            top = 0
            new_h = h
        else: top = np.random.randint(0, h - new_h)

        if new_w >= w:
            left = 0
            new_w = w
        else: left = np.random.randint(0, w - new_w)
        
        sample['photo']           = photo_img[                top:  top + new_h, left: left + new_w]
        if len(sample) < 2: return sample
        sample['segmentation']    = sample['segmentation'][   top:  top + new_h, left: left + new_w]
        sample['target']          = sample['target'][         top:  top + new_h, left: left + new_w]
        sample['background']      = sample['background'][     top:  top + new_h, left: left + new_w]
        return sample

class ApplyMask(object): 
    def __init__(self, dilate_sz=2, get_color_segmentation = False):
        if dilate_sz == 'rnd': dilate_sz = 100 
        self.dilate_sz = dilate_sz
        self.kernel =  np.ones((2*dilate_sz+1, 2*dilate_sz+1))
        self.padding = (dilate_sz, dilate_sz)
        self.get_color_segmentation = get_color_segmentation

    def __call__(self, sample):
        if self.dilate_sz < 0: return sample 
        photo_img = sample['photo']
        segmentation_img = sample['segmentation']
        background_img = sample['background']

        if self.dilate_sz == 100:
            self.dilate_sz = np.random.randint(0,10)
            self.kernel =  np.ones((2*self.dilate_sz+1, 2*self.dilate_sz+1))
            self.padding = (self.dilate_sz, self.dilate_sz)
        
        are_images_tensors = torch.is_tensor(segmentation_img)
        channel_axis = 0 if are_images_tensors else 2
        if are_images_tensors: 
            segmentation_img = np.array(segmentation_img[0:3,:,:]) 
        else: 
            segmentation_img = np.array(segmentation_img[:,:,0:3])
        mask_img = np.max(segmentation_img, channel_axis)
        mask_img[mask_img > 1e-3] = 1
        mask_img[mask_img < 1] = 0
        
        if self.get_color_segmentation:
            divide = 255 if np.amax(segmentation_img) > 1 else 1
            segmentation_img = torch.from_numpy(segmentation_img / divide).float()
        else: 
            segmentation_img = torch.from_numpy(np.stack((mask_img,mask_img,mask_img), axis=channel_axis)).float()

        im_tensor = torch.Tensor(np.expand_dims(np.expand_dims(mask_img, 0), 0))
        kernel_tensor = torch.Tensor(np.expand_dims(np.expand_dims(self.kernel, 0), 0))
        mask_img = torch.clamp(torch.nn.functional.conv2d(im_tensor, kernel_tensor, padding=self.padding), 0, 1)[0,0,:,:]
        mask_img = np.stack((mask_img,mask_img,mask_img), axis=channel_axis)
        photo_img      = photo_img      *  mask_img
        background_img = background_img * (1-mask_img) 
        photo_img = photo_img + background_img

        sample['segmentation'] = segmentation_img
        sample['photo'] = photo_img
        return  sample

class ErodeSegmentation(object): 
    def __init__(self, dilate_sz=2, get_color_segmentation=False):
        self.is_erode = True if dilate_sz >= 0 else False
        if dilate_sz == 'rnd': 
            dilate_sz = 100
        elif dilate_sz < 0: 
            dilate_sz = -dilate_sz
        
        self.dilate_sz = dilate_sz
        self.kernel =  np.ones((2*dilate_sz+1, 2*dilate_sz+1))
        self.padding = (dilate_sz, dilate_sz)
        self.get_color = get_color_segmentation
        
    def __call__(self, sample):
        if self.get_color: return sample
        if self.dilate_sz == 100:
            self.dilate_sz = np.random.randint(-10,11)
            self.is_erode = True if self.dilate_sz >= 0 else False 
            self.dilate_sz = self.dilate_sz if self.is_erode else -self.dilate_sz
            self.kernel =  np.ones((2*self.dilate_sz+1, 2*self.dilate_sz+1))
            self.padding = (self.dilate_sz, self.dilate_sz)
            
        segmentation_img = sample['segmentation']
        segmentation_img = np.array(segmentation_img) 
        ch_ax = 2
        mask_img = np.max(segmentation_img, ch_ax)
        mask_img[mask_img > 1e-3] = 1
        mask_img[mask_img < 1] = 0

        im_tensor = torch.Tensor(np.expand_dims(np.expand_dims(mask_img, 0), 0))
        kernel_tensor = torch.Tensor(np.expand_dims(np.expand_dims(self.kernel, 0), 0))
        if self.is_erode:
            mask_img = 1 - torch.clamp(torch.nn.functional.conv2d(1- im_tensor, kernel_tensor, padding=self.padding), 0, 1)[0,0,:,:]
        else:
            mask_img = torch.clamp(torch.nn.functional.conv2d(im_tensor, kernel_tensor, padding=self.padding), 0, 1)[0,0,:,:]

        mask_img = np.stack((mask_img,mask_img,mask_img), axis=ch_ax)
        #mask_img  = torch.from_numpy(mask_img)   # look at
        sample['segmentation'] = mask_img
        return  sample

class ToTensor(object):
    def __call__(self, sample):
        #(ndarray)w,h,c -->(tensor) c,w,h
        photo_img        = sample['photo'].transpose((2, 0, 1))    
        target_img       = sample['target'].transpose((2, 0, 1))    
        segmentation_img = np.array(sample['segmentation']).transpose((2, 0, 1))    
        background_img   = sample['background'].transpose((2, 0, 1))    

        photo_img        = torch.from_numpy(photo_img).float()
        target_img       = torch.from_numpy(target_img).float()
        segmentation_img = torch.from_numpy(segmentation_img).float()
        background_img   = torch.from_numpy(background_img).float()

        if torch.max(segmentation_img)>1: photo_img/=255
        if torch.max(photo_img)>1: photo_img/=255
        if torch.max(target_img)>1: target_img/=255  
        if torch.max(background_img)>1: background_img/=255  

        return {'photo': photo_img, 'target': target_img, 'segmentation': segmentation_img, 'background': background_img }

class Rescale_bg_down_and_up(object):
    def __call__(self, sample):
        if np.random.randint(0, 10) == 0: 
            bg_img = sample['background']
            photo_img = sample['photo']
            are_images_tensors = torch.is_tensor(bg_img)
            if are_images_tensors:
                h, w = bg_img.shape[1:3]
            else:
                h, w = bg_img.shape[0:2]
            new_h, new_w = h/8, w/8
            new_h, new_w = int(new_h), int(new_w)

            bg_img = transform.resize(bg_img, (new_h, new_w))
            bg_img = transform.resize(bg_img, (h, w))
            if np.max(photo_img)>1:
               bg_img*=255
            sample['background'] = bg_img
        return sample    

#these 3 functions assume ToTensor has been already called, else error
class RotateMult(object):
    def __call__(self, sample):
        angle = np.random.randint(-30, 30)

        photo_img = sample['photo']
        segmentation_img = sample['segmentation']
        target_img = sample['target']
 
        photo_img = TF.rotate(photo_img, angle, interpolation=TF.InterpolationMode.BILINEAR)
        segmentation_img = TF.rotate(segmentation_img, angle)
        target_img = TF.rotate(sample['target'], angle)
        return {'photo': photo_img, 'target': target_img, 'segmentation': segmentation_img, 'background': sample['background']}

class LightingMult(object):
    def __call__(self, sample):
        brightness_factor = np.random.uniform(0.7, 1.3)
        photo_img = TF.adjust_brightness(sample['photo'], brightness_factor)
        sample['photo'] = photo_img
        return sample

class NormalizeMult(object):
    def __call__(self, sample): 
        photo_img = TF.normalize(sample['photo'],(0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        segmentation_img = TF.normalize(sample['segmentation'],(0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        sample['photo'] = photo_img
        sample['segmentation'] = segmentation_img
        return sample

#new classes mostly with albumentations
class A_transforms(object):
    def __call__(self, sample): 
        photo_img        = sample['photo']#uint8 format
        target_img       = sample['target']
        segmentation_img = sample['segmentation']
        background_img   = sample['background']

        #augment all images
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.25),
            ],
            additional_targets={'background': 'image','target':'image','segmentation':'image'}
        )
        augmented = transform(image=photo_img, background=background_img, target=target_img, segmentation=segmentation_img)
        augmented = clip(augmented) 
        photo_img         = augmented["image"]
        background_img    = augmented["background"]
        target_img        = augmented["target"]
        segmentation_img  = augmented["segmentation"]

        p = 0.25
        #augment input image only:
        trms = [ # might want to reduce probabilities 
            A.Spatter(p=p,std=0.15), # bad
            A.RandomShadow(p=p+0.15,shadow_roi=(0,0,1,1),num_shadows_upper=1,shadow_dimension=4),
            A.Defocus(p=p,radius=(1,4)),
            A.Downscale(p=p, scale_min=0.4, scale_max=0.8),
            A.ISONoise(p=0.5), #this augmentation usefull. look up what ISO does!
            A.RandomSunFlare(p=p, src_radius=125, num_flare_circles_upper=20),
            A.Sharpen(p=p),
            A.RandomBrightnessContrast(p=0.5,brightness_limit=0.25, contrast_limit=0.25),            
        ]
        transform = A.Compose(trms)
        augmented = transform(image=photo_img)
        photo_img = augmented["image"]
        
        #fix data format
        transform = A.Compose([
                A.Rotate(limit=30,border_mode=cv2.BORDER_CONSTANT), # look at cv2.border modes and variations!
                A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ToTensorV2(),
            ],
            additional_targets={'background': 'image','target':'image','segmentation':'image'}
        )

        augmented = transform(image=photo_img, background=background_img, target=target_img, segmentation=segmentation_img)
        sample['photo']          = augmented["image"]
        sample['background']     = augmented["background"]
        sample['target']         = (augmented["target"]+1)/2
        sample['segmentation']   = augmented["segmentation"]

        return sample

class ToErodedMask(object): 
    def __init__(self, erode_sz=0):        
        self.is_dilate = erode_sz < 0
        self.dilate_sz = np.abs(erode_sz)
        self.kernel =  np.ones((2*self.dilate_sz + 1, 2*self.dilate_sz +1))
        self.padding = (self.dilate_sz , self.dilate_sz )
        
    def __call__(self, sample):            
        segmentation_img = sample['segmentation']
        segmentation_img = np.array(segmentation_img) 
        ch_ax = 2
        mask_img = np.max(segmentation_img, ch_ax)
        mask_img[mask_img > 1e-3] = 1
        mask_img[mask_img < 1] = 0

        im_tensor = torch.Tensor(np.expand_dims(np.expand_dims(mask_img, 0), 0))
        kernel_tensor = torch.Tensor(np.expand_dims(np.expand_dims(self.kernel, 0), 0))
        if self.is_dilate:
            mask_img = torch.clamp(torch.nn.functional.conv2d(im_tensor, kernel_tensor, padding=self.padding), 0, 1)[0,0,:,:]
        else:
            mask_img = 1-torch.clamp(torch.nn.functional.conv2d(1-im_tensor, kernel_tensor, padding=self.padding), 0, 1)[0,0,:,:]
        mask_img = np.stack((mask_img,mask_img,mask_img), axis=ch_ax)
        sample['segmentation'] = (mask_img*255).astype(np.uint8)

        return  sample

class AddDilatedBackground(object): 
    def __init__(self, dilate_sz=0):        
        self.dilate_sz = dilate_sz
    def __call__(self, sample):
        photo_img = sample['photo']
        mask_img = sample['segmentation']
        background_img = sample['background']
        
        transform = transforms.Compose([
            ToErodedMask(-self.dilate_sz)
        ])
        mask_img = transform({'segmentation':mask_img})['segmentation']
        mask_img[mask_img>1e-3]=1 
            
        photo_img      = photo_img*mask_img + background_img*(1-mask_img) 
        sample['photo'] = photo_img

        return  sample

class A_Norm(object):    
    def __call__(self, sample): 
        transform = A.Compose([
                A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ToTensorV2(),
            ],
            additional_targets={'background': 'image','target':'image','segmentation':'image'}
        )

        augmented = transform(image=sample['photo'], background=sample['background'], target=sample['target'], segmentation=sample['segmentation'])
        sample['photo']          = augmented["image"]
        sample['background']     = augmented["background"]
        sample['target']         = (augmented["target"]+1)/2
        sample['segmentation']   = augmented["segmentation"]
        return sample 
    
class CropAndPad(object):    
    def __init__(self, desired_shape=(512,384)):        
        self.desired_shape = desired_shape
    def __call__(self, sample): 
        image = sample['photo']
        current_shape = image.shape[:2]
        scale_factor = np.min(np.array(self.desired_shape)/current_shape) #make sure we don't scale 10x5 -> 9x1
        resized_image = resize(image, (int(current_shape[0]*scale_factor), int(current_shape[1]*scale_factor)),
                            anti_aliasing=False)

        pad_amount = [(self.desired_shape[i]-resized_image.shape[i])//2 for i in range(2)]
        padded_image = np.pad(resized_image, ((pad_amount[0], pad_amount[0]), (pad_amount[1], pad_amount[1]), (0,0)), mode='constant')
        return padded_image 
        
