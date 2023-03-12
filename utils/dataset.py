import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class create_dataset(Dataset):
    def __init__(self, path_to_csv_file, image_dir, transform=None):
        self.csv = pd.read_csv(path_to_csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx): 
            idx = idx.tolist() 
        photo_name = os.path.join(self.image_dir,
                                self.csv.iloc[idx, 0])
        photo_img = io.imread(photo_name)
        sample = {'photo': photo_img}

        if self.transform:
            sample = self.transform(sample)
        return sample

class domain_transfer_dataset(Dataset):
    def __init__(self, root_dir, transform=None, subset=-1):
        csv_path = os.path.join(root_dir,'dataset.csv') 
        self.has_segmentation = os.path.isfile(csv_path)
        if not self.has_segmentation: csv_path = os.path.join(root_dir,'photo.csv') 
        
        self.csv = pd.read_csv(csv_path)
        if subset > 0: self.csv = self.csv.iloc[0:subset,:] 
        self.root_dir = root_dir
        self.transform = transform
        self.background_dir = os.path.join( os.path.dirname(root_dir), "background")
        self.number_of_backgrounds = len( os.listdir(self.background_dir) )

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx): 
            idx = idx.tolist() 
        photo_name        = os.path.join(self.root_dir, 'photo',        self.csv.iloc[idx, 0])
        if self.has_segmentation:
            segmentation_name = os.path.join(self.root_dir, 'segmentation', self.csv.iloc[idx, 1])
        else: segmentation_name = photo_name
        bg_index          = np.random.randint(0, self.number_of_backgrounds)
        background_name   = os.path.join(self.background_dir, str(bg_index)+'.jpg')

        photo_img  = io.imread(photo_name)
        target_img = io.imread(photo_name)
        background_img = io.imread(background_name) 
        segmentation_img = io.imread(segmentation_name)[:,:,0:3]
        
        sample = {'photo': photo_img, 'target': target_img, 'segmentation': segmentation_img, 'background': background_img}

        if self.transform:
            sample = self.transform(sample)
        return sample

class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        photo_img = sample['photo']
        h, w = photo_img.shape[:2]
        if isinstance(self.output_size, int): 
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
            new_h, new_w = int(new_h), int(new_w)
        
        photo_img = transform.resize(photo_img, (new_h, new_w),anti_aliasing=True, order=2)# check resize mode, anti-aliasing is used
        if len(sample) < 2: return {'photo': photo_img}
        
        segmentation_img = transform.resize(sample['segmentation'], (new_h, new_w))
        target_img = transform.resize(sample['target'], (new_h, new_w))
        background_img = transform.resize(sample['background'], (new_h, new_w))
        return  {'photo': photo_img, 'target': target_img, 'segmentation': segmentation_img, 'background': background_img }

class RandomCrop(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size) 
        else:
            assert len(output_size) == 2
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
        
        photo_img           = photo_img[                top:  top + new_h, left: left + new_w]
        if len(sample) < 2: return {'photo': photo_img}
        segmentation_img    = sample['segmentation'][   top:  top + new_h, left: left + new_w]
        target_img          = sample['target'][         top:  top + new_h, left: left + new_w]
        background_img      = sample['background'][     top:  top + new_h, left: left + new_w]
        return  {'photo': photo_img, 'target': target_img, 'segmentation': segmentation_img, 'background': background_img }

class ApplyMask(object): 
    def __init__(self, get_color_segmentation = False, dilate_sz=2, apply_mask=True):
        self.dilate_sz = dilate_sz
        self.kernel =  np.ones((2*dilate_sz+1, 2*dilate_sz+1))
        self.padding = (dilate_sz, dilate_sz)
        self.get_color_segmentation = get_color_segmentation
        self.apply_mask = apply_mask

    def __call__(self, sample):
        if self.dilate_sz < 1: return sample
        photo_img = sample['photo']
        segmentation_img = sample['segmentation']
        background_img = sample['background']
        
        are_images_tensors = torch.is_tensor(segmentation_img)
        channel_axis = 0 if are_images_tensors else 2
        if are_images_tensors: 
            segmentation_img = np.array(segmentation_img[0:3,:,:]) 
        else: 
            segmentation_img = np.array(segmentation_img[:,:,0:3])
        
        mask_img = np.max(segmentation_img, channel_axis)
        mask_img[mask_img > 1e-3] = 1
        mask_img[mask_img < 1] = 0
        
        if not self.get_color_segmentation:
            segmentation_img = torch.from_numpy(np.stack((mask_img,mask_img,mask_img), axis=channel_axis)).float()
        else: 
            divide = 255 if np.amax(segmentation_img) > 1 else 1
            segmentation_img = torch.from_numpy(segmentation_img / divide).float()
        if not self.apply_mask:
            sample['segmentation'] = segmentation_img
            return  sample

        if self.dilate_sz > 0: 
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
    def __init__(self, dilate_sz=2):
        self.dilate_sz = dilate_sz
        self.kernel =  np.ones((2*dilate_sz+1, 2*dilate_sz+1))
        self.padding = (dilate_sz, dilate_sz)

    def __call__(self, sample):
        if self.dilate_sz <1: return sample
        
        segmentation_img = sample['segmentation']
        are_images_tensors = torch.is_tensor(segmentation_img)
        channel_axis = 0 if are_images_tensors else 2
        if are_images_tensors: 
            segmentation_img = np.array(segmentation_img[0:3,:,:]) 
        else: 
            segmentation_img = np.array(segmentation_img[:,:,0:3])
        
        mask_img = np.max(segmentation_img, channel_axis)
        mask_img[mask_img > 1e-3] = 1
        mask_img[mask_img < 1] = 0
        
        im_tensor = torch.Tensor(np.expand_dims(np.expand_dims(mask_img, 0), 0))
        kernel_tensor = torch.Tensor(np.expand_dims(np.expand_dims(self.kernel, 0), 0))
        mask_img = 1 - torch.clamp(torch.nn.functional.conv2d(1- im_tensor, kernel_tensor, padding=self.padding), 0, 1)[0,0,:,:]
        mask_img = np.stack((mask_img,mask_img,mask_img), axis=channel_axis)
        mask_img  = torch.from_numpy(mask_img)

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


#these 3 functions assume ToTensor has been already called, else error
class RotateMult(object):
    def __call__(self, sample):
        angle = np.random.randint(-30, 30)

        photo_img = sample['photo']
        segmentation_img = sample['segmentation']
        target_img = sample['target']
 
        photo_img = TF.rotate(photo_img, angle)
        segmentation_img = TF.rotate(segmentation_img, angle)
        target_img = TF.rotate(sample['target'], angle)
        return {'photo': photo_img, 'target': target_img, 'segmentation': segmentation_img, 'background': sample['background']}

class LightingMult(object):
    def __call__(self, sample):
        brightness_factor = np.random.uniform(0.7, 1.3)
        photo_img = TF.adjust_brightness(sample['photo'], brightness_factor)
        sample['photo'] = photo_img
        return sample

class NormalizeMult(object): #I don't normalize the output image as well right?, and think having mean & std as 0.5 is ok although better to generate own values. 
    def __call__(self, sample): 
        photo_img = TF.normalize(sample['photo'],(0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        segmentation_img = TF.normalize(sample['segmentation'],(0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        sample['photo'] = photo_img
        sample['segmentation'] = segmentation_img
        return sample

