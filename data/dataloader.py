import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms.functional as TF
import csv
import re

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
    def __init__(self, path_to_csv_file, root_dir, transform=None):
        self.csv = pd.read_csv(path_to_csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.number_of_backgrounds = len(os.listdir(os.path.join(self.root_dir,'background')))

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx): 
            idx = idx.tolist() 
        photo_name = os.path.join(self.root_dir, 'photo',
                                self.csv.iloc[idx, 0])
        segmentation_name = os.path.join(self.root_dir,'segmentation',  
                                self.csv.iloc[idx, 1])
        bg_index = np.random.randint(0, self.number_of_backgrounds)
        background_name = os.path.join(self.root_dir, 'background', str(bg_index)+'.jpg')

        photo_img  = io.imread(photo_name)
        target_img = io.imread(photo_name)
        background_img = io.imread(background_name) 
        segmentation_img = io.imread(segmentation_name)
        
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
        
        photo_img = transform.resize(photo_img, (new_h, new_w))
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
    def __init__(self, segmentation_to_mask = True, dilate_sz=2):
        self.dilate_sz = dilate_sz
        self.kernel =  np.ones((2*dilate_sz+1, 2*dilate_sz+1))
        self.padding = (dilate_sz, dilate_sz)
        self.segmentation_to_mask = segmentation_to_mask

    def __call__(self, sample):
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
        
        if self.segmentation_to_mask: #if fals --> get segmentation_image with colors
            segmentation_img = torch.from_numpy(np.stack((mask_img,mask_img,mask_img), axis=channel_axis)).float()
        else: 
            divide = 255 if np.amax(segmentation_img) > 1 else 1
            segmentation_img = torch.from_numpy(segmentation_img / divide).float()
        
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

class ToTensor(object):
    def __call__(self, sample):
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

        #w,h,c --> c,w,h
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


#-----------------Create dataset functions-------------------

def save_dataset_images(folder, dataset): # create dataset
    data_len = len(dataset)
    data_1_percent = data_len//100 
    for idx in range(data_len):
        if idx % data1percent == 0: print(f'{idx/data_len:2.f}')
        sample = dataset[idx]
        img = sample['photo']
        file_name = os.path.join(folder, str(idx)+'.jpg')
        plt.imsave(file_name, img)

    print('saved all images in' + folder)

def sort_key(file_name): # create dataset
    #make filename into integer, which is sorted
    file_number_str = re.match('[0-9]+', file_name)
    if file_number_str is not None:
        return int(file_number_str.group())
    else:
        return -1

def write_csv(csv_file_path, image_dir, description='image'): #create dataset
    if os.path.exists(csv_file_path): return 0

    with open(csv_file_path, 'w') as file:
        writer = csv.writer(file)
        writer.writerow([description])
                
        entries = os.listdir(image_dir)
        entries.sort(key=sort_key)

        for entry in entries:
            if entry[-1]!='g': continue
            writer.writerow([entry])

def combine_csv(save_path, file_path1, file_path2): #create dataset
    f1 = pd.read_csv(file_path1)
    f2 = pd.read_csv(file_path2)

    with open(save_path, 'w') as file:
        writer = csv.writer(file)
        writer.writerow([f1.columns[0],f2.columns[0]])

        for idx in range(len(f1)):
            f1_file = f1.iloc[idx, 0]
            f2_file = f2.iloc[idx, 0]
            writer.writerow([f1_file, f2_file])
            

#----------training & functions---------------

def split_data(dataset):
    i = (0.95*len(dataset))//1
    training_set = dataset[0:i]
    test_set = dataset[i:]
    return training_set, test_set

def tensor_to_saveable_img(tensor):
    y = torch.transpose(tensor, 0, 2)
    y = torch.transpose(y, 0, 1)
    y = np.array(y)
    if np.amin(y) < 0 : y = (y+1)/2 
    return y

def save_batch_images(savefolder, sample_batched, show_photos_or_target='photo'): #TODO:
   2 #saves all images into a save_folder
