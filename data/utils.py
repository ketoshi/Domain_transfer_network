import sys,os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import csv
import re
import torchvision
from torch import tensor
from tqdm import tqdm
sys.path.append('data')
sys.path.append('model')
from dataset import domain_transfer_dataset, RandomCrop, ToTensor, ApplyMask, LightingMult, RotateMult, NormalizeMult
from layers import *
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import json



#-----------------Create dataset functions-------------------

def save_dataset_images(folder, dataset): # create dataset
    data_len = len(dataset)
    data_1_percent = data_len//100 
    for idx in range(data_len):
        if idx % data_1_percent == 0: print(f'{idx/data_len:2.f}')
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

def write_csv(csv_file_path, image_dir, description='image', overwrite=False): #create dataset
    #creade csv file from folder of images
    if os.path.exists(csv_file_path): 
        if not overwrite: return 0

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
            





#----------training & using network---------------

def tensor_to_saveable_img(tensor):
    if torch.cuda.is_available():
        tensor = tensor.detach().cpu()
    y = torch.transpose(tensor, 0, 2)
    y = torch.transpose(y, 0, 1)
    y = np.array(y)
    if np.amin(y) < 0 : y = (y+1)/2 
    y[y<0] = 0
    y[y>1] = 1
    return y

def save_images(savefolder, batch_images, as_batch=False):
    existing_images = len(os.listdir(savefolder))
    if as_batch :
        grid = torchvision.utils.make_grid(batch_images)
        save_path = os.path.join(savefolder, "img" + str(existing_images)+".jpg")
        plt.imsave(save_path, tensor_to_saveable_img(grid))
    else :
        for i in range( batch_images.shape[0] ) :
            save_path = os.path.join(savefolder, "img" + str(existing_images+i)+".jpg")
            plt.imsave(save_path, tensor_to_saveable_img(batch_images[i]))
   
def get_loss(generated_image, target_image):
    loss = F.mse_loss(generated_image, target_image) 
    tot_loss = loss
    return tot_loss    #add more loss potentially to guide training

def get_loader_vals(values, device):
    return values['photo'].to(device), values['segmentation'].to(device), values['target'].to(device)

def train_n_epochs(layer_channels, skip_layers, lr, batch_size, epochs, save_folder, update_dataset_per_epoch = True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model = DTCNN(layer_channels=layer_channels, skip_layers=skip_layers).to(device)
    model = test_model3().to(device)

    root_dir = '/home/isac/data/viton_hd'
    csv_file = os.path.join(root_dir,'dataset.csv')
    trms = torchvision.transforms.Compose([RandomCrop((512,384)), ApplyMask(), \
      ToTensor(), LightingMult(), RotateMult(),  NormalizeMult() ]) 
   
    dataset = domain_transfer_dataset(csv_file, root_dir, transform=trms)
    train_set, val_set = torch.utils.data.random_split(dataset, [len(dataset)-500,500], generator=torch.Generator().manual_seed(0)) 
    dataset_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    ld = len(dataset_loader)


    optimizer = optim.Adam(model.parameters(), lr=lr)
    writer = SummaryWriter(save_folder)
    for epoch in range(epochs):   
        if update_dataset_per_epoch and epoch > 0:     
            dataset = domain_transfer_dataset(csv_file, root_dir, transform=trms)
            train_set, val_set = torch.utils.data.random_split(dataset, [len(dataset)-500,500], generator=torch.Generator().manual_seed(0)) 
            dataset_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)

        for ix, x in enumerate(tqdm(dataset_loader)):
            i_scalar = epoch*ld + ix
            optimizer.zero_grad()
            
            input_image, segmentation_image, target_image = get_loader_vals(x, device)
            generated_image = model(input_image, segmentation_image)
            loss = get_loss(generated_image, target_image)

            writer.add_scalar('Training Loss', loss, global_step=i_scalar)
            loss.backward()
            optimizer.step()

        gen_img = generated_image.clone()   #Change if we normalize output! to (img+1)/2
        grid = torchvision.utils.make_grid(gen_img)
        writer.add_image('images', grid, global_step=epoch)
        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), save_folder + '/test'+str(epoch)+'.pth')
        
        if epoch == 0:
            info = {'lr':lr, 'batch_size':batch_size, 'epochs':epochs, 'update_dataset_per_epoch':update_dataset_per_epoch,  'layer_channels':layer_channels, 'skip_layers':skip_layers}
            info_file = os.path.join(save_folder,'model_info.json')
            with open(info_file, 'w') as outfile:
                json.dump(info, outfile)
    writer.close()
