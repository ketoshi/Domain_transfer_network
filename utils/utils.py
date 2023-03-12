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
sys.path.append('utils') # important but unsure why
from dataset import domain_transfer_dataset, RandomCrop, ToTensor, ApplyMask, LightingMult, RotateMult, NormalizeMult, ErodeSegmentation
from layers import *
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import json
import warnings
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import torchmetrics
from torchvision import transforms




#-----------------Create dataset functions-------------------

def save_dataset_images(folder, dataset): # create dataset
    data_len = len(dataset)
    data_1_percent = data_len//100 
    for idx in range(data_len):
        if idx % data_1_percent == 0: print(f'{idx/data_len:.2f}')
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
    #if type(batch_images) == type((1,2)):
        #normalize correct 
    
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

def get_model(model_path, device):
    model_info_path = os.path.join( os.path.dirname(model_path), "model_info.json")
    with open(model_info_path) as json_file:
        model_info = json.load(json_file)

    layer_channels = model_info["layer_channels"]
    skip_layers = model_info["skip_layers"]
    model = DTCNN(layer_channels=layer_channels, skip_layers=skip_layers).to(device) 
    model.load_state_dict( torch.load( model_path, map_location=torch.device(device) ) )
    return model

def get_dataloader(root_dir, usage="train_model", use_validation_set=False, validation_set_length=500, BATCH_SIZE=3, kernel_size_dilation=0, kernel_size_erosion=0, get_color_segmentation=False):
    
    if usage == 'use_model_use_mask':
        trms = torchvision.transforms.Compose([RandomCrop((512,384)), ApplyMask(get_color_segmentation, kernel_size_dilation),ErodeSegmentation(kernel_size_erosion), ToTensor(), NormalizeMult() ]) 
    if usage == 'use_model_no_mask':    
        trms = torchvision.transforms.Compose([RandomCrop((512,384)), ErodeSegmentation(kernel_size_erosion), ToTensor(), NormalizeMult() ]) 
    if usage == 'train_model':
        trms = torchvision.transforms.Compose([RandomCrop((512,384)), ApplyMask(get_color_segmentation, kernel_size_dilation), ErodeSegmentation(kernel_size_erosion), ToTensor(), LightingMult(), RotateMult(),  NormalizeMult() ])  

    dataset = domain_transfer_dataset(root_dir, transform=trms)
    train_set, val_set = torch.utils.data.random_split(dataset, [len(dataset)-validation_set_length,validation_set_length], generator=torch.Generator().manual_seed(0))
    use_set = train_set
    if use_validation_set: use_set = val_set
    dataset_loader = torch.utils.data.DataLoader(use_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    return dataset_loader

def train_n_epochs(root_dir, layer_channels, skip_layers, lr, batch_size, epochs, save_folder, update_dataset_per_epoch = True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DTCNN(layer_channels=layer_channels, skip_layers=skip_layers).to(device)

    dataset_loader = get_dataloader(root_dir, BATCH_SIZE=batch_size)
    ld = len(dataset_loader)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    writer = SummaryWriter(save_folder)
    for epoch in range(epochs):   
        if update_dataset_per_epoch and epoch > 0:   dataset_loader = get_dataloader(root_dir, BATCH_SIZE=batch_size)

        for ix, x in enumerate(tqdm(dataset_loader)):
            i_scalar = epoch*ld + ix
            optimizer.zero_grad()
            
            input_image, segmentation_image, target_image = get_loader_vals(x, device)
            generated_image = model(input_image, segmentation_image)
            loss = get_loss(generated_image, target_image)

            writer.add_scalar('Training Loss', loss, global_step=i_scalar)
            loss.backward()
            optimizer.step()

        gen_img = generated_image.clone()   
        grid = torchvision.utils.make_grid(gen_img)
        writer.add_image('images', grid, global_step=epoch)
        if (epoch) % 2 == 0:
            torch.save(model.state_dict(), save_folder + '/test'+str(epoch)+'.pth')
        
        if epoch == 0:
            info = {'lr':lr, 'batch_size':batch_size, 'epochs':epochs, 'update_dataset_per_epoch':update_dataset_per_epoch,  'layer_channels':layer_channels, 'skip_layers':skip_layers}
            info_file = os.path.join(save_folder,'model_info.json')
            with open(info_file, 'w') as outfile:
                json.dump(info, outfile)
    writer.close()

def train_n_epochs_double(root_dir, layer_channels, skip_layers, lr, batch_size, epochs, save_folder, update_dataset_per_epoch = True):#fail
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model1 = DTCNN(layer_channels=layer_channels, skip_layers=skip_layers).to(device)
    model2 = DTCNN(layer_channels=layer_channels, skip_layers=skip_layers).to(device)

    dataset_loader = get_dataloader(root_dir, BATCH_SIZE=batch_size)
    ld = len(dataset_loader)

    optimizer1 = optim.Adam(model1.parameters(), lr=lr)
    optimizer2 = optim.Adam(model2.parameters(), lr=lr)
    writer = SummaryWriter(save_folder)
    for epoch in range(epochs):   
        if update_dataset_per_epoch and epoch > 0: dataset_loader = get_dataloader(root_dir, BATCH_SIZE=batch_size)

        for ix, x in enumerate(tqdm(dataset_loader)):
            i_scalar = epoch*ld + ix
            
            input_image, segmentation_image, target_image = get_loader_vals(x, device)
            generated_image1 = model1(input_image, segmentation_image)
            generated_image2 = model2(input_image, segmentation_image)
            
            optimizer1.zero_grad()
            loss1 = get_loss(generated_image1, target_image).detach()*0.9
            loss1 += 0.1*get_loss(generated_image1, generated_image2)
            loss1.backward(retain_graph=True)
            optimizer1.step()
            
            optimizer2.zero_grad()
            loss2 = get_loss(generated_image2, target_image).detach()*0.9
            loss2 += 0.1*get_loss(generated_image2, generated_image1)
            loss2.backward()
            optimizer2.step()
    
            writer.add_scalar('Training Loss', loss1, global_step=i_scalar)

        gen_img = generated_image1.clone()   #Change if we normalize output! to (img+1)/2
        grid = torchvision.utils.make_grid(gen_img)
        writer.add_image('images', grid, global_step=epoch)
        if (epoch) % 4 == 0:
            torch.save(model1.state_dict(), save_folder + '/test1'+str(epoch)+'.pth')
            torch.save(model2.state_dict(), save_folder + '/test2'+str(epoch)+'.pth')
        
        if epoch == 0:
            info = {'lr':lr, 'batch_size':batch_size, 'epochs':epochs, 'update_dataset_per_epoch':update_dataset_per_epoch,  'layer_channels':layer_channels, 'skip_layers':skip_layers}
            info_file = os.path.join(save_folder,'model_info.json')
            with open(info_file, 'w') as outfile:
                json.dump(info, outfile)
    writer.close()

def train_n_epochs_twice(root_dir, layer_channels, skip_layers, lr, batch_size, epochs, save_folder, update_dataset_per_epoch = True):#seems succesfull?
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DTCNN(layer_channels=layer_channels, skip_layers=skip_layers).to(device)

    dataset_loader = get_dataloader(root_dir, BATCH_SIZE=batch_size)
    ld = len(dataset_loader)


    optimizer = optim.Adam(model.parameters(), lr=lr)
    writer = SummaryWriter(save_folder)
    for epoch in range(epochs):   
        if update_dataset_per_epoch and epoch > 0:   dataset_loader = get_dataloader(root_dir, BATCH_SIZE=batch_size)

        for ix, x in enumerate(tqdm(dataset_loader)):
            i_scalar = epoch*ld + ix
            optimizer.zero_grad()
            
            input_image, segmentation_image, target_image = get_loader_vals(x, device)
            generated_image = model(input_image, segmentation_image)
            generated_image2 = model(generated_image, segmentation_image)
            loss = get_loss(generated_image, target_image)
            loss += 0.1*get_loss(generated_image2, target_image)

            writer.add_scalar('Training Loss', loss, global_step=i_scalar)
            loss.backward()
            optimizer.step()

        gen_img = generated_image.clone()   #Change if we normalize output! to (img+1)/2
        grid = torchvision.utils.make_grid(gen_img)
        writer.add_image('images', grid, global_step=epoch)
        if (epoch) % 2 == 0:
            torch.save(model.state_dict(), save_folder + '/test'+str(epoch)+'.pth')
        
        if epoch == 0:
            info = {'lr':lr, 'batch_size':batch_size, 'epochs':epochs, 'update_dataset_per_epoch':update_dataset_per_epoch,  'layer_channels':layer_channels, 'skip_layers':skip_layers}
            info_file = os.path.join(save_folder,'model_info.json')
            with open(info_file, 'w') as outfile:
                json.dump(info, outfile)
    writer.close()

def generate_images(dataloader, model_path, save_folder, max_images=-1):

    if not os.path.isdir(save_folder): os.mkdir(save_folder) 
    BATCH_SIZE = 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(model_path, device)
    model.eval()
    
    for ix, x in enumerate(tqdm(dataloader)):
        input_image, segmentation_image, target_image = get_loader_vals(x, device)
        generated_images = model(input_image, segmentation_image)
        input_image = (input_image+1)/2 
        segmentation_image = (segmentation_image+1)/2 

        grid0 = torchvision.utils.make_grid(segmentation_image)
        grid1 = torchvision.utils.make_grid(input_image)
        grid2 = torchvision.utils.make_grid(target_image)
        grid3 = torchvision.utils.make_grid(generated_images)

        grid_tot = torch.concat((grid0,grid1,grid2,grid3),dim=1)
        save_images(save_folder, grid_tot, as_batch=True)
        if ix > max_images-2 and max_images > 0: break

    print("program complete, images saved in test_results")

def get_metrics(dataloader1, dataloader2, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(model_path, device)
    model.eval()

    ssim_model = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0,).to(device)
    psnr_model = torchmetrics.PeakSignalNoiseRatio().to(device)
    fid = FrechetInceptionDistance(feature=64, normalize=True).to(device)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lpips_model = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)

    ssim_tot = 0 
    lpips_tot = 0
    psnr_tot = 0
    for ix, x in enumerate(dataloader1):
        photo         = x['photo'].to(device)
        segmentation  = x['segmentation'].to(device)
        generated_img = model(photo, segmentation)
        photo         = (photo+1)/2
        ssim_tot  += ssim_model(photo, generated_img).item()
        lpips_tot += lpips_model(photo, generated_img).item()
        psnr_tot  += psnr_model(photo, generated_img).item()
        fid.update(photo, real=True)
        fid.update(generated_img, real=False)

    print(f'fid: {fid.compute()}')
    print(f'ssim_avg = {ssim_tot/len(dataloader1)}')
    print(f'lpips_avg = {lpips_tot/len(dataloader1)}')
    print(f'psnr_avg = {psnr_tot/len(dataloader1)}')
