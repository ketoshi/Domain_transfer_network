import sys
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import csv
import io
import re
import torchvision
from torch import tensor
from tqdm import tqdm
sys.path.append('utils') # important
try:
    from dataset import A_transforms, A_Norm, AddDilatedBackground, ToErodedMask, domain_transfer_dataset, RandomCrop
    from layers import DTCNN, SDTCNN, GAN_discriminator, VGGPerceptualLoss,VGG_SDTCNN,VGG_discriminator
except:
    from utils.dataset import A_transforms, A_Norm, AddDilatedBackground, ToErodedMask, domain_transfer_dataset, RandomCrop
    from utils.layers import DTCNN, SDTCNN, GAN_discriminator, VGGPerceptualLoss,VGG_SDTCNN,VGG_discriminator
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import json
import warnings
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import torchmetrics
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader 
from torch import device
from skimage.transform import resize
from torchvision.models import vgg19, VGG19_Weights

#-----------------Create dataset functions-------------------
def create_folder(root_dir):
    if not os.path.isdir(root_dir):         os.mkdir(root_dir)
    photo_dir        = os.path.join(root_dir,"photo")
    segmentation_dir = os.path.join(root_dir,"segmentation")
    if not os.path.isdir(photo_dir):        os.mkdir(photo_dir)
    if not os.path.isdir(segmentation_dir): os.mkdir(segmentation_dir)

def resize_and_pad_image(image, desired_shape=(512, 384)):
    current_shape = image.shape[:2]
    scale_factor = np.min(np.array(desired_shape)/current_shape) #make sure we don't scale 10x5 -> 9x1
    resized_image = resize(image, (int(current_shape[0]*scale_factor), int(current_shape[1]*scale_factor)),
                           anti_aliasing=False)

    pad_amount = [(desired_shape[i]-resized_image.shape[i])//2 for i in range(2)]
    padded_image = np.pad(resized_image, ((pad_amount[0], pad_amount[0]), (pad_amount[1], pad_amount[1]), (0,0)), mode='constant')
    return padded_image 

def rescale_and_save(save_folder, image_folder, resize_shape=(512, 384)):
    if not os.path.isdir(save_folder): os.mkdir(save_folder)
    if len(os.listdir(save_folder)) < 10:
        image_paths = os.listdir(image_folder)
        i = 0
        for image_name in tqdm(image_paths):
            image_path = os.path.join(image_folder, image_name)
            image = io.imread(image_path)
            image = resize_and_pad_image(image, resize_shape)
            file_name = os.path.join(save_folder,"img"+str(i)+".png")
            plt.imsave(file_name, image)
            i+=1

def write_csv(csv_file_path:str, 
              image_dir:str, 
              description='image', 
              overwrite=False):
    if os.path.exists(csv_file_path): 
        if not overwrite: return 0

    with open(csv_file_path, 'w') as file:
        writer = csv.writer(file)
        writer.writerow([description])
                
        entries = os.listdir(image_dir)
        entries.sort()

        for entry in entries:
            if entry[-1]!='g': continue
            writer.writerow([entry])

def combine_csv(save_path:str, 
                file_path1:str, 
                file_path2:str): 
    f1 = pd.read_csv(file_path1)
    f2 = pd.read_csv(file_path2)

    with open(save_path, 'w') as file:
        writer = csv.writer(file)
        writer.writerow([f1.columns[0],f2.columns[0]])

        for idx in range(len(f1)):
            f1_file = f1.iloc[idx, 0]
            f2_file = f2.iloc[idx, 0]
            writer.writerow([f1_file, f2_file])           

def create_dataset_csv(root_dir):
    dir_ph = os.path.join(root_dir,"photo")
    dir_seg = os.path.join(root_dir,"segmentation")
    if len(os.listdir(dir_seg)) > 10:
        csv_ph = os.path.join(root_dir,"photo.csv")
        csv_seg = os.path.join(root_dir,"segmentation.csv")
        csv_data = os.path.join(root_dir,"dataset.csv")
        write_csv(csv_ph, dir_ph, overwrite=True)
        write_csv(csv_seg, dir_seg, overwrite=True)
        combine_csv(csv_data, csv_ph, csv_seg)
        os.remove(csv_ph)
        os.remove(csv_seg)

def create_folder_and_resize_photo(new_folder_name:str, image_folder:str, resize_shape=(512,384)):

    root_dir = new_folder_name
    save_photo_dir = os.path.join(root_dir,"photo")

    create_folder(root_dir)
    rescale_and_save(image_folder=image_folder, save_folder=save_photo_dir, resize_shape=resize_shape)
    create_dataset_csv(root_dir)

#----------training & using network---------------

def get_loss(generated_image:tensor, 
             target_image:tensor,
             ssim=0,
             lpips=0, 
             psnr=0):
    loss = F.l1_loss(generated_image, target_image)
    if type(ssim)!=type(0): loss += (1  - ssim(generated_image, target_image) )/10 #1-ssim #lpips, #1/(1+|psnr|) 
    if type(lpips)!=type(0): loss += lpips(generated_image, target_image)/10 
    if type(psnr)!=type(0): loss += 1/(1 + psnr(generated_image, target_image) )/10 
    tot_loss = loss
    return tot_loss 

def GAN_get_loss(label:tensor, 
                 predict_label:tensor):
    GAN_loss = F.binary_cross_entropy(label,predict_label)
    return GAN_loss 

def get_loader_vals(values:dict, device:device):
    return values['photo'].to(device), values['segmentation'].to(device), values['target'].to(device)

def get_model(model_path:str, device:device):
    model_info_path = os.path.join( os.path.dirname(model_path), "model_info.json")
    with open(model_info_path) as json_file:
        model_info = json.load(json_file)
    layer_channels = model_info["layer_channels"]
    skip_layers = model_info["skip_layers"]
    is_try = True
    try: 
        if is_try:
            model = DTCNN(layer_channels=layer_channels, skip_layers=skip_layers).to(device) 
            model.load_state_dict( torch.load( model_path, map_location=torch.device(device) ) )
            is_try=False
    except: 
        pass
    try: 
        if is_try:
            model = SDTCNN(layer_channels=layer_channels, skip_layers=skip_layers).to(device) 
            model.load_state_dict( torch.load( model_path, map_location=torch.device(device) ) )
            is_try=False
    except: 
        pass
    try: 
        if is_try:
            model = VGG_SDTCNN(lay=28).to(device) 
            model.load_state_dict( torch.load( model_path, map_location=torch.device(device) ) )
            is_try=False
    except: 
        pass
    try: 
        if is_try:
            model = VGG_SDTCNN(lay=35).to(device) 
            model.load_state_dict( torch.load( model_path, map_location=torch.device(device) ) )
            is_try=False
    except: 
        pass

    return model

def get_dataloader( root_dir:str,
                    usage="train", 
                    bg_mode="train", #train,test,255
                    validation_length=10, 
                    BATCH_SIZE=3, 
                    mask_erosion=0, 
                    bg_dilation=0, 
                    special_img = "no",
                    subset = -1):
    
    trms = [RandomCrop((512,384))] 
    if usage != "no_mask": 
        trms.append( AddDilatedBackground(bg_dilation) )
    trms.append( ToErodedMask(mask_erosion) )
    if usage == "train":  
        trms.append( A_transforms() )
    else: trms.append( A_Norm() )
    trms = transforms.Compose(trms)
    
    dataset = domain_transfer_dataset(root_dir, transform=trms, background_mode=bg_mode, specific_background_path=special_img, subset=subset)
    train_set, val_set = torch.utils.data.random_split(dataset, [len(dataset)-validation_length,validation_length], generator=torch.Generator().manual_seed(0))
    
    active_dataset = train_set
    is_shuffle = True
    if usage != "train" and usage != "train2":
        active_dataset = val_set
        is_shuffle = False

    dataset_loader = torch.utils.data.DataLoader(active_dataset, batch_size=BATCH_SIZE, shuffle=is_shuffle, num_workers=4)
    return dataset_loader

def train_n_epochs(root_dir:str, 
                   model:DTCNN,#OR SDTCNN 
                   train_info:dict, 
                   save_folder="-1", 
                   update_dataset_per_epoch = True, 
                   extra_info={0:0}):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    bg_mode = "255" if "bg255" in extra_info else "train"

    dataset_loader = get_dataloader(root_dir, BATCH_SIZE=train_info['batch_size'], usage="train", bg_mode=bg_mode, bg_dilation=extra_info['dilation'], mask_erosion=extra_info['erosion'])    
    ld = len(dataset_loader)

    #VGG_loss = VGGPerceptualLoss(extra_info['resize'], device)
    ssim = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0,).to(device) if "ssim" in extra_info else 1
    psnr = torchmetrics.PeakSignalNoiseRatio().to(device) if "psnr" in extra_info else 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device) if "lpips" in extra_info else 1

    optimizer = optim.Adam(model.parameters(), lr=train_info['lr'])
    writer = SummaryWriter(save_folder)
    if save_folder=="-1": writer = SummaryWriter()
    
    for epoch in range(train_info['epochs']):   
        if update_dataset_per_epoch and epoch > 0:  dataset_loader = get_dataloader(root_dir, BATCH_SIZE=train_info['batch_size'], usage="train", bg_mode=bg_mode, bg_dilation=extra_info['dilation'], mask_erosion=extra_info['erosion'])    

        for ix, x in enumerate(tqdm(dataset_loader)):
            i_scalar = epoch*ld + ix
            optimizer.zero_grad()
            
            input_image, segmentation_image, target_image = get_loader_vals(x, device)
            generated_image = model(input_image, segmentation_image)
            loss = get_loss(generated_image, target_image, ssim=ssim, lpips=lpips, psnr=psnr)
            #loss = VGG_loss(generated_image, target_image, extra_info['feature_layers'])

            writer.add_scalar('Training Loss', loss, global_step=i_scalar)
            loss.backward()
            optimizer.step()

        grid1 = torchvision.utils.make_grid((input_image+1)/2)
        grid2 = torchvision.utils.make_grid(generated_image)
        grid3 = torchvision.utils.make_grid(target_image)
        grid = torch.concatenate((grid1,grid2,grid3),dim=1)
        writer.add_image('images', grid, global_step=epoch)
        torch.save(model.state_dict(), save_folder + '/test'+str(epoch)+'.pth')
        
        if epoch == 0:
            info = {'lr':train_info['lr'], 'batch_size':train_info['batch_size'], 'epochs':train_info['epochs'], 'update_dataset_per_epoch':update_dataset_per_epoch,  'layer_channels':extra_info['layer_channels'], 'skip_layers':extra_info['skip_layers'], 'dilation':extra_info['dilation'], 'erosion':extra_info['erosion']}
            info_file = os.path.join(save_folder,'model_info.json')
            with open(info_file, 'w') as outfile:
                json.dump(info, outfile)
    writer.close()

def train_n_epochs_twice(root_dir:str, 
                         model:DTCNN,#OR SDTCNN 
                         train_info:dict, 
                         save_folder="-1", 
                         update_dataset_per_epoch = True, 
                         extra_info={0:0}):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    bg_mode = "255" if "bg255" in extra_info else "train"

    dataset_loader = get_dataloader(root_dir, BATCH_SIZE=train_info['batch_size'], usage="train", bg_mode=bg_mode, bg_dilation=extra_info['dilation'], mask_erosion=extra_info['erosion'])    
    ld = len(dataset_loader)

    ssim = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0,).to(device) if "ssim" in extra_info else 1
    psnr = torchmetrics.PeakSignalNoiseRatio().to(device) if "psnr" in extra_info else 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device) if "lpips" in extra_info else 1

    optimizer = optim.Adam(model.parameters(), lr=train_info['lr'])
    writer = SummaryWriter(save_folder)
    if save_folder=="-1": writer = SummaryWriter()
    
    for epoch in range(train_info['epochs']):   
        if update_dataset_per_epoch and epoch > 0: dataset_loader = get_dataloader(root_dir, BATCH_SIZE=train_info['batch_size'], usage="train", bg_mode=bg_mode, bg_dilation=extra_info['dilation'], mask_erosion=extra_info['erosion'])    
            
        for ix, x in enumerate(tqdm(dataset_loader)):
            i_scalar = epoch*ld + ix
            optimizer.zero_grad()
            
            input_image, segmentation_image, target_image = get_loader_vals(x, device)
            generated_image = model(input_image, segmentation_image)
            generated_image2 = model(generated_image*2-1, segmentation_image)
            
            loss = get_loss(generated_image, target_image, ssim=ssim, lpips=lpips, psnr=psnr)
            loss += 0.2*get_loss(generated_image2, target_image, ssim=ssim, lpips=lpips, psnr=psnr)

            writer.add_scalar('Training Loss', loss, global_step=i_scalar)
            loss.backward()
            optimizer.step()

        grid1 = torchvision.utils.make_grid((input_image+1)/2)
        grid2 = torchvision.utils.make_grid(generated_image)
        grid3 = torchvision.utils.make_grid(generated_image2)
        grid4 = torchvision.utils.make_grid(target_image)
        grid  = torch.concatenate((grid1,grid2,grid3,grid4),dim=1)
        writer.add_image('images', grid, global_step=epoch)
        torch.save(model.state_dict(), save_folder + '/test'+str(epoch)+'.pth')
        
        if epoch == 0:
            info = {'lr':train_info['lr'], 'batch_size':train_info['batch_size'], 'epochs':train_info['epochs'], 'update_dataset_per_epoch':update_dataset_per_epoch,  'layer_channels':extra_info['layer_channels'], 'skip_layers':extra_info['skip_layers'], 'dilation':extra_info['dilation'], 'erosion':extra_info['erosion']}
            info_file = os.path.join(save_folder,'model_info.json')
            with open(info_file, 'w') as outfile:
                json.dump(info, outfile)
    writer.close()

def train_GAN_epochs(root_dir:str, 
                   model:SDTCNN, 
                   discriminator:GAN_discriminator,
                   train_info:dict, 
                   save_folder="-1", 
                   update_dataset_per_epoch = True, 
                   extra_info={0:0}):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    discriminator = discriminator.to(device)
    bg_mode = "255" if "bg255" in extra_info else "train"

    BATCH_SIZE = train_info['batch_size']
    dataset_loader = get_dataloader(root_dir, BATCH_SIZE=BATCH_SIZE, usage="train", bg_mode=bg_mode, bg_dilation=extra_info['dilation'], mask_erosion=extra_info['erosion'])    
    ld = len(dataset_loader)

    ssim,psnr,lpips = 1,1,1
    gen_optimizer = optim.Adam(model.parameters(), lr=train_info['lr'])
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=train_info['lr'])

    writer = SummaryWriter(save_folder)
    if save_folder=="-1": writer = SummaryWriter()
    for epoch in range(train_info['epochs']):   
        dataset_loader = get_dataloader(root_dir, BATCH_SIZE=train_info['batch_size'], usage="train", bg_mode=bg_mode, bg_dilation=extra_info['dilation'], mask_erosion=extra_info['erosion'],subset=(ld-1)*BATCH_SIZE)    

        for ix, x in enumerate(tqdm(dataset_loader)):
            i_scalar = epoch*ld + ix
            
            input_image, segmentation_image, target_image = get_loader_vals(x, device)
            generated_image = model(input_image, segmentation_image)
            
            real_label = discriminator(target_image)
            gen_label = discriminator(generated_image)
            real_target = torch.ones((BATCH_SIZE,1),requires_grad=True).to(device)
            gen_target = torch.zeros((BATCH_SIZE,1),requires_grad=True).to(device)

            if i_scalar%3==0: 
                disc_optimizer.zero_grad()
                disc_loss =  GAN_get_loss(gen_label, gen_target)/1500
                disc_loss += GAN_get_loss(real_label, real_target)/1500
                disc_loss.backward(retain_graph=True)
                disc_optimizer.step()

            gen_optimizer.zero_grad()
            gen_loss = get_loss(generated_image, target_image, ssim=ssim, lpips=lpips, psnr=psnr)   
            gen_loss += GAN_get_loss(gen_label.detach(), real_target)/1500

            writer.add_scalar('Training Loss', gen_loss.item(), global_step=i_scalar)
            gen_loss.backward()   
            gen_optimizer.step()

        grid1 = torchvision.utils.make_grid((input_image+1)/2)
        grid2 = torchvision.utils.make_grid(generated_image)
        grid3 = torchvision.utils.make_grid(target_image)
        grid = torch.concatenate((grid1,grid2,grid3),dim=1)
        writer.add_image('images', grid, global_step=epoch)
        if epoch % 2 == 0:
            torch.save(model.state_dict(), save_folder + '/test'+str(epoch)+'.pth')
            torch.save(discriminator.state_dict(), save_folder + "/disc"+str(epoch)+".pth")
        if epoch == 0:
            info = {'lr':train_info['lr'], 'batch_size':train_info['batch_size'], 'epochs':train_info['epochs'], 'update_dataset_per_epoch':update_dataset_per_epoch,  'layer_channels':extra_info['layer_channels'], 'skip_layers':extra_info['skip_layers'], 'dilation':extra_info['dilation'], 'erosion':extra_info['erosion']}
            info_file = os.path.join(save_folder,'model_info.json')
            with open(info_file, 'w') as outfile:
                json.dump(info, outfile)
    writer.close()

def train_GAN2_epochs(root_dir:str, 
                   model:SDTCNN, 
                   discriminator:GAN_discriminator,
                   train_info:dict, 
                   save_folder="-1", 
                   update_dataset_per_epoch = True, 
                   extra_info={0:0}):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "/home/isac/data/tensorboard_info/20230422-185344vgg19stdnn_lay35_l1_gan/test8.pth"
    disc_path = "/home/isac/data/tensorboard_info/20230422-185344vgg19stdnn_lay35_l1_gan/disc8.pth"
    
    model = VGG_SDTCNN(lay=35).to(device)
    model.load_state_dict( torch.load(model_path, map_location=torch.device(device)) )  

    discriminator = VGG_discriminator(lay=28).to(device)
    discriminator.load_state_dict( torch.load(disc_path, map_location=torch.device(device)) )  

    bg_mode = "255" if "bg255" in extra_info else "train"
    BATCH_SIZE = train_info['batch_size']
    root_dir_viton = "/home/isac/data/viton_hd"
    root_dir_wild = "/home/isac/data/f550k"
    dataset_loader_viton = get_dataloader(root_dir_viton, BATCH_SIZE=BATCH_SIZE, usage="train", bg_mode=bg_mode, bg_dilation=extra_info['dilation'], mask_erosion=extra_info['erosion'])    
    ld = len(dataset_loader_viton)-1
    ssim,psnr,lpips = 1,1,1
    gen_optimizer = optim.Adam(model.parameters(), lr=train_info['lr'])
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=train_info['lr'])

    writer = SummaryWriter(save_folder)
    if save_folder=="-1": writer = SummaryWriter()
    for epoch in range(train_info['epochs']):   
        dataset_loader_viton = get_dataloader(root_dir_viton, BATCH_SIZE=BATCH_SIZE, usage="train2", bg_mode=bg_mode, bg_dilation=extra_info['dilation'], mask_erosion=extra_info['erosion'],subset=ld*BATCH_SIZE)    
        dataset_loader_wild = get_dataloader(root_dir_wild, BATCH_SIZE=BATCH_SIZE, usage="train2", bg_mode=bg_mode, bg_dilation=extra_info['dilation'], mask_erosion=extra_info['erosion'],subset=ld*BATCH_SIZE)    

        for ix, datas in tqdm(enumerate(zip(dataset_loader_viton, dataset_loader_wild)),total=ld): # fail with dataloder + iter
            viton_data = datas[0]
            wild_data = datas[1]
            i_scalar = epoch*ld + ix
            input_image_viton, segmentation_image_viton, target_image_viton = get_loader_vals(viton_data, device)
            input_image_wild, segmentation_image_wild, target_image_wild = get_loader_vals(wild_data, device)

            generated_image_viton = model(input_image_viton, segmentation_image_viton)
            generated_image_wild = model(input_image_wild, segmentation_image_wild)
            gen_viton_label = discriminator(generated_image_viton)
            gen_wild_label = discriminator(generated_image_wild)
            real_target = torch.ones((BATCH_SIZE,1),requires_grad=True).to(device)
            gen_target = torch.zeros((BATCH_SIZE,1),requires_grad=True).to(device)

            if i_scalar%3==0: 
                disc_optimizer.zero_grad()
                real_viton_label = discriminator(target_image_viton)
                disc_loss  = GAN_get_loss(real_viton_label, real_target)/1500
                disc_loss += GAN_get_loss(gen_viton_label,  gen_target)/1500
                disc_loss += GAN_get_loss(gen_wild_label,   gen_target)/5000
                disc_loss.backward(retain_graph=True)
                disc_optimizer.step()
            
            gen_optimizer.zero_grad()
            gen_loss = get_loss(generated_image_viton, target_image_viton, ssim=ssim, lpips=lpips, psnr=psnr)   
            gen_loss += GAN_get_loss(gen_viton_label.detach(), real_target)/1500
            gen_loss += GAN_get_loss(gen_wild_label.detach(), real_target)/4000

            writer.add_scalar('Training Loss', gen_loss.item(), global_step=i_scalar)
            gen_loss.backward()   
            gen_optimizer.step()

        grid1 = torchvision.utils.make_grid((input_image_wild+1)/2)
        grid2 = torchvision.utils.make_grid(generated_image_wild)
        grid3 = torchvision.utils.make_grid(target_image_wild)
        grid = torch.concatenate((grid1,grid2,grid3),dim=1)
        writer.add_image('images', grid, global_step=epoch)

        if epoch % 2 == 0:
            torch.save(model.state_dict(), save_folder + '/test'+str(epoch)+'.pth')
            torch.save(discriminator.state_dict(), save_folder + "/disc"+str(epoch)+ ".pth")
        if epoch == 0:
            info = {'lr':train_info['lr'], 'batch_size':train_info['batch_size'], 'epochs':train_info['epochs'], 'update_dataset_per_epoch':update_dataset_per_epoch,  'layer_channels':extra_info['layer_channels'], 'skip_layers':extra_info['skip_layers'], 'dilation':extra_info['dilation'], 'erosion':extra_info['erosion']}
            info_file = os.path.join(save_folder,'model_info.json')
            with open(info_file, 'w') as outfile:
                json.dump(info, outfile)
    writer.close()

def get_psnr_lpips_ssim(target_data:tensor, 
                        generated_data:tensor, 
                        model:DTCNN,#OR SDTCNN
                        score_and_device:dict, 
                        name_extend="viton"):
    device = score_and_device['device']

    ssim_tot = 0 
    lpips_tot = 0
    psnr_tot = 0
    for x1, x2 in tqdm(zip(target_data, generated_data),total=len(target_data)):
        target               = x1['target'].to(device)
        photo                = x2['photo'].to(device)
        segmentation         = x2['segmentation'].to(device)
        generated_img        = model(photo, segmentation)
        ssim_tot  += score_and_device['ssim'](target, generated_img).item()
        lpips_tot += score_and_device['lpips'](target, generated_img).item()
        psnr_tot  += score_and_device['psnr'](target, generated_img).item()

    ssim_avg  = ssim_tot/len(target_data)
    lpips_avg = lpips_tot/len(target_data)
    psrn_avg  = psnr_tot/len(target_data)
    scores = {'ssim_'+name_extend:ssim_avg, 'lpips_'+name_extend:lpips_avg, 'psnr_'+name_extend:psrn_avg}
    return scores

def get_fid_score(target_data:tensor, 
                  generated_data:tensor, 
                  model:DTCNN,#OR SDTCNN 
                  score_and_device:dict):
    device = score_and_device['device']
    for x1, x2 in tqdm(zip(target_data, generated_data), total=len(target_data)):
        target         = x1['target'].to(device)
        photo          = x2['photo'].to(device)
        segmentation   = x2['segmentation'].to(device)
        generated_img  = model(photo, segmentation)
        score_and_device['fid'].update(target, real=True)
        score_and_device['fid'].update(generated_img, real=False)

    fid_score = {'fid':score_and_device['fid'].compute().item()}
    return fid_score

def get_metrics(model_path:str, 
                viton_500:DataLoader, 
                f550k_500:DataLoader, 
                viton_5000:DataLoader, 
                f550k_5000:DataLoader, 
                score_and_device:dict):

    device = score_and_device['device']
    model = get_model(model_path, device)
    model.eval()
    with torch.no_grad():
        scores1 = get_psnr_lpips_ssim(viton_500, viton_500, model, score_and_device, "viton")
        scores2 = get_psnr_lpips_ssim(viton_500, f550k_500, model, score_and_device, "f550k")
    if 'fid' not in score_and_device:
        fid_score={'fid':-1}
    else:    
        fid_score = get_fid_score(viton_5000, f550k_5000, model, score_and_device)
    scores = scores1 | scores2 | fid_score
    file = os.path.join( os.path.dirname(model_path), 'scores1.json')#scores1 == new test
    with open(file, 'w') as outfile:
        json.dump(scores, outfile)

def tensor_to_saveable_img(tensor:tensor):
    if torch.cuda.is_available():
        tensor = tensor.detach().cpu()
    y = torch.transpose(tensor, 0, 2)
    y = torch.transpose(y, 0, 1)
    y = np.array(y)
    if np.amin(y) < 0 : y = (y+1)/2 
    y[y<0] = 0
    y[y>1] = 1
    return y

def save_images(savefolder:str, 
                batch_images:tensor, 
                as_batch=False):
    existing_images = len(os.listdir(savefolder))
    
    if as_batch :    
        grid = torchvision.utils.make_grid(batch_images)
        save_path = os.path.join(savefolder, "img" + str(existing_images)+".png")
        plt.imsave(save_path, tensor_to_saveable_img(grid))
    else :
        for i in range( batch_images.shape[0] ) :
            save_path = os.path.join(savefolder, "img" + str(existing_images+i)+".png")
            plt.imsave(save_path, tensor_to_saveable_img(batch_images[i]))
   
def generate_images(dataloader:DataLoader,
                    model_path:str,
                    save_folder:str,
                    max_images=-1, 
                    photo_mode=0):
    if not os.path.isdir(save_folder): os.mkdir(save_folder) 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(model_path, device)
    model.eval()
    ix = 0

    s1 = os.path.join(save_folder,"photo")
    s2 = os.path.join(save_folder,"segmentation")


    for x in tqdm(dataloader):
        input_image, segmentation_image, target_image = get_loader_vals(x, device)
        generated_images = model(input_image, segmentation_image)
        input_image = (input_image+1)/2 
        segmentation_image = (segmentation_image+1)/2 

        if photo_mode==0:
            grid0 = torchvision.utils.make_grid(segmentation_image)
            grid1 = torchvision.utils.make_grid(input_image)
            grid2 = torchvision.utils.make_grid(target_image)
            grid3 = torchvision.utils.make_grid(generated_images)
            grid_tot = torch.concat((grid0,grid1,grid2,grid3),dim=1)
            save_images(save_folder, grid_tot, as_batch=True)
            ix+=1
        elif photo_mode == 1:
            for i in range(segmentation_image.shape[0]):
                img1 = segmentation_image[i,:,:,:]
                img2 =        input_image[i,:,:,:]
                img3 =       target_image[i,:,:,:]
                img4 =   generated_images[i,:,:,:]
                grid0 = torchvision.utils.make_grid(img1)
                grid1 = torchvision.utils.make_grid(img2)
                grid2 = torchvision.utils.make_grid(img3)
                grid3 = torchvision.utils.make_grid(img4)
                grid_tot1 = torch.concat((grid0,grid1),dim=1)
                grid_tot2 = torch.concat((grid2,grid3),dim=1)
                grid_tot = torch.concat((grid_tot1,grid_tot2),dim=2)
                save_images(save_folder, grid_tot, as_batch=True)
                ix+=1
        elif photo_mode == 2:
            save_images(save_folder, input_image, as_batch=False)
            ix+=1
        elif photo_mode == 3:
            save_images(s1, generated_images, as_batch=False)
            save_images(s2, segmentation_image, as_batch=False)
            ix+=1
        elif photo_mode == 4:
            save_images(save_folder, generated_images, as_batch=False)
            ix+=1
        else: break
        if ix > max_images-2 and max_images > 0: break

def compare_bg_images(dataloader:DataLoader, save_folder, sub_img=(-1,0,0,0), dim='vertical'):
    for x in dataloader:
        break
    if dim=='vertical': dim = 1
    else: dim = 2
    target = x['target'].to(float)
    mask = (x['segmentation']+1)/2
    mask.to(int)
    background = x['background'].to(float)
    output = target*mask + background*(1-mask)
    if sub_img[0]>=0:
        output = output[:,:,sub_img[0]:sub_img[1],sub_img[2]:sub_img[3]]
        target = target[:,:,sub_img[0]:sub_img[1],sub_img[2]:sub_img[3]]

    for i in range(x['target'].shape[0]):
        target1      = target[    i, :, :, :]
        output1      = output[    i, :, :, :]
        
        grid_tot = torch.cat((target1, output1),dim=dim)
        save_images(save_folder, grid_tot, as_batch=True)

def concat_save_2imgs(save_path:str, img1:str, img2:str, dim='vertical'):
    if dim=='vertical': dim = 0
    else: dim = 1
    img1 = io.imread(img1)[:,:,:3]
    img2 = io.imread(img2)[:,:,:3]
    try:
        grid = np.concatenate((img1,img2),axis=dim)
    except: 
        grid = np.concatenate((img1,img2),axis=1-dim)
    plt.imsave(save_path, grid)

#fun remove later
def print_np_img(img, space = 4):

    if np.max(img)<2: img *= 255
    img = img[:,:,0].astype(int)
    Y, X = img.shape
    
    format_str1 = ""
    for x in range(X):
        format_str1 += "{:<{}}"
    format_str = format_str1.format
    
    spaces = [space]*X
    for y in range(Y):
        im_row = list(img[y,:]) 
        params = [elem for pair in zip(im_row, spaces) for elem in pair]
        print(format_str(*params))

def print_colored_box(arr):
    c = [92,93,91,95,96,94,97,98,2,90,7]
    imin = 0
    cd = {}
    for i in c:
        imax = imin + 255//len(c)
        cd[(imin,imax)]=str(i)
        imin=imax

    bc = '1'
    b_char = '\u2588'
    for i in arr:
        bg_code='1'
        for x in cd:
            if i >=x[0] and i < x[1]:
                bg_code = cd[x]
        top_border = f'\x1b[{bc};{bg_code}m{b_char}\x1b[0m'
        print(top_border,end="")
    print("")

def print_colored_img(arr,scale=3):
    arr = arr[:,:,0]
    Y,X = arr.shape
    Y = [y for y in range(Y)][::scale]
    
    for y in Y:
        arr2 = list(arr[y,:].astype(int))[::scale]
        print_colored_box(arr2)

#add scheduler and run new 
