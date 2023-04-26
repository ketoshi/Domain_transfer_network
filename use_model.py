import os
import torch
import warnings
import torchmetrics
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import json
from utils.utils import get_dataloader, generate_images, get_metrics, train_n_epochs, train_n_epochs_twice,train_GAN_epochs,train_GAN2_epochs
from utils.layers import DTCNN, SDTCNN, GAN_discriminator,VGG_SDTCNN, VGG_discriminator

import datetime
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image.fid import FrechetInceptionDistance

root_dir   = '/home/isac/data/viton_hd'
root_f550k_dir = '/home/isac/data/f550k'
model_path = "/home/isac/data/tensorboard_info/20230323-002150{'dilation': 0, 'erosion': 0, 'color': False, 'skip_layers': [0, 1, 2, 3, 4, 5], 'layer_channels': (3, 64, 128, 256, 512, 1024, 2048), 'function': 'train_n_epochs'}/test19.pth"
save_folder = '/home/isac/data/use_model_output' # for generating images

PROGRAM_TASK = "EVALUATE" #GENERATE, EVALUATE, TRAIN 

if PROGRAM_TASK == "GENERATE": #try with both backgrounds
    dataloader = get_dataloader(root_dir=root_f550k_dir,
                                usage='use_model_use_mask',
                                bg_mode="validation",
                                validation_length=500,
                                BATCH_SIZE=3,
                                bg_dilation=0,
                                mask_erosion=0
    )    
    dir = "/home/isac/data/tensorboard_info"
    sc = os.listdir(dir) 
    sc.sort()
    for x in sc:
        model_path = os.path.join(dir,x,"test4.pth")
    
    model_path = "/home/isac/data/tensorboard_info/20230323-002150{'dilation': 0, 'erosion': 0, 'color': False, 'skip_layers': [0, 1, 2, 3, 4, 5], 'layer_channels': (3, 64, 128, 256, 512, 1024, 2048), 'function': 'train_n_epochs'}/test19.pth"
    generate_images(dataloader, model_path, save_folder, max_images=1, photo_mode=0)

if PROGRAM_TASK == "EVALUATE":
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fid = FrechetInceptionDistance(feature=768, normalize=True).to(device)
    ssim_model = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0,).to(device)
    psnr_model = torchmetrics.PeakSignalNoiseRatio().to(device)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lpips_model = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
    score_and_device = {'fid':fid, 'ssim':ssim_model, 'psnr':psnr_model,'lpips':lpips_model, 'device':device}

    dirs = [
            #"/home/isac/data/tensorboard_info_sorted/segmentation",
            "/home/isac/data/tensorboard_info_sorted/losses/test"
        ]
    for dir in dirs:
        model_paths = os.listdir(dir)
        model_paths.sort()
        model_paths = [os.path.join(dir,x,"test4.pth") for x in model_paths]
        for model_path in model_paths:    

            model_info_path = os.path.join( os.path.dirname(model_path), "model_info.json")
            with open(model_info_path) as json_file:
                model_info = json.load(json_file)
            
            bg_mode  = "255" if "bg255" in model_info else "validation"
            usage    = "test"
            bg_dilation = model_info['dilation']
            mask_erosion  = model_info['erosion']

            viton_500 = get_dataloader(root_dir=root_dir,
                                        usage="test",
                                        bg_mode=bg_mode,
                                        validation_length=500,
                                        BATCH_SIZE=1,
                                        bg_dilation=bg_dilation,
                                        mask_erosion=mask_erosion
            )

            f550k_500 = get_dataloader(root_dir=root_dir,
                                        usage="test",
                                        bg_mode=bg_mode,
                                        validation_length=500,
                                        BATCH_SIZE=1,
                                        bg_dilation=bg_dilation,
                                        mask_erosion=mask_erosion,
            )

            viton_5000 = get_dataloader(root_dir=root_dir,
                                        usage=usage,
                                        bg_mode=bg_mode,
                                        validation_length=5000,
                                        BATCH_SIZE=1,
                                        bg_dilation=bg_dilation,
                                        mask_erosion=mask_erosion
            )

            f550k_5000 = get_dataloader(root_dir=root_dir,
                                        usage=usage,
                                        bg_mode=bg_mode,
                                        validation_length=5000,
                                        BATCH_SIZE=1,
                                        bg_dilation=bg_dilation,
                                        mask_erosion=mask_erosion
            )

            get_metrics(model_path, viton_500, f550k_500, viton_5000, f550k_5000, score_and_device)

if PROGRAM_TASK == "TRAIN": 

    train_info = {
    'batch_size': 2, 'epochs':5, 'lr': 1e-3
    }

    # add bg255:0 in extra info if use bg255
    extra_infos = [
        {'dilation':0, 'erosion':0},
    ]
    extra_info = extra_infos[0]

    lll = 64
    params = {'lc':[(3,lll,lll*2,lll*4,lll*8,lll*16), (3,lll,lll*2,lll*4,lll*8,lll*16), (3,lll,lll*2,lll*4,lll*8,lll*16),(3,lll,lll*2,lll*4,lll*8,lll*16),(3,lll,lll*2,lll*4,lll*8,lll*16) ],
     'sk': [[0,1,2,3,4],[0,1,2,3,4],[0,1,2,3,4],[0,1,2,3,4],[0,1,2,3,4]],
     'fc': [train_n_epochs,train_n_epochs,train_n_epochs,train_n_epochs,train_n_epochs]}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ssim = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0).to(device) if "ssim" in extra_info else 1
    psnr = torchmetrics.PeakSignalNoiseRatio().to(device) if "psnr" in extra_info else 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device) if "lpips" in extra_info else 1

    for i in range(len(params['lc'])):
        extra_info = extra_infos[i]
        str_info = str(extra_info)
        SKIP_LAYERS    = params['sk'][i]
        LAYER_CHANNELS = params['lc'][i]
        i_function     = params['fc'][i]
        extra_info.update({"skip_layers":SKIP_LAYERS,"layer_channels":LAYER_CHANNELS, "function":i_function.__name__})

        #discriminator = GAN_discriminator()
        model = SDTCNN(layer_channels=LAYER_CHANNELS, skip_layers=SKIP_LAYERS, img_sz=(-1000,-1000)) #DTCNN or SDTCNN, img_sz=(-1000,-1000)==batchNorm
        #discrimin_model = VGG_discriminator()
        SAVE_FOLDER = '/home/isac/data/tensorboard_info/' +  datetime.datetime.now().strftime("%Y%m%d-%H%M%S") +str_info
        i_function(root_dir, model=model, train_info=train_info, save_folder=SAVE_FOLDER, update_dataset_per_epoch=True, extra_info=extra_info)
