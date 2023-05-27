import os
import torch
import warnings
import torchmetrics
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import json
from utils.utils import get_dataloader, generate_images, generate_images2, get_metrics, train_n_epochs,train_GAN_epochs
from utils.layers import DTCNN, SDTCNN, GAN_discriminator,VGG_SDTCNN, VGG_discriminator

import datetime
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image.fid import FrechetInceptionDistance

root_dir   = '/home/isac/data/viton_hd'
root_f550k_dir = '/home/isac/data/f550k'
PROGRAM_TASK = "EVALUATE" #GENERATE, EVALUATE, TRAIN 

if PROGRAM_TASK == "GENERATE": #try with both backgrounds
    save_folder = '/home/isac/data/use_model_output' # for generating images

    dataloader = get_dataloader(root_dir=root_dir,
                                usage='train',
                                validation_length=500,
                                BATCH_SIZE=1,
                                bg_dilation=0,
                                mask_erosion=0
    )    
    model_path = "/home/isac/data/final/seg/test24.pth"
    generate_images(dataloader, model_path, save_folder, max_images=10,photo_mode=0)

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
            "/home/isac/data/tensorboard_info_sorted/architecture_param/test"
        ]
    for dir in dirs:
        model_paths = os.listdir(dir)
        model_paths.sort()
        model_paths = [os.path.join(dir,x,"test2.pth") for x in model_paths]
        for model_path in model_paths:    

            model_info_path = os.path.join( os.path.dirname(model_path), "model_info.json")
            with open(model_info_path) as json_file:
                model_info = json.load(json_file)    
            bg_dilation = model_info['dilation']
            mask_erosion  = model_info['erosion']

            special_image = "/home/isac/data/bg_test.jpg"#remove
            viton_500 = get_dataloader(root_dir=root_dir,
                                        usage="test",
                                        validation_length=500,
                                        BATCH_SIZE=1,
                                        bg_dilation=bg_dilation,
                                        mask_erosion=mask_erosion,
                                        special_img=special_image,
            )

            f550k_500 = get_dataloader(root_dir=root_f550k_dir,
                                        usage="no_mask",
                                        validation_length=500,
                                        BATCH_SIZE=1,
                                        bg_dilation=bg_dilation,
                                        mask_erosion=mask_erosion,
            )

            viton_5000 = get_dataloader(root_dir=root_dir,
                                        usage="test",
                                        validation_length=5000,
                                        BATCH_SIZE=1,
                                        bg_dilation=bg_dilation,
                                        mask_erosion=mask_erosion,
            )

            f550k_5000 = get_dataloader(root_dir=root_f550k_dir,
                                        usage="no_mask",
                                        validation_length=5000,
                                        BATCH_SIZE=1,
                                        bg_dilation=bg_dilation,
                                        mask_erosion=mask_erosion,
            )

            get_metrics(model_path, viton_500, f550k_500, viton_5000, f550k_5000, score_and_device)

if PROGRAM_TASK == "TRAIN": 

    train_info = {
    'batch_size': 2, 'epochs':41, 'lr':1E-3
    }

    extra_infos = [
        {'dilation':0, 'erosion':0},
    ]
    extra_info = extra_infos[0]

    lll = 60
    params = {'lc':[(3,lll,lll*2,lll*4,lll*8,lll*16)],
     'sk': [[0,1,2,3,4]]}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i in range(len(params['lc'])):
        extra_info = extra_infos[i]
        str_info = str(extra_info)
        SKIP_LAYERS    = params['sk'][i]
        LAYER_CHANNELS = params['lc'][i]
        extra_info.update({"skip_layers":SKIP_LAYERS,"layer_channels":LAYER_CHANNELS})

        discriminator = GAN_discriminator(layer_channels=LAYER_CHANNELS)
        model = SDTCNN(layer_channels=LAYER_CHANNELS, skip_layers=SKIP_LAYERS, img_sz=(-1000,-1000)) #DTCNN or SDTCNN,  img_sz=(-1000,-1000)==batchNorm
        SAVE_FOLDER = '/home/isac/data/tensorboard_info/' +  datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "SEG-GAN"+str(train_info)+"_Exp_scheduler" +"gen_div5_disc_div5every_1_train"
        #train_n_epochs(root_dir, model=model, train_info=train_info, save_folder=SAVE_FOLDER, extra_info=extra_info)
        train_GAN_epochs(root_dir, model=model, discriminator=discriminator, train_info=train_info, save_folder=SAVE_FOLDER, extra_info=extra_info)
