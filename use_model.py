import os
import torch
import torchvision
from utils.dataset import domain_transfer_dataset 
from utils.utils import *
from utils.layers import *
import datetime

root_dir   = '/home/isac/data/viton_hd'
root_f550k_dir = '/home/isac/data/f550k'
model_path = '/home/isac/data/tensorboard_info/20230307-020256layers: 3 32/test4.pth'
save_folder = 'data/use_model_output/skip0123'

PROGRAM_TASK = "GENERATE" #GENERATE, EVALUATE, TRAIN 


if PROGRAM_TASK == "GENERATE":
    dataloader = get_dataloader(root_dir=root_dir,
                                usage='use_model_use_mask',
                                use_validation_set=True,
                                validation_set_length=500,
                                BATCH_SIZE=3,
                                kernel_size_dilation=10,
                                kernel_size_erosion=0,
                                get_color_segmentation=True
    )
    generate_images(dataloader, model_path, save_folder, max_images=2)

if PROGRAM_TASK == "EVALUATE":
    dataloader1 = get_dataloader(root_dir=root_dir,
                                usage='use_model_use_mask',
                                use_validation_set=True,
                                validation_set_length=500,
                                BATCH_SIZE=3,
                                kernel_size_dilation=10,
                                kernel_size_erosion=0,
                                get_color_Segmentation=False
    )
    dataloader2 = get_dataloader(root_dir=root_f550k_dir,
                                usage='use_model_use_mask',
                                use_validation_set=True,
                                validation_set_length=500,
                                BATCH_SIZE=3,
                                kernel_size_dilation=10,
                                kernel_size_erosion=0,
                                get_color_Segmentation=False
    )
    get_metrics(dataloader1, dataloader2, model_path)#TODO: uncomplete so far

if PROGRAM_TASK == "TRAIN": #remember to try e.g. erode function!
    LAYER_CHANNELS = (3,32,64,128,256)
    BATCH_SIZE = 3
    EPOCHS = 1
    LR = 1e-3
    SKIP_LAYERS = [0,1,2,3]   #changed network, but before 1e-5 <LR <1e-2  was good  1e-1 was bad maybe anothe run to see that
    functions = [train_n_epochs]

    for i_function in functions:
        SAVE_FOLDER = 'home/isac/data/tensorboard_info/' +  datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "function:" + str(i_function)
        i_function(root_dir,layer_channels=LAYER_CHANNELS, skip_layers=SKIP_LAYERS, lr=LR, batch_size = BATCH_SIZE, epochs=EPOCHS, save_folder=SAVE_FOLDER, update_dataset_per_epoch=True)

