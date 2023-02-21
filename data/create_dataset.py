import os
import sys
import matplotlib.pyplot as plt
from PIL import Image
from dataloader import *

#1)
#   download: viton-hd dataset to 'high_res_photo_dir'
#   download  backgrounds from https://grail.cs.washington.edu/projects/background-matting-v2/#/datasets
#   download  repo: https://github.com/GoGoDuck912/Self-Correction-Human-Parsing  and their LIP model

#2 run python script once
#3 run " python simple_extractor.py --dataset 'lip' --model-restore 'path_to_model' --input-dir 'path_to_high_res_photo_dir' --output-dir 'path_to_segmentation_dir' "
#4 run python script once again


root_dir                = 'data/viton_hd' 
high_res_photo_dir      = os.path.join(root_dir, 'high_res_photo')
photo_dir               = os.path.join(root_dir, 'photo')
segmentation_dir        = os.path.join(root_dir, 'segmentation')
background_dir          = os.path.join(root_dir, 'background')
high_res_background_dir = os.path.join(root_dir, 'high_res_background')

photo_csv               = os.path.join(root_dir, 'photo.csv')
segmentation_csv        = os.path.join(root_dir, 'segmentation.csv')
background_csv          = os.path.join(root_dir, 'background.csv')
high_res_photo_csv      = os.path.join(root_dir, 'high_res_photo.csv')
high_res_background_csv = os.path.join(root_dir, 'high_res_background.csv')
dataset_csv             = os.path.join(root_dir, 'dataset.csv')


write_csv(csv_file_path=high_res_photo_csv, image_dir=high_res_photo_dir, description='photo')
write_csv(csv_file_path=high_res_background_csv, image_dir=high_res_background_dir, description='background')

if len(os.listdir(photo_dir)) < 10: 
    rescaled_images = create_dataset(path_to_csv_file=high_res_photo_csv, image_dir = high_res_photo_dir, transform=Rescale((512,384)))
    save_dataset_images(folder = photo_dir, dataset = rescaled_images)

if len(os.listdir(background_dir)) < 10: 
    cropped_backgrounds = create_dataset(path_to_csv_file = background_csv, image_dir = high_res_background_dir, transform = RandomCrop((512,384)))
    save_dataset_images(background_dir, cropped_backgrounds)

if len(os.listdir(segmentation_dir)) < 10: 
    write_csv(csv_file_path=photo_csv, image_dir=photo_dir, description='photo')
    write_csv(csv_file_path=segmentation_csv, image_dir=segmentation_dir, description='segmentation')
    combine_csv(save_path=dataset_csv, file_path1=photo_csv, file_path2=segmentation_csv)

#TODO: clean up unceesary csv files perhaps?




