import os
from dataset import *
from utils import write_csv, save_dataset_images, combine_csv
#1)
#   download: viton-hd dataset folder to 'download_dir'
#   download  backgrounds from https://grail.cs.washington.edu/projects/background-matting-v2/#/datasets
#   download  repo: https://github.com/GoGoDuck912/Self-Correction-Human-Parsing  and their LIP model

#2 run python script once
#3 run " python simple_extractor.py --dataset 'lip' --model-restore 'path_to_model' --input-dir 'path_to_high_res_photo_dir' --output-dir 'path_to_segmentation_dir' "
#4 run python script once again


downloaded_dir          = 'data/viton_hd'  
background_dir          = 'data/background'
viton_dir               = 'data/viton_hd'
f550k_dir               = 'data/f550k' 

high_res_photo_dir      = os.path.join(downloaded_dir, 'high_res_photo_dir')# change back
high_res_photo_csv      = os.path.join(downloaded_dir, 'high_res_photo.csv')
high_res_background_dir = os.path.join(downloaded_dir, 'high_res_background')
high_res_background_csv = os.path.join(downloaded_dir, 'high_res_background.csv')
f550k_high_res_dir      = os.path.join(downloaded_dir, 'f550k_high_res')
f550k_high_res_csv      = os.path.join(downloaded_dir, 'f550k_high_res.csv')

viton_photo_dir               = os.path.join(viton_dir, 'photo')
viton_photo_csv               = os.path.join(viton_dir, 'photo.csv')
viton_segmentation_dir        = os.path.join(viton_dir, 'segmentation')
viton_segmentation_csv        = os.path.join(viton_dir, 'segmentation.csv')

f550k_photo_dir               = os.path.join(f550k_dir, 'photo')
f550k_photo_csv               = os.path.join(f550k_dir, 'photo.csv')
f550k_segmentation_dir        = os.path.join(f550k_dir, 'segmentation')
f550k_segmentation_csv        = os.path.join(f550k_dir, 'segmentation.csv')

f550k_dataset_csv       = os.path.join(f550k_dir, 'dataset.csv')
viton_dataset_csv       = os.path.join(viton_dir, 'dataset.csv')

if len(os.listdir(viton_photo_dir)) < 10: 
    write_csv(csv_file_path=high_res_photo_csv, image_dir=high_res_photo_dir, description='photo')
    rescaled_images = create_dataset(path_to_csv_file=high_res_photo_csv, image_dir = high_res_photo_dir, transform=Rescale((512,384)))
    save_dataset_images(folder = viton_photo_dir, dataset = rescaled_images) # takes some time!

if len(os.listdir(viton_segmentation_dir)) < 10: 
    write_csv(csv_file_path=viton_photo_csv,        image_dir=viton_photo_dir,        description='photo')
    write_csv(csv_file_path=viton_segmentation_csv, image_dir=viton_segmentation_dir, description='segmentation')
    combine_csv(save_path=viton_dataset_csv, file_path1=viton_photo_csv, file_path2=viton_segmentation_csv)

if len(os.listdir(background_dir)) < 10: 
    write_csv(csv_file_path=high_res_background_csv, image_dir=high_res_background_dir, description='background')
    cropped_backgrounds = create_dataset(path_to_csv_file = high_res_background_csv, image_dir = high_res_background_dir, transform = RandomCrop((512,384)))
    save_dataset_images(background_dir, cropped_backgrounds)

if len(os.listdir(f550k_high_res_dir)) > 750: # fix 
    x = 1
    for img in os.listdir(f550k_high_res_dir):
        if x>700: 
            file = os.path.join(f550k_high_res_dir, img)
            os.remove(file)
        x+=1
    write_csv(csv_file_path=f550k_high_res_csv, image_dir=f550k_high_res_dir, description='photo')
    rescaled_images = create_dataset(path_to_csv_file=f550k_high_res_csv, image_dir = f550k_high_res_dir, transform=Rescale((512,384)))
    save_dataset_images(folder = f550k_dir, dataset = rescaled_images)
    write_csv(csv_file_path=f550k_photo_csv, image_dir=f550k_dir, description='photo')

if len(os.listdir(f550k_segmentation_csv)) < 10: 
    write_csv(csv_file_path=f550k_photo_csv,        image_dir=f550k_photo_dir,        description='photo')
    write_csv(csv_file_path=f550k_segmentation_csv, image_dir=f550k_segmentation_dir, description='segmentation')
    combine_csv(save_path=f550k_dataset_csv, file_path1=f550k_photo_csv, file_path2=f550k_segmentation_csv)





