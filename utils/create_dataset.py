import os
from dataset import *
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
import cv2
from utils import rescale_and_save, create_folder_and_resize_photo

def create_dataset(save_folder, image_folder, mode = "photo", resize_shape=(512,384)):
    if mode == "photo":
        create_folder_and_resize_photo(save_folder,image_folder, resize_shape)
    if mode == "bg":
        rescale_and_save(save_folder,image_folder)

    #   download  repo: https://github.com/GoGoDuck912/Self-Correction-Human-Parsing  and their LIP model
    #   and  run " python simple_extractor.py --dataset 'lip' --model-restore 'path_to_model' --input-dir 'path_to_high_res_photo_dir' --output-dir 'path_to_segmentation_dir' "
# run script once before and after creating segmented image (of resized images) via comment above

#resize_photos
new_dir = "/home/isac/data/zalando"
photo_dir = "/home/isac/data/zalando/photo"
create_dataset(new_dir, photo_dir, mode="photo")

#resize_background
background_dir = "/home/isac/data/background_255"
save_background_dir = os.path.join( os.path.dirname(new_dir), "background_255")
create_dataset(save_background_dir, background_dir, mode="bg")
