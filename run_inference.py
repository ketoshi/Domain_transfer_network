import os
from utils.utils import get_dataloader, generate_images, write_csv, combine_csv
root_dir   = '/home/isac/data/use_model_input'
model_path = "/home/isac/data/tensorboard_info/20230323-002150{'dilation': 0, 'erosion': 0, 'color': False, 'skip_layers': [0, 1, 2, 3, 4, 5], 'layer_channels': (3, 64, 128, 256, 512, 1024, 2048), 'function': 'train_n_epochs'}/test19.pth"

save_folder = '/home/isac/data/use_model_input/output'
if not os.path.isdir(save_folder): os.mkdir(save_folder)

write_csv(os.path.join(root_dir,"photo.csv"),        os.path.join(root_dir,"photo"), description='image',        overwrite=True)
write_csv(os.path.join(root_dir,"segmentation.csv"), os.path.join(root_dir,"segmentation"), description='segmentation', overwrite=True)
combine_csv(os.path.join(root_dir,"dataset.csv"), os.path.join(root_dir,"photo.csv"), os.path.join(root_dir,"segmentation.csv"))

dataloader = get_dataloader(root_dir=root_dir,
                            usage='use_model_use_mask',
                            bg_mode="validation",
                            validation_length=500,
                            BATCH_SIZE=3,
                            dilation=0,
                            erosion=0,
                            get_color_segmentation=False
)
generate_images(dataloader, model_path, save_folder, max_images=1, photo_mode=1)#photo_mode 0,1,2













