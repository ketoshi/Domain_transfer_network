import os
import torch
import torchvision
from skimage import io, transform
import matplotlib.pyplot as plt
from utils.utils import get_dataloader, generate_images, write_csv, combine_csv, compare_bg_images, concat_save_2imgs, get_model
root_dir   = '/home/isac/data/f550k'
model_path = "/home/isac/data/tensorboard_info/20230330-041426{'dilation': 0, 'erosion': 0, 'p': 0.25, 'skip_layers': [0, 1, 2, 3, 4], 'layer_channels': (3, 64, 128, 256, 512, 1024), 'function': 'train_n_epochs'}/test9.pth"
save_folder = '/home/isac/data/zalando/output'

img1 = "/home/isac/data/zalando/photo/img48.jpg"
img2 = "/home/isac/data/zalando/segmentation/img48.jpg"
dataloader = get_dataloader(root_dir=root_dir,
                            usage='use_mask',
                            bg_mode="validation",
                            validation_length=500,
                            BATCH_SIZE=3,
                            bg_dilation=0,
                            mask_erosion=0,
                    #TODO:
                    #to improve: 1) lighting (hopefully shado fixes it),  2) (backup plan) HDR imaging-luminence estimation (relighting related) (image->hdr)
                    # get rid of this dataloader, and torch.utils.data.DataLoader(
                    # use only neccesary/ data extraction  (priority on 1 image but else multiple, no batch size should be used)   
                    #A.ISONoise(p=1), this augmentation usefull. look up what ISO does!
                    #to improve: 1) lighting (hopefully shado fixes it),  2) (backup plan) HDR imaging-luminence estimation (relighting related) (image->hdr)
                    # if doesn' t got wll can use GANloss
)

#generate_images(dataloader, model_path, save_folder, max_images=1, photo_mode=0)#photo_mode 0,1,2, max images = -1 for generate all photos

#compare_bg_images(dataloader,save_folder)
#concat_save_2imgs(save_path="/home/isac/data/zalando/output/img5.png" , img1="/home/isac/data/zalando/output/img1.png", img2="/home/isac/data/viton_hd/photo/0.jpg")



if not os.path.isdir(save_folder): os.mkdir(save_folder) 
ix = len(os.listdir(save_folder))
save_path = os.path.join(save_folder,"img"+str(ix)+".png")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_model(model_path, device)
model.eval()
input_image = io.imread(img1)
segmentation_image = io.imread(img2)
generated_images = model(input_image, segmentation_image)
input_image = (input_image+1)/2 
segmentation_image = (segmentation_image+1)/2 

grid0 = torchvision.utils.make_grid(segmentation_image)
grid1 = torchvision.utils.make_grid(input_image)
grid2 = torchvision.utils.make_grid(generated_images)
grid_tot = torch.concat((grid0,grid1,grid2),dim=1)


plt.imsave(save_path,grid_tot)

