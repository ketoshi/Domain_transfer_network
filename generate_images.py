import os
import torch
import matplotlib.pyplot as plt
import torchvision
from torchinfo import summary
from data.dataset import * #don't need sys .path beacuse this is root folder
from data.utils import *
from model.layers import *
import torch.optim as optim
import datetime


#would like a arg parser here which controls some part of code


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root_dir = '/home/isac/data/viton_hd'
csv_file = os.path.join(root_dir,'dataset.csv')

trms = torchvision.transforms.Compose([RandomCrop((512,384)), ApplyMask(), \
      ToTensor(), LightingMult(), RotateMult(),  NormalizeMult() ]) 
dataset = domain_transfer_dataset(csv_file, root_dir, transform=trms)

train_set, val_set = torch.utils.data.random_split(dataset, [len(dataset)-500,500], generator=torch.Generator().manual_seed(42)) #fixed seed very good! but change to 0

model = DTCNN(layer_channels=(3,64,128,256,512)).to(device) #just to test things then increase
#model_path = 'data/tensorboard_info/20230228-203927 LR 0.1/test1.pth'
#model_path = 'data/tensorboard_info/20230228-203927 LR 0.1/test9.pth'
#model_path = 'data/tensorboard_info/20230228-230357 LR 0.01/test9.pth'
#model_path = 'data/tensorboard_info/20230301-012830 LR 0.001/test9.pth'
#model_path = 'data/tensorboard_info/20230301-035227 LR 0.0001/test9.pth'
#model_path = 'data/tensorboard_info/20230301-061619 LR 1e-05/test9.pth'

#model_path = 'data/tensorboard_info/20230301-012830 LR 0.001/test1.pth'
#model_path = 'data/tensorboard_info/20230301-012830 LR 0.001/test3.pth'
#model_path = 'data/tensorboard_info/20230301-012830 LR 0.001/test5.pth'
#model_path = 'data/tensorboard_info/20230301-012830 LR 0.001/test7.pth'
model_path = 'data/tensorboard_info/20230301-012830 LR 0.001/test9.pth'

model.load_state_dict(torch.load(model_path))
model.eval()

BATCH_SIZE = 4
save_folder = '/home/isac/Domain_transfer_network/data/use_model_output'
dataset_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)#use 4-8 workers, number needs to be tuned so schedule gpu/cpu collaborate (gpu should be faster to proccess than to load a new batch)

for ix, x in enumerate(tqdm(dataset_loader)):
    input_image, segmentation_image, target_image = get_loader_vals(x, device)
    generated_images = model(input_image, segmentation_image)
    save_images(save_folder, generated_images, as_batch=True)
    break

print("program complete, images saved in test_results")



#training ideas (excluding arhictecture modification)
#try include psnr score as loss
#2 losses, one is a downscaled image for more wide but blurry, and original for more detailed
#