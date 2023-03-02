import os
import torch
import matplotlib.pyplot as plt
import torchvision
from torchinfo import summary
from data.dataset import * #don't need sys .path beacuse this is root folder
from data.utils import *
from model.layers import *
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root_dir = '/home/isac/data/viton_hd'
csv_file = os.path.join(root_dir,'dataset.csv')

trms = torchvision.transforms.Compose([RandomCrop((512,384)), ApplyMask(), \
      ToTensor(), LightingMult(), RotateMult(),  NormalizeMult() ]) 
dataset = domain_transfer_dataset(csv_file, root_dir, transform=trms)
train_set, val_set = torch.utils.data.random_split(dataset, [len(dataset)-500,500], generator=torch.Generator().manual_seed(42)) #fixed seed very good!

model = DTCNN(layer_channels=(3,64,128,256,512)).to(device) #just to test things then increase

BATCH_SIZE = 4
EPOCHS = 10
for LR in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
      save_folder = 'data/tensorboard_info/' +  datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + " LR "+str(LR)
      writer = SummaryWriter(save_folder)
      dataset_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)#use 4-8 workers, number needs to be tuned so schedule gpu/cpu collaborate (gpu should be faster to proccess than to load a new batch)
      optimizer = optim.Adam(model.parameters(), lr=LR)

      train_n_epochs(dataset_loader, model, optimizer, EPOCHS, device, writer, save_folder)
      writer.close()

print("program complete")



#training ideas (excluding arhictecture modification)
#try include psnr score as loss
#2 losses, one is a downscaled image for more wide but blurry, and original for more detailed
#