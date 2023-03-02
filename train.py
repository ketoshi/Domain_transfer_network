from data.dataset import * #don't need sys .path beacuse this is root folder
from data.utils import *
from model.layers import *
import datetime

LAYER_CHANNELS = (3,45,128,256,512)
BATCH_SIZE = 4
EPOCHS = 10
LR = 1e-3
SKIP_LAYERS_arr = [[-1], [0], [1], [2], [3]]
#changed network, but before 1e-5 <LR <1e-2  was good  1e-1 was bad

for SKIP_LAYERS in SKIP_LAYERS_arr:
      SAVE_FOLDER = 'data/tensorboard_info/' +  datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + " LR "+str(LR)
      train_n_epochs(layer_channels=LAYER_CHANNELS, skip_layers=SKIP_LAYERS, lr=LR, batch_size = BATCH_SIZE, epochs=EPOCHS, save_folder=SAVE_FOLDER, update_dataset_per_epoch=True)
      break

#training ideas (excluding arhictecture modification)
#try include psnr score as loss
#2 losses, one is a downscaled image for more wide but blurry, and original for more detailed


#start tmux before running training! always
#17:00 och frammåt nästan alla dagar fredrik ledig