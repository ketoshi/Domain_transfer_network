from utils.dataset import * #don't need sys .path beacuse this is root folder
from utils.utils import *
from utils.layers import *
import datetime

LAYER_CHANNELS = (3,32,64,128,256)
BATCH_SIZE = 3
EPOCHS = 1
LR = 1e-3
SKIP_LAYERS = [0,1,2,3]   #changed network, but before 1e-5 <LR <1e-2  was good  1e-1 was bad maybe anothe run to see that
functions = [train_n_epochs_twice]

for i_function in functions:
      SAVE_FOLDER = 'data/tensorboard_info/' +  datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "function:" + str(i_function)
      i_function(layer_channels=LAYER_CHANNELS, skip_layers=SKIP_LAYERS, lr=LR, batch_size = BATCH_SIZE, epochs=EPOCHS, save_folder=SAVE_FOLDER, update_dataset_per_epoch=True)


#Recreate dataset with new Rescale function, make sure the image looks smooth!
#when training add target, inut and generated image for easy look of improvement (to Writer)
#can use more metric scores and check the images if they seem to work as intended. SSIM, KID, PSNR, PPL,
#use both vitonhd and f550k for metric evaluation












# 
# 

# )
#try include psnr score as loss
#2 losses, one is a downscaled image for more wide but blurry, and original for more detailed


#start tmux before running training! always
#17:00 och frammåt nästan alla dagar fredrik ledig

#Code as is now says it should take toughly 1000 mb .... but it increases and increases to almost 10 (could there be something similar to a memory-leak?)
#could i try debug this if so and see what might be the problem?