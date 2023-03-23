import sys
import torch
from torchinfo import summary
sys.path.append('utils')
from utils import get_dataloader
from layers import DTCNN, SDTCNN

LAYER_CHANNELS = (3,64,128,256,512,1024)
BATCH_SIZE = 3
SKIP_LAYERS = [0,1,2,3,4]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root_dir = '/home/isac/data/viton_hd'
model = SDTCNN(layer_channels=LAYER_CHANNELS, skip_layers=SKIP_LAYERS).to(device)
dataloader = get_dataloader(root_dir=root_dir,
                                usage='use_model_use_mask',
                                bg_mode="train",
                                validation_length=500,
                                BATCH_SIZE=3,
                                dilation=0,
                                erosion=0,
                                get_color_segmentation=False
)

for ix, x in enumerate(dataloader):
    photo = x['photo'].to(device)
    segmentation = x['segmentation'].to(device)
    output = model(photo,segmentation)
    summary(model, input_size=[photo.shape,segmentation.shape])
    break