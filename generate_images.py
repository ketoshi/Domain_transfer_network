import argparse
import os
import torch
import torchvision
from data.dataset import * 
from data.utils import *
from model.layers import *

#argparse
parser = argparse.ArgumentParser()
parser.add_argument('-o','--own_dataset', help='use flag if you want to use own dataset of images (in data/use_model_input)', action="store_true")
parser.add_argument('-m','--model_path', help='use flag to specify model_path, else model/saved_models/pre_trained_model.pth will be used', default="model/saved_models/pre_trained_model.pth")
args = parser.parse_args()
args_dict = vars(args)

#param
if args.own_dataset:
    root_dir = 'data/use_model_input'
else:
    root_dir = '/home/isac/data/viton_hd' #TODO change to real path later

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_folder = 'data/use_model_output'
BATCH_SIZE = 4

#load model
model_info_path = args_dict['model_path'][0:-3]+"json"
with open(model_info_path) as json_file:
    model_info = json.load(json_file)
layer_channels = model_info["layer_channels"]
skip_layers = model_info["skip_layers"]
model = DTCNN(layer_channels=layer_channels, skip_layers=skip_layers).to(device) 
model.load_state_dict( torch.load( args_dict['model_path'] ) )
model.eval()

#fix/create/load dataset
if args.own_dataset:
    photo_dir = 'data/use_model_input/photo'
    segmentation_dir = 'data/use_model_input/segmentation'

    csv_file_path = 'data/use_model_input/dataset.csv'
    photo_csv_path = 'data/use_model_input/dataset_photo.csv'
    segmentation_csv_path = 'data/use_model_input/dataset_segmentation.csv'
    
    write_csv(photo_csv_path,        photo_dir,        description='image', overwrite=True)
    write_csv(segmentation_csv_path, segmentation_dir, description='segmentation', overwrite=True) 
    combine_csv(csv_file_path, file_path1 = photo_csv_path, file_path2 = segmentation_csv_path)
else:
    csv_file_path = os.path.join(root_dir,'dataset.csv')
trms = torchvision.transforms.Compose([RandomCrop((512,384)), ApplyMask(do_not_apply_mask=True), \
      ToTensor(), NormalizeMult() ]) 
dataset = domain_transfer_dataset(csv_file_path, root_dir, transform=trms)
if not args.own_dataset:
    train_set, val_set = torch.utils.data.random_split(dataset, [len(dataset)-500,500], generator=torch.Generator().manual_seed(0))
    dataset = val_set
dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

#generate images
for ix, x in enumerate(tqdm(dataset_loader)):
    input_image, segmentation_image, target_image = get_loader_vals(x, device)
    generated_images = model(input_image, segmentation_image)
    save_images(save_folder, generated_images, as_batch=True)
 
print("program complete, images saved in test_results")

