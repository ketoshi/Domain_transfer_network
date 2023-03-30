from layers import *
import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


save_folder = '/home/isac/Domain_transfer_network/data/test_results'
save_path1= os.path.join(save_folder,"res1.png")
save_path2= os.path.join(save_folder,"res2.png")
save_path3= os.path.join(save_folder,"res3.png")
save_path4= os.path.join(save_folder,"res4.png")
st = "/home/isac/data/f550k/photo/0.jpg"
st2 = "/home/isac/data/f550k/photo/1.jpg"
'''

image = io.imread(st)
image2 = io.imread(st2)
target = io.imread(st2)

transform = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=1, contrast_limit=1, p=1.0),
])
#transform = A.HorizontalFlip(p=0.9)
images = np.array(image)
#images = [np.array(image),np.array(image2)]
augmented_image = transform(img=images)
augmented_image = augmented_image['image'][0]     
plt.imsave(save_path, augmented_image)


'''

# Load two images
image1 = Image.open(st)
image2 = Image.open(st2)

image1 = np.array(image1)
image2 = np.array(image2)

image1 = io.imread(st)
image2 = io.imread(st2)
# Define augmentation pipeline
transform = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=30),
    A.RandomBrightnessContrast()
    ],
    additional_targets={'image0': 'image',}
)

# Apply the same transformation to both images
augmented = transform(image=image1, image0=image2)

# Retrieve the augmented images
augmented_image1 = augmented["image"]
augmented_image2 = augmented["image0"]


# Save the augmented images
plt.imsave(save_path1, augmented_image1)
plt.imsave(save_path2, image1)
plt.imsave(save_path3, augmented_image2)
plt.imsave(save_path4, image2)






