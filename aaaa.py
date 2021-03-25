from torch.utils.data import Dataset
import PIL.Image as Image
import os, sys

image_root='data2/train/images/'
mask_root='data2/train/masks/'
list = os.listdir(image_root)

for fileName in list:
    img = os.path.join(image_root, fileName)
    list2 = os.path.splitext(mask_root + fileName)[0]
    mask = os.path.join(mask_root, fileName)
    #mask = os.path.join(mask_root + list2 + '_mask.tif')
    print(fileName)

#name=os.path.basename('data2/train')
#print(list)
#print(name)