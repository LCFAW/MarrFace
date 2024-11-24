import os
import sys
import torchvision.transforms as tfs
import torch.utils.data
import numpy as np
from PIL import Image
from pathlib import Path
from networks.dinoutils import *

def get_data_loaders(cfgs):

    batch_size = cfgs.get('batch_size', 64)
    num_workers = cfgs.get('num_workers', 4)
    image_size = cfgs.get('image_size', 256)
    crop = cfgs.get('crop', None)
    device = cfgs.get('device', 'cpu')

    run_train = cfgs.get('run_train', False)
    train_data_dir = cfgs.get('train_data_dir', './data')
    train_data_file = cfgs.get('train_data_file', './data')
    val_data_dir = cfgs.get('val_data_dir', './data')
    val_data_file = cfgs.get('val_data_file', './data')
    
    run_test = cfgs.get('run_test', False)
    test_data_dir = cfgs.get('test_data_dir', './data')
    test_data_file = cfgs.get('test_data_file', './data')

    train_loader = val_loader = test_loader = None
   
    if run_train:
        print(f"Loading training data from {train_data_dir}")
        train_loader = get_image_loader(data_dir=train_data_dir, file_dir = train_data_file,is_validation=False, batch_size=batch_size, image_size=image_size, crop=crop)
        print(f"Loading validation data from {val_data_dir}")
        val_loader = get_image_loader(data_dir=val_data_dir, file_dir = val_data_file,is_validation=False, batch_size=batch_size, image_size=image_size, crop=crop)
    if run_test:
        test_loader = get_image_loader(data_dir=test_data_dir, file_dir = test_data_file, is_validation=True, batch_size=batch_size, image_size=image_size, crop=crop)

    return train_loader, val_loader, test_loader


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', 'webp')
def is_image_file(filename):
    return filename.lower().endswith(IMG_EXTENSIONS)


## simple image dataset ##
def make_dataset(dir):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                fpath = os.path.join(root, fname)
                images.append(fpath)
    return images


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, file_dir, image_size=256, crop=None, is_validation=False):
        super(ImageDataset, self).__init__()
        self.root = data_dir
        self.filelist = []
        for di in range(len(data_dir)):
            lines = Path(file_dir[di]).read_text().strip().split('\n')
            for line in lines:
                self.filelist.append((line, di))
            
        self.size = len(self.filelist)
        # self.mask_size = len(self.mask_paths)
        self.image_size = image_size
        self.crop = crop
        self.is_validation = is_validation
        self.p =1

        # for k in range(self.size):
        #     file_name = self.paths[k]
        #     im_name = file_name[file_name.rfind('/')+1:file_name.rfind('.')]
        #     mask_name =  mask_dir+'/'+im_name+'.png'
        #     if mask_name not in self.mask_paths:
        #         print(mask_name)
    def ori_transform(self, img, hflip=False):
        if self.crop is not None:
            if isinstance(self.crop, int):
                img = tfs.CenterCrop(self.crop)(img)
            else:
                assert len(self.crop) == 4, 'Crop size must be an integer for center crop, or a list of 4 integers (y0,x0,h,w)'
                img = tfs.functional.crop(img, *self.crop)
        img = tfs.functional.resize(img, (128, 128))
        if hflip:
            img = tfs.functional.hflip(img)
        return tfs.functional.to_tensor(img)


    def transform(self, img, hflip=False):
        if self.crop is not None:
            if isinstance(self.crop, int):
                img = tfs.CenterCrop(self.crop)(img)
            else:
                assert len(self.crop) == 4, 'Crop size must be an integer for center crop, or a list of 4 integers (y0,x0,h,w)'
                img = tfs.functional.crop(img, *self.crop)
        img = tfs.functional.resize(img, (self.image_size, self.image_size))
        

        # if self.p % 2 ==0:
        #     img = tfs.functional.hflip(img)
        return tfs.functional.to_tensor(img)
    
    # def transform_dino(self,img):
    #     return self.TransformDino(img)

    def __getitem__(self, index):

        filename, di = self.filelist[index]
        fpath = os.path.join(self.root[di], filename)
        # mpath = self.mask_paths[index % self.size]
      
        img_ori = Image.open(fpath).convert('RGB')
        # mask = Image.open(mpath).convert('RGB')
        img = self.transform(img_ori)
        self.p = self.p+1
        # if self.p % 3 < 2:
        #     img = img.flip(2)
        # if self.p % 2 == 0:
        #     img = img.flip(2)
        #     im_name = 'p'+fpath[fpath.rfind('/')+1:fpath.rfind('.')]
        # else:
        im_name = fpath[fpath.rfind('/')+1:fpath.rfind('.')]


        # hflip = not self.is_validation and np.random.rand()>0.5
        # return self.transform(img), self.transform(mask), im_name, self.ori_transform(img)
        return img, im_name #self.ori_transform(img)#, im_name#, self.transform(mask)

    def __len__(self):
        return self.size

    def name(self):
        return 'ImageDataset'


def get_image_loader(data_dir, file_dir, is_validation=False,
    batch_size=256, num_workers=4, image_size=256, crop=None):

    dataset = ImageDataset(data_dir, file_dir, image_size=image_size, crop=crop, is_validation=is_validation)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        # pin_memory=True,
    )
    return loader


## paired AB image dataset ##
def make_paied_dataset(dir, AB_dnames=None, AB_fnames=None):
    A_dname, B_dname = AB_dnames or ('A', 'B')
    dir_A = os.path.join(dir, A_dname)
    dir_B = os.path.join(dir, B_dname)
    assert os.path.isdir(dir_A), '%s is not a valid directory' % dir_A
    assert os.path.isdir(dir_B), '%s is not a valid directory' % dir_B

    images = []
    for root_A, _, fnames_A in sorted(os.walk(dir_A)):
        for fname_A in sorted(fnames_A):
            if is_image_file(fname_A):
                path_A = os.path.join(root_A, fname_A)
                root_B = root_A.replace(dir_A, dir_B, 1)
                if AB_fnames is not None:
                    fname_B = fname_A.replace(*AB_fnames)
                else:
                    fname_B = fname_A
                path_B = os.path.join(root_B, fname_B)
                if os.path.isfile(path_B):
                    images.append((path_A, path_B))
    return images


class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, image_size=256, crop=None, is_validation=False, AB_dnames=None, AB_fnames=None):
        super(PairedDataset, self).__init__()
        self.root = data_dir
        self.paths = make_paied_dataset(data_dir, AB_dnames=AB_dnames, AB_fnames=AB_fnames)
        self.size = len(self.paths)
        self.image_size = image_size
        self.crop = crop
        self.is_validation = is_validation

    def transform(self, img, hflip=False):
        if self.crop is not None:
            if isinstance(self.crop, int):
                img = tfs.CenterCrop(self.crop)(img)
            else:
                assert len(self.crop) == 4, 'Crop size must be an integer for center crop, or a list of 4 integers (y0,x0,h,w)'
                img = tfs.functional.crop(img, *self.crop)
        img = tfs.functional.resize(img, (self.image_size, self.image_size))
        if hflip:
            img = tfs.functional.hflip(img)
        return tfs.functional.to_tensor(img)

    def __getitem__(self, index):
        path_A, path_B = self.paths[index % self.size]
        img_A = Image.open(path_A).convert('RGB')
        img_B = Image.open(path_B).convert('RGB')
        hflip = not self.is_validation and np.random.rand()>0.5
        return self.transform(img_A, hflip=hflip), self.transform(img_B, hflip=hflip)

    def __len__(self):
        return self.size

    def name(self):
        return 'PairedDataset'


def get_paired_image_loader(data_dir, is_validation=False,
    batch_size=256, num_workers=4, image_size=256, crop=None, AB_dnames=None, AB_fnames=None):

    dataset = PairedDataset(data_dir, image_size=image_size, crop=crop, \
        is_validation=is_validation, AB_dnames=AB_dnames, AB_fnames=AB_fnames)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not is_validation,
        num_workers=num_workers,
        pin_memory=True
    )
    return loader
