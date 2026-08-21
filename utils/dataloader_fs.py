import os
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

import numpy as np
import random


IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.npy')
MASK_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.npy')



def load_image(path):
    ext = os.path.splitext(path)[1].lower()

    if ext != ".npy":
        with open(path, "rb") as f:
            return Image.open(f).convert("RGB")

    arr = np.load(path)

    # CHW -> HWC
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = arr.transpose(1, 2, 0)

    # grayscale
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)

    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    arr = arr.astype(np.float32)

    if arr.max() <= 1.0:
        arr *= 255.0

    arr = np.clip(arr, 0, 255).astype(np.uint8)

    return Image.fromarray(arr).convert("RGB")


def load_mask(path):
    ext = os.path.splitext(path)[1].lower()

    if ext != ".npy":
        with open(path, "rb") as f:
            img = Image.open(f).convert("L")
            arr = np.array(img)
            arr = (arr > 127).astype(np.uint8) * 255
            return Image.fromarray(arr)

    arr = np.load(path)

    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr.squeeze(0)
        elif arr.shape[-1] == 1:
            arr = arr.squeeze(-1)
        else:
            arr = arr.max(axis=-1)

    arr = arr.astype(np.float32)

    if arr.max() <= 1.0:
        arr = arr > 0.5
    else:
        arr = arr > 0

    arr = arr.astype(np.uint8) * 255

    return Image.fromarray(arr)


def get_size(path, is_mask=False):
    ext = os.path.splitext(path)[1].lower()

    if ext != ".npy":
        return Image.open(path).size

    arr = np.load(path)

    if arr.ndim == 2:
        return (arr.shape[1], arr.shape[0])

    if arr.ndim == 3:

        if arr.shape[0] in (1, 3):
            return (arr.shape[2], arr.shape[1])

        return (arr.shape[1], arr.shape[0])

    raise RuntimeError(path)


# ------------------------------------------------------------
# Pairing
# ------------------------------------------------------------

def build_pairs(image_root, mask_root):

    images = sorted([
        os.path.join(image_root, f)
        for f in os.listdir(image_root)
        if f.lower().endswith(IMG_EXTS)
    ])

    masks = sorted([
        os.path.join(mask_root, f)
        for f in os.listdir(mask_root)
        if f.lower().endswith(MASK_EXTS)
    ])

    img_dict = {
        os.path.splitext(os.path.basename(x))[0]: x
        for x in images
    }

    mask_dict = {
        os.path.splitext(os.path.basename(x))[0]: x
        for x in masks
    }

    common = sorted(set(img_dict.keys()) & set(mask_dict.keys()))

    if len(common) > 0:
        return [(img_dict[k], mask_dict[k]) for k in common]

    if len(images) != len(masks):
        raise RuntimeError(
            f"Image-mask mismatch ({len(images)} vs {len(masks)})"
        )

    return list(zip(images, masks))


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

class FullDataset(data.Dataset):

    def __init__(self,
                 image_root,
                 gt_root,
                 trainsize,
                 augmentations=True):

        self.trainsize = trainsize
        self.augmentations = augmentations

        self.pairs = build_pairs(image_root, gt_root)

        self.filter_files()

        self.size = len(self.pairs)

        if augmentations in ("True", True):

            print("Using augmentation")

            self.img_transform = transforms.Compose([
                transforms.RandomRotation(90),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomHorizontalFlip(0.5),
                transforms.Resize((trainsize, trainsize)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485,0.456,0.406],
                    [0.229,0.224,0.225]
                )
            ])

            self.gt_transform = transforms.Compose([
                transforms.RandomRotation(90),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomHorizontalFlip(0.5),
                transforms.Resize(
                    (trainsize, trainsize),
                    interpolation=InterpolationMode.NEAREST
                ),
                transforms.ToTensor()
            ])

        else:

            print("No augmentation")

            self.img_transform = transforms.Compose([
                transforms.Resize((trainsize,trainsize)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485,0.456,0.406],
                    [0.229,0.224,0.225]
                )
            ])

            self.gt_transform = transforms.Compose([
                transforms.Resize(
                    (trainsize,trainsize),
                    interpolation=InterpolationMode.NEAREST
                ),
                transforms.ToTensor()
            ])

    def filter_files(self):

        valid = []

        for img_path, mask_path in self.pairs:

            try:

                if get_size(img_path)==get_size(mask_path,True):
                    valid.append((img_path,mask_path))

            except:
                pass

        self.pairs = valid

    def __getitem__(self,index):

        img_path, mask_path = self.pairs[index]

        image = load_image(img_path)
        gt = load_mask(mask_path)

        seed = np.random.randint(2147483647)

        random.seed(seed)
        torch.manual_seed(seed)
        image = self.img_transform(image)

        random.seed(seed)
        torch.manual_seed(seed)
        gt = self.gt_transform(gt)

        return image, gt

    def __len__(self):
        return self.size


# ------------------------------------------------------------
# Loader
# ------------------------------------------------------------

def get_loader(
        image_root,
        gt_root,
        batchsize,
        trainsize,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        augmentation=False):

    dataset = FullDataset(
        image_root,
        gt_root,
        trainsize,
        augmentation
    )

    return data.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )


# ------------------------------------------------------------
# Test dataset
# ------------------------------------------------------------

class test_dataset:

    def __init__(self,image_root,gt_root,testsize):

        self.testsize=testsize
        self.pairs=build_pairs(image_root,gt_root)

        self.transform=transforms.Compose([
            transforms.Resize((testsize,testsize)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485,0.456,0.406],
                [0.229,0.224,0.225]
            )
        ])

        self.index=0
        self.size=len(self.pairs)

    def load_data(self):

        img_path,gt_path=self.pairs[self.index]

        image=load_image(img_path)
        image=self.transform(image).unsqueeze(0)

        gt=load_mask(gt_path)

        name=os.path.splitext(os.path.basename(img_path))[0]+".png"

        self.index+=1

        return image,gt,name