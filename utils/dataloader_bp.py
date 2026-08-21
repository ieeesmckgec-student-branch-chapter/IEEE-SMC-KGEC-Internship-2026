# dataloader_bp.py — Original EAUWSeg BPAnno with .npy support
# Strictly follows the BPAnno methodology: 3-class CCG, global rings only.

import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import torch
import cv2


# ---------------------------------------------------------------------------
# BPAnno Configuration (Exact EAUWSeg defaults)
# ---------------------------------------------------------------------------
BPANNO_DIL_SCALE    = 0.40
BPANNO_ERO_SCALE    = 0.22
BPANNO_DP_RATIO     = 0.02
BPANNO_MIN_VERTICES = 5
BPANNO_MAX_VERTICES = 15

# Supported formats
IMG_EXTS  = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.npy')
MASK_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.npy')


# ---------------------------------------------------------------------------
# Universal Loaders (Handles PNG/JPG and NPY)
# ---------------------------------------------------------------------------

def _load_any(path, is_mask=False):
    """
    Robust loader for both standard images and .npy arrays.
    Forces masks to strictly binary 0/255 to prevent HD95 bugs.
    """
    ext = os.path.splitext(path)[1].lower()
    
    if ext == '.npy':
        arr = np.load(path)
        
        # Normalize shape to (H, W) or (H, W, C)
        if arr.ndim == 3:
            if arr.shape[0] in (1, 3):           # CHW -> HWC
                arr = np.transpose(arr, (1, 2, 0))
            if arr.shape[-1] == 1:               # (H,W,1) -> (H,W)
                arr = arr.squeeze(-1)
        
        if is_mask:
            # Force strict binary mask (critical for HD95)
            if arr.dtype == bool or arr.max() <= 1.0:
                arr = (arr > 0.5).astype(np.uint8) * 255
            else:
                arr = (arr > 0).astype(np.uint8) * 255
            return Image.fromarray(arr, mode='L')
        else:
            # Image: convert to RGB uint8
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            if arr.dtype != np.uint8:
                if arr.max() <= 1.0:
                    arr = (arr * 255).clip(0, 255).astype(np.uint8)
                else:
                    arr = arr.clip(0, 255).astype(np.uint8)
            return Image.fromarray(arr).convert('RGB')
    
    else:
        # Standard image file
        mode = 'L' if is_mask else 'RGB'
        with open(path, 'rb') as f:
            img = Image.open(f).convert(mode)
            if is_mask:
                # Force binary for standard images too, just in case
                arr = np.array(img)
                arr = (arr > 127).astype(np.uint8) * 255
                return Image.fromarray(arr, mode='L')
            return img


def _get_size_any(path, is_mask=False):
    """Get (W, H) without loading full array when possible."""
    if path.lower().endswith('.npy'):
        arr = np.load(path)
        if arr.ndim == 2:
            return (arr.shape[1], arr.shape[0])
        if arr.ndim == 3:
            if arr.shape[0] in (1, 3):
                return (arr.shape[2], arr.shape[1])
            return (arr.shape[1], arr.shape[0])
    else:
        return Image.open(path).size


# ---------------------------------------------------------------------------
# BPAnno Generation (Exact EAUWSeg Implementation)
# ---------------------------------------------------------------------------

def generate_bpanno(gt_np, dp_epsilon_ratio=BPANNO_DP_RATIO):
    """
    Generate BPAnno polygon masks from binary GT mask.
    Creates polygon shapes with straight edges (5-15 vertices).
    """
    gt_u8 = (gt_np * 255).astype(np.uint8)
    H, W  = gt_np.shape

    # Adaptive kernel sizes (Standard EAUWSeg logic)
    area = float(gt_u8.sum()) / 255.0

    if area > 10:
        radius = np.sqrt(area / np.pi)
        dil_ks = max(7, int(radius * BPANNO_DIL_SCALE))
        ero_ks = max(5, int(radius * BPANNO_ERO_SCALE))
    else:
        dil_ks = 9
        ero_ks = 7

    if dil_ks % 2 == 0: dil_ks += 1
    if ero_ks % 2 == 0: ero_ks += 1
    
    dil_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_ks, dil_ks))
    ero_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ero_ks, ero_ks))
    
    dilated = cv2.dilate(gt_u8, dil_kernel, iterations=1)
    eroded  = cv2.erode(gt_u8,  ero_kernel, iterations=1)

    def to_poly_mask(binary_u8):
        contours, _ = cv2.findContours(
            binary_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        out = np.zeros((H, W), dtype=np.uint8)
        if not contours:
            return out

        contour   = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, closed=True)
        if perimeter < 10:
            return out

        epsilon = perimeter * dp_epsilon_ratio
        polygon = cv2.approxPolyDP(contour, epsilon, closed=True)

        attempts = 0
        while len(polygon) > BPANNO_MAX_VERTICES and attempts < 12:
            epsilon *= 1.25
            polygon  = cv2.approxPolyDP(contour, epsilon, closed=True)
            attempts += 1

        attempts = 0
        while len(polygon) < BPANNO_MIN_VERTICES and attempts < 12:
            epsilon *= 0.75
            polygon  = cv2.approxPolyDP(contour, epsilon, closed=True)
            attempts += 1

        if len(polygon) >= 3:
            cv2.fillPoly(out, [polygon], 255)
        return out

    y_en_u8 = to_poly_mask(dilated)
    y_in_u8 = to_poly_mask(eroded)

    if y_in_u8.sum() == 0:
        y_in_u8 = gt_u8.copy()

    y_en = (y_en_u8 > 127).astype(np.float32)
    y_in = (y_in_u8 > 127).astype(np.float32)

    y_in = y_in * y_en
    omega_delta = np.clip(y_en - y_in, 0, 1).astype(np.float32)

    # 3-class CCG label map (EAUWSeg standard)
    y_c = np.zeros_like(y_in, dtype=np.int64)
    y_c[y_en == 1] = 1    # inside envelope -> uncertain
    y_c[y_in == 1] = 2    # inside inscribed -> certain fg
    # pixels outside y_en stay 0 -> certain bg
    
    return y_in, y_en, omega_delta, y_c


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PolypDataset(data.Dataset):

    def __init__(self, image_root, gt_root, trainsize,
                 augmentations, split='train', color_image=True):
        self.trainsize    = trainsize
        self.augmentations= augmentations
        self.split        = split
        self.color_image  = color_image

        self.images = sorted([
            image_root + f for f in os.listdir(image_root)
            if f.lower().endswith(IMG_EXTS)
        ])
        self.gts = sorted([
            gt_root + f for f in os.listdir(gt_root)
            if f.lower().endswith(MASK_EXTS)
        ])

        self.filter_files()
        self.size = len(self.images)

        if self.augmentations == 'True' or self.augmentations is True:
            print('Using RandomRotation, RandomFlip')
            self.img_transform = transforms.Compose([
                transforms.RandomRotation(90),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
            self.gt_transform = transforms.Compose([
                transforms.RandomRotation(90),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()
            ])
        else:
            print('no augmentation')
            self.img_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
            self.gt_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()
            ])

    def __getitem__(self, index):
        image = _load_any(self.images[index], is_mask=False)
        gt    = _load_any(self.gts[index], is_mask=True)

        # Synchronized augmentation for image and GT
        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        image = self.img_transform(image)
        
        random.seed(seed)
        torch.manual_seed(seed)
        gt = self.gt_transform(gt)

        gt_np = gt.squeeze().cpu().numpy()
        gt_np = (gt_np > 0.5).astype(np.float32)
        
        y_in, y_en, omega_delta, y_c = generate_bpanno(gt_np)
        
        y_in_t  = torch.from_numpy(y_in).float()
        y_en_t  = torch.from_numpy(y_en).float()
        omega_delta_t = torch.from_numpy(omega_delta).float()
        y_c_t   = torch.from_numpy(y_c).long()
        
        return image, y_in_t, y_en_t, omega_delta_t, y_c_t, gt

    def filter_files(self):
        images, gts = [], []
        for img_path, gt_path in zip(self.images, self.gts):
            try:
                img_size = _get_size_any(img_path, is_mask=False)
                gt_size  = _get_size_any(gt_path, is_mask=True)
                if img_size == gt_size:
                    images.append(img_path)
                    gts.append(gt_path)
            except Exception as e:
                print(f'[filter_files] Skipping pair due to error: {e}')
        self.images = images
        self.gts    = gts

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, batchsize, trainsize,
               shuffle=False, num_workers=4,
               pin_memory=True, augmentation=False,
               split='train', color_image=True):

    dataset = PolypDataset(
        image_root, gt_root, trainsize,
        augmentation, split, color_image
    )
    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return data_loader


# ---------------------------------------------------------------------------
# Test dataset
# ---------------------------------------------------------------------------

class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = sorted([
            image_root + f for f in os.listdir(image_root)
            if f.lower().endswith(IMG_EXTS)
        ])
        self.gts = sorted([
            gt_root + f for f in os.listdir(gt_root)
            if f.lower().endswith(MASK_EXTS)
        ])
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.ToTensor()
        self.size  = len(self.images)
        self.index = 0

    def load_data(self):
        image = _load_any(self.images[self.index], is_mask=False)
        image = self.transform(image).unsqueeze(0)
        gt    = _load_any(self.gts[self.index], is_mask=True)
        name  = os.path.basename(self.images[self.index])
        name  = os.path.splitext(name)[0] + '.png'
        self.index += 1
        return image, gt, name