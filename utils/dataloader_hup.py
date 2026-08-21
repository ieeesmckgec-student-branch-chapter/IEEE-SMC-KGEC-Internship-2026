# dataloader_hup.py  —  HUPAnno dataloader with Dynamic Sizing + Multi-Format Support
# Handles: .png, .jpg, .jpeg, .bmp, .tif, .tiff, .npy
# Auto-scales annotation to lesion size (adapts to Kvasir large polyps vs Liver small lesions)

import os
import re
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import cv2
from torchvision.transforms import InterpolationMode

# ---------------------------------------------------------------------------
# Configuration (Dynamic - scales automatically per image)
# ---------------------------------------------------------------------------

HUP_DP_RATIO  = 0.02
HUP_MIN_VTX   = 5
HUP_MAX_VTX   = 15

# LRP settings
HUP_K_PATCHES = 2
HUP_PATCH_ARC = 0.12
HUP_LRP_DIL   = 0.08   # Relative to lesion radius
HUP_LRP_ERO   = 0.06   # Relative to lesion radius
HUP_MIN_AREA  = 20     # Reduced from 50 for small liver lesions

# Supported formats
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.npy')
MASK_EXTS  = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.npy')


# ---------------------------------------------------------------------------
# Robust File I/O Helpers
# ---------------------------------------------------------------------------

def _stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def _load_npy_image(path):
    """Load .npy image -> PIL RGB. Handles (H,W), (H,W,3), (3,H,W), float[0,1], uint8[0,255]"""
    arr = np.load(path, allow_pickle=True)
    
    # Handle channel-first formats (3, H, W) or (1, H, W)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    
    # Handle grayscale
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[:, :, :3]  # Take first 3 channels if RGBA
    
    # Normalize to uint8
    if arr.dtype == np.bool_:
        arr = arr.astype(np.uint8) * 255
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.max() <= 1.0:
            arr = arr * 255.0
        else:
            # Min-max normalize if range is weird
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    
    return Image.fromarray(arr).convert('RGB')


def _load_npy_mask(path):
    """Load .npy mask -> PIL L (grayscale). Handles binary [0,1], [0,255], bool"""
    arr = np.load(path, allow_pickle=True)
    
    # Squeeze channel dims
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr.squeeze(0)
        elif arr.shape[-1] == 1:
            arr = arr.squeeze(-1)
        else:
            # Multi-channel mask: take max
            arr = arr.max(axis=-1) if arr.shape[-1] < 10 else arr.max(axis=0)
    
    # Force binary 0/255
    if arr.dtype == np.bool_:
        mask = arr.astype(np.uint8) * 255
    else:
        arr = arr.astype(np.float32)
        # Detect if normalized [0,1] or [0,255]
        if arr.max() <= 1.0:
            mask = (arr > 0.5).astype(np.uint8) * 255
        else:
            mask = (arr > 0).astype(np.uint8) * 255
    
    return Image.fromarray(mask, mode='L')


def _load_image_any(path):
    """Universal image loader"""
    if path.lower().endswith('.npy'):
        return _load_npy_image(path)
    return Image.open(path).convert('RGB')


def _load_mask_any(path):
    """Universal mask loader"""
    if path.lower().endswith('.npy'):
        return _load_npy_mask(path)
    return Image.open(path).convert('L')


def _get_size_any(path, is_mask=False):
    """Get (W, H) without fully loading"""
    if path.lower().endswith('.npy'):
        arr = np.load(path, allow_pickle=True, mmap_mode='r')
        if arr.ndim == 2:
            return (arr.shape[1], arr.shape[0])
        elif arr.ndim == 3:
            # Return (W, H) regardless of channel position
            return (arr.shape[1], arr.shape[0]) if arr.shape[0] in (1, 3, 4) else (arr.shape[1], arr.shape[0])
        return (arr.shape[-1], arr.shape[-2])
    else:
        with Image.open(path) as img:
            return img.size


def _pair_images_masks(image_root, gt_root):
    """Smart pairing: tries exact stem match, then numeric match, then ordered"""
    img_files = sorted([f for f in os.listdir(image_root) if f.lower().endswith(IMAGE_EXTS)])
    msk_files = sorted([f for f in os.listdir(gt_root) if f.lower().endswith(MASK_EXTS)])
    
    img_paths = [os.path.join(image_root, f) for f in img_files]
    msk_paths = [os.path.join(gt_root, f) for f in msk_files]
    
    # Try exact stem matching first
    img_map = {_stem(p): p for p in img_paths}
    msk_map = {_stem(p): p for p in msk_paths}
    common = sorted(set(img_map.keys()) & set(msk_map.keys()))
    
    if len(common) > 0:
        return [(img_map[k], msk_map[k]) for k in common]
    
    # Fallback: numeric extraction (img_001 -> mask_001)
    def extract_num(s):
        nums = re.findall(r'\d+', s)
        return nums[0] if nums else s
    
    img_nums = {extract_num(_stem(p)): p for p in img_paths}
    msk_nums = {extract_num(_stem(p)): p for p in msk_paths}
    common_nums = sorted(set(img_nums.keys()) & set(msk_nums.keys()))
    
    if len(common_nums) > 0:
        return [(img_nums[k], msk_nums[k]) for k in common_nums]
    
    # Final fallback: assume ordered
    if len(img_paths) == len(msk_paths):
        return list(zip(img_paths, msk_paths))
    
    raise ValueError(f"Cannot pair {len(img_paths)} images with {len(msk_paths)} masks")


# ---------------------------------------------------------------------------
# Dynamic HUPAnno Generation
# ---------------------------------------------------------------------------

def to_poly_mask(binary_u8, H, W, dp_ratio=HUP_DP_RATIO, min_vtx=HUP_MIN_VTX, max_vtx=HUP_MAX_VTX):
    """Convert binary mask to filled polygon"""
    contours, _ = cv2.findContours(binary_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros((H, W), dtype=np.uint8)
    if not contours:
        return out
    
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < 10:  # Too small
        return out
    
    perimeter = cv2.arcLength(cnt, True)
    if perimeter < 10:
        return out
    
    epsilon = perimeter * dp_ratio
    polygon = cv2.approxPolyDP(cnt, epsilon, True)
    
    # Adjust vertex count
    for _ in range(12):
        if len(polygon) > max_vtx:
            epsilon *= 1.25
            polygon = cv2.approxPolyDP(cnt, epsilon, True)
        else:
            break
    for _ in range(12):
        if len(polygon) < min_vtx:
            epsilon *= 0.75
            polygon = cv2.approxPolyDP(cnt, epsilon, True)
        else:
            break
    
    if len(polygon) >= 3:
        cv2.fillPoly(out, [polygon], 255)
    
    return out


def find_hard_segments(gt_u8, K=HUP_K_PATCHES, patch_arc=HUP_PATCH_ARC):
    """Find K hardest boundary segments by curvature"""
    contours, _ = cv2.findContours(gt_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return [], None
    
    cnt = max(contours, key=cv2.contourArea)
    N = len(cnt)
    if N < 20:
        return [], cnt
    
    pts = cnt.reshape(-1, 2).astype(np.float32)
    scores = np.zeros(N)
    
    for i in range(N):
        p0, p1, p2 = pts[(i-2)%N], pts[i], pts[(i+2)%N]
        v1, v2 = p1-p0, p2-p1
        n1, n2 = np.linalg.norm(v1)+1e-6, np.linalg.norm(v2)+1e-6
        cos_a = np.clip(np.dot(v1/n1, v2/n2), -1.0, 1.0)
        scores[i] = 1.0 - cos_a
    
    scores = np.convolve(scores, np.ones(5)/5.0, mode='same')
    
    half_patch = max(1, int(N * patch_arc / 2))
    selected, suppressed = [], np.zeros(N, dtype=bool)
    
    for _ in range(K):
        masked = scores.copy()
        masked[suppressed] = -1.0
        if masked.max() < 0:
            break
        center = int(np.argmax(masked))
        start, end = (center-half_patch)%N, (center+half_patch)%N
        selected.append((center, start, end))
        for offset in range(-half_patch*2, half_patch*2+1):
            suppressed[(center+offset)%N] = True
    
    return selected, cnt


def generate_hupanno(gt_np, K=HUP_K_PATCHES):
    """
    Generate HUPAnno with DYNAMIC scaling based on lesion size.
    Works for both large polyps (Kvasir) and small lesions (Liver).
    """
    # Robust normalization: force to 0/1 float first
    gt_np = np.asarray(gt_np, dtype=np.float32)
    gt_binary = (gt_np > 0.5).astype(np.uint8) * 255
    H, W = gt_binary.shape
    
    area_px = float(gt_binary.sum()) / 255.0
    
    # Dynamic sizing: if lesion is tiny, use absolute minimums
    # If large, scale proportionally
    if area_px < 100:  # Very small lesion (liver)
        # Use fixed small kernels instead of relative scaling
        dil_ks, ero_ks = 5, 3
        lrp_dil_abs, lrp_ero_abs = 3, 2
    else:
        # Normal relative scaling for larger lesions
        radius = np.sqrt(area_px / np.pi)
        dil_ks = max(5, int(radius * 0.40))
        ero_ks = max(3, int(radius * 0.22))
        lrp_dil_abs = max(3, int(radius * HUP_LRP_DIL))
        lrp_ero_abs = max(3, int(radius * HUP_LRP_ERO))
    
    # Ensure odd kernel sizes
    dil_ks = dil_ks + 1 if dil_ks % 2 == 0 else dil_ks
    ero_ks = ero_ks + 1 if ero_ks % 2 == 0 else ero_ks
    lrp_dil_abs = lrp_dil_abs + 1 if lrp_dil_abs % 2 == 0 else lrp_dil_abs
    lrp_ero_abs = lrp_ero_abs + 1 if lrp_ero_abs % 2 == 0 else lrp_ero_abs
    
    # Global rings
    dilated = cv2.dilate(gt_binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_ks, dil_ks)))
    eroded = cv2.erode(gt_binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ero_ks, ero_ks)))
    
    m_out_u8 = to_poly_mask(dilated, H, W)
    m_in_u8 = to_poly_mask(eroded, H, W)
    
    if m_in_u8.sum() == 0:
        m_in_u8 = gt_binary.copy()
    
    # Ensure containment
    m_in_u8 = cv2.bitwise_and(m_in_u8, m_out_u8)
    
    y_in = (m_in_u8 > 127).astype(np.float32)
    y_out = (m_out_u8 > 127).astype(np.float32)
    omega_delta = np.clip(y_out - y_in, 0.0, 1.0).astype(np.float32)
    
    # LRP patches with dynamic kernels
    lrp_fg, lrp_bg, lrp_uncertain, lrp_mask = _generate_lrp_dynamic(
        gt_binary, H, W, K, lrp_ero_abs, lrp_dil_abs
    )
    
    # 4-class map
    y_c = np.zeros((H, W), dtype=np.int64)
    y_c[omega_delta == 1] = 1
    y_c[lrp_uncertain == 1] = 2
    y_c[lrp_bg == 1] = 0
    y_c[y_in == 1] = 3
    y_c[lrp_fg == 1] = 3
    
    return y_in, y_out, omega_delta, lrp_fg, lrp_bg, lrp_uncertain, lrp_mask, y_c


def _generate_lrp_dynamic(gt_u8, H, W, K, ero_ks, dil_ks):
    """LRP generation with pre-calculated absolute kernel sizes"""
    _empty = np.zeros((H, W), dtype=np.float32)
    area = float(gt_u8.sum()) / 255.0
    
    if area < HUP_MIN_AREA:
        return _empty.copy(), _empty.copy(), _empty.copy(), _empty.copy()
    
    segments, cnt = find_hard_segments(gt_u8, K=K)
    if not segments or cnt is None:
        return _empty.copy(), _empty.copy(), _empty.copy(), _empty.copy()
    
    pts, N = cnt.reshape(-1, 2), len(cnt)
    
    ero_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ero_ks, ero_ks))
    dil_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_ks, dil_ks))
    
    lrp_inner_all = np.zeros((H, W), dtype=np.uint8)
    lrp_outer_all = np.zeros((H, W), dtype=np.uint8)
    lrp_mask_all = np.zeros((H, W), dtype=np.uint8)
    
    for (center, start, end) in segments:
        seg_idx = list(range(start, end+1)) if start <= end else list(range(start, N)) + list(range(0, end+1))
        seg_pts = pts[seg_idx]
        
        pad = dil_ks * 2
        x_min, x_max = max(0, int(seg_pts[:,0].min())-pad), min(W, int(seg_pts[:,0].max())+pad)
        y_min, y_max = max(0, int(seg_pts[:,1].min())-pad), min(H, int(seg_pts[:,1].max())+pad)
        
        patch_gt = gt_u8[y_min:y_max, x_min:x_max]
        if patch_gt.size == 0 or patch_gt.sum() == 0:
            continue
        
        patch_ero = cv2.erode(patch_gt, ero_kernel, iterations=1)
        patch_dil = cv2.dilate(patch_gt, dil_kernel, iterations=1)
        
        ph, pw = patch_gt.shape
        pm_ero = to_poly_mask(patch_ero, ph, pw)
        pm_dil = to_poly_mask(patch_dil, ph, pw)
        
        lrp_inner_all[y_min:y_max, x_min:x_max] = np.maximum(lrp_inner_all[y_min:y_max, x_min:x_max], pm_ero)
        lrp_outer_all[y_min:y_max, x_min:x_max] = np.maximum(lrp_outer_all[y_min:y_max, x_min:x_max], pm_dil)
        lrp_mask_all[y_min:y_max, x_min:x_max] = 1
    
    inner_f = (lrp_inner_all > 127).astype(np.float32)
    outer_f = (lrp_outer_all > 127).astype(np.float32)
    mask_f = lrp_mask_all.astype(np.float32)
    
    return inner_f, mask_f * (1.0 - outer_f), np.clip(outer_f - inner_f, 0.0, 1.0), mask_f


# ---------------------------------------------------------------------------
# Dataset Classes
# ---------------------------------------------------------------------------

class PolypDataset(data.Dataset):
    def __init__(self, image_root, gt_root, trainsize, augmentations=True, split='train', K=HUP_K_PATCHES):
        self.trainsize = trainsize
        self.augmentations = augmentations
        self.split = split
        self.K = K
        self.pairs = _pair_images_masks(image_root, gt_root)
        self.size = len(self.pairs)
        
        # Use NEAREST for masks to preserve binary boundaries
        if self.augmentations in (True, 'True'):
            self.img_transform = transforms.Compose([
                transforms.RandomRotation(90),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((trainsize, trainsize), interpolation=InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            self.gt_transform = transforms.Compose([
                transforms.RandomRotation(90),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((trainsize, trainsize), interpolation=InterpolationMode.NEAREST),
                transforms.ToTensor()
            ])
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize((trainsize, trainsize), interpolation=InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            self.gt_transform = transforms.Compose([
                transforms.Resize((trainsize, trainsize), interpolation=InterpolationMode.NEAREST),
                transforms.ToTensor()
            ])
    
    def __getitem__(self, index):
        img_path, gt_path = self.pairs[index]
        
        image = _load_image_any(img_path)
        gt = _load_mask_any(gt_path)
        
        # Synchronized augmentation
        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        image = self.img_transform(image)
        
        random.seed(seed)
        torch.manual_seed(seed)
        gt = self.gt_transform(gt)
        
        gt_np = gt.squeeze().cpu().numpy()
        # Ensure binary
        gt_np = (gt_np > 0.5).astype(np.float32)
        
        y_in, y_out, omega_delta, lrp_fg, lrp_bg, lrp_uncertain, lrp_mask, y_c = generate_hupanno(gt_np, K=self.K)
        
        return (
            image,
            torch.from_numpy(y_in).float(),
            torch.from_numpy(y_out).float(),
            torch.from_numpy(omega_delta).float(),
            torch.from_numpy(lrp_fg).float(),
            torch.from_numpy(lrp_bg).float(),
            torch.from_numpy(lrp_uncertain).float(),
            torch.from_numpy(lrp_mask).float(),
            torch.from_numpy(y_c).long(),
            gt
        )
    
    def __len__(self):
        return self.size


class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.pairs = _pair_images_masks(image_root, gt_root)
        self.size = len(self.pairs)
        self.index = 0
        
        self.transform = transforms.Compose([
            transforms.Resize((testsize, testsize), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.ToTensor()
    
    def load_data(self):
        if self.index >= self.size:
            self.index = 0
        
        img_path, gt_path = self.pairs[self.index]
        
        image = _load_image_any(img_path)
        image = self.transform(image).unsqueeze(0)
        
        gt = _load_mask_any(gt_path)
        # Keep original size for metrics, or resize? Resize to testsize for consistency
        gt = transforms.Resize((self.testsize, self.testsize), interpolation=InterpolationMode.NEAREST)(gt)
        
        name = os.path.basename(img_path)
        name = os.path.splitext(name)[0] + '.png'
        
        self.index += 1
        return image, gt, name
    
    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=True, augmentation=True, split='train', K=HUP_K_PATCHES):
    dataset = PolypDataset(image_root, gt_root, trainsize, augmentations=augmentation, split=split, K=K)
    return data.DataLoader(dataset, batch_size=batchsize, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)