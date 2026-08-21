import os
import time
import argparse
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from lib.networks_fs import EMCADNet
from utils.dataloader_fs import get_loader
from utils.utils import clip_gradient, AvgMeter


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

def set_seed(seed=2024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------
# Dataset path helpers
# ------------------------------------------------------------

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


def count_files(root):
    if not os.path.isdir(root):
        return 0
    return len([
        f for f in os.listdir(root)
        if f.lower().endswith(IMG_EXTS)
    ])


def get_data_roots(data_path, split_name='Data', verbose=False):
    image_root = os.path.join(data_path, 'images')
    mask_root = os.path.join(data_path, 'masks')

    if not os.path.isdir(image_root):
        raise FileNotFoundError(
            f'{split_name} image folder not found:\n'
            f'  {image_root}\n\n'
            f'Expected structure:\n'
            f'  {data_path}/images/\n'
            f'  {data_path}/masks/'
        )

    if not os.path.isdir(mask_root):
        raise FileNotFoundError(
            f'{split_name} mask folder not found:\n'
            f'  {mask_root}\n\n'
            f'Expected structure:\n'
            f'  {data_path}/images/\n'
            f'  {data_path}/masks/'
        )

    if verbose:
        print(f'{split_name} path : {data_path}')
        print(f'  Images     : {count_files(image_root)}')
        print(f'  Masks      : {count_files(mask_root)}')

    return image_root + '/', mask_root + '/'


# ------------------------------------------------------------
# Model forward helper
# ------------------------------------------------------------

def model_forward(model, x, mode='train'):
    """
    Supports both versions:
    1. EMCADNet.forward(x, mode='train'/'test')
    2. EMCADNet.forward(x)
    """
    try:
        return model(x, mode=mode)
    except TypeError as e:
        if "unexpected keyword argument 'mode'" in str(e):
            return model(x)
        raise e


def to_list(preds):
    if isinstance(preds, (list, tuple)):
        return list(preds)
    return [preds]


def resize_logits(logits, size):
    if logits.shape[2:] == size:
        return logits

    return F.interpolate(
        logits,
        size=size,
        mode='bilinear',
        align_corners=False
    )


def ensemble_logits(preds, target_size):
    """
    Makes validation robust to output order.

    If model returns multiple outputs, average all logits after resizing.
    """
    pred_list = to_list(preds)
    resized = [resize_logits(p, target_size) for p in pred_list]
    return torch.stack(resized, dim=0).mean(dim=0)


# ------------------------------------------------------------
# Mask helper
# ------------------------------------------------------------

def prepare_mask(mask):
    mask = mask.float()

    if mask.dim() == 3:
        mask = mask.unsqueeze(1)

    if mask.dim() == 4 and mask.size(1) != 1:
        mask = mask[:, :1]

    if mask.max().item() > 1.0:
        mask = mask / 255.0

    return mask.clamp(0.0, 1.0)


# ------------------------------------------------------------
# Loss
# Official-style structure loss: weighted BCE + weighted IoU
# ------------------------------------------------------------

def structure_loss_full(pred, gt):
    """
    pred: logits, shape B x 1 x H x W
    gt  : binary mask, shape B x 1 x H x W
    """
    if gt.shape[2:] != pred.shape[2:]:
        gt = F.interpolate(
            gt,
            size=pred.shape[2:],
            mode='nearest'
        )

    weit = 1 + 5 * torch.abs(
        F.avg_pool2d(gt, kernel_size=31, stride=1, padding=15) - gt
    )

    wbce = F.binary_cross_entropy_with_logits(
        pred,
        gt,
        reduction='none'
    )
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    wbce = wbce.mean()

    pred_prob = torch.sigmoid(pred)

    inter = ((pred_prob * gt) * weit).sum(dim=(2, 3))
    union = ((pred_prob + gt) * weit).sum(dim=(2, 3))

    wiou = 1 - (inter + 1) / (union - inter + 1)
    wiou = wiou.mean()

    return wbce + wiou


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

def dice_coefficient(predicted, labels):
    smooth = 1e-6

    predicted_flat = predicted.contiguous().view(-1)
    labels_flat = labels.contiguous().view(-1)

    intersection = (predicted_flat * labels_flat).sum()
    total = predicted_flat.sum() + labels_flat.sum()

    return (2.0 * intersection + smooth) / (total + smooth)


def iou_score(predicted, labels):
    smooth = 1e-6

    predicted_flat = predicted.contiguous().view(-1)
    labels_flat = labels.contiguous().view(-1)

    intersection = (predicted_flat * labels_flat).sum()
    union = predicted_flat.sum() + labels_flat.sum() - intersection

    return (intersection + smooth) / (union + smooth)


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

def evaluate(model, data_path, opt, device, split_name='Val'):
    image_root, gt_root = get_data_roots(
        data_path,
        split_name=split_name,
        verbose=False
    )

    loader = get_loader(
        image_root=image_root,
        gt_root=gt_root,
        batchsize=opt.test_batchsize,
        trainsize=opt.img_size,
        shuffle=False,
        augmentation=False
    )

    model.eval()

    total_dice = 0.0
    total_iou = 0.0
    total_images = 0

    with torch.no_grad():
        for images, gts in loader:
            images = images.to(device, non_blocking=True)
            gts = prepare_mask(gts.to(device, non_blocking=True))

            preds = model_forward(model, images, mode='test')

            # Average all outputs if model returns deep supervision outputs
            logits = ensemble_logits(preds, target_size=gts.shape[2:])
            probs = torch.sigmoid(logits)

            pred_binary = (probs >= 0.5).float()
            gt_binary = (gts >= 0.5).float()

            batch_size = images.size(0)

            for b in range(batch_size):
                total_dice += dice_coefficient(
                    pred_binary[b],
                    gt_binary[b]
                ).item()

                total_iou += iou_score(
                    pred_binary[b],
                    gt_binary[b]
                ).item()

                total_images += 1

    total_images = max(total_images, 1)

    return (
        total_dice / total_images,
        total_iou / total_images,
        total_images
    )


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

def train_one_epoch(train_loader, model, optimizer, epoch, opt, model_name, device):
    global best_val_dice
    global total_train_time

    model.train()

    epoch_start = time.time()
    loss_record = AvgMeter()

    size_rates = [0.75, 1.0, 1.25]
    total_step = len(train_loader)

    for step, (images, gts) in enumerate(train_loader, start=1):
        images_ori = images.to(device, non_blocking=True)
        gts_ori = prepare_mask(gts.to(device, non_blocking=True))

        batch_size = images_ori.size(0)

        for rate in size_rates:
            optimizer.zero_grad(set_to_none=True)

            if rate != 1.0:
                trainsize = int(round(opt.img_size * rate / 32) * 32)

                images_r = F.interpolate(
                    images_ori,
                    size=(trainsize, trainsize),
                    mode='bilinear',
                    align_corners=True
                )
            else:
                images_r = images_ori

            preds = model_forward(model, images_r, mode='train')
            pred_list = to_list(preds)

            loss = 0.0

            # Deep supervision loss for every prediction
            for pred in pred_list:
                gt_r = F.interpolate(
                    gts_ori,
                    size=pred.shape[2:],
                    mode='nearest'
                )

                loss = loss + structure_loss_full(pred, gt_r)

            # Sum-logits loss
            # Resize all predictions to largest output resolution
            largest_size = max(
                [p.shape[2:] for p in pred_list],
                key=lambda s: s[0] * s[1]
            )

            sum_logits = None

            for pred in pred_list:
                pred_r = resize_logits(pred, largest_size)

                if sum_logits is None:
                    sum_logits = pred_r
                else:
                    sum_logits = sum_logits + pred_r

            gt_sum = F.interpolate(
                gts_ori,
                size=largest_size,
                mode='nearest'
            )

            loss = loss + structure_loss_full(sum_logits, gt_sum)

            loss.backward()

            clip_gradient(optimizer, opt.clip)

            optimizer.step()

            if rate == 1.0:
                loss_record.update(loss.item(), batch_size)

        if step % 50 == 0 or step == total_step:
            current_lr = optimizer.param_groups[0]['lr']

            print(
                f'{datetime.now()}  '
                f'Epoch [{epoch:03d}/{opt.epoch}]  '
                f'Step [{step:04d}/{total_step}]  '
                f'Loss: {loss_record.show():.4f}  '
                f'LR: {current_lr:.7f}'
            )

    total_train_time += time.time() - epoch_start

    # Save last checkpoint
    os.makedirs(opt.train_save, exist_ok=True)

    last_path = os.path.join(
        opt.train_save,
        f'{model_name}-last.pth'
    )

    torch.save(model.state_dict(), last_path)

    # Validation
    val_dice, val_iou, n_val = evaluate(
        model=model,
        data_path=opt.val_path,
        opt=opt,
        device=device,
        split_name='Val'
    )

    print(
        f'Epoch {epoch:03d} | '
        f'Val Dice: {val_dice:.4f} | '
        f'Val IoU: {val_iou:.4f} | '
        f'Images: {n_val}'
    )

    # Save best checkpoint
    if val_dice > best_val_dice:
        best_val_dice = val_dice

        best_path = os.path.join(
            opt.train_save,
            f'{model_name}-best.pth'
        )

        torch.save(model.state_dict(), best_path)

        print(f'  >> Best model updated. Best Val Dice: {best_val_dice:.4f}')


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batchsize', type=int, default=8)
    parser.add_argument('--test_batchsize', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=352)
    parser.add_argument('--clip', type=float, default=0.5)

    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--val_path', type=str, required=True)
    parser.add_argument('--test_path', type=str, required=True)

    parser.add_argument('--train_save', type=str, default='./model_pth_full/')
    parser.add_argument('--seed', type=int, default=2024)

    opt = parser.parse_args()

    set_seed(opt.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # Check dataset folders before training
    train_image_root, train_gt_root = get_data_roots(
        opt.train_path,
        split_name='Train',
        verbose=True
    )

    get_data_roots(
        opt.val_path,
        split_name='Val',
        verbose=True
    )

    get_data_roots(
        opt.test_path,
        split_name='Test',
        verbose=True
    )

    print(f'Device     : {device}')
    print(f'Epochs     : {opt.epoch}')
    print(f'Batch size : {opt.batchsize}')
    print(f'Image size : {opt.img_size}')
    print(f'LR         : {opt.lr}')
    print(f'Save path  : {opt.train_save}')

    # Global trackers
    best_val_dice = 0.0
    total_train_time = 0.0

    # Model
    model = EMCADNet(
        num_classes=1,
        kernel_sizes=[1, 3, 5],
        expansion_factor=2,
        dw_parallel=True,
        add=True,
        lgag_ks=3,
        activation='relu6',
        encoder='pvt_v2_b2',
        pretrain=True
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt.lr,
        weight_decay=1e-4
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=opt.epoch,
        eta_min=1e-6
    )

    train_loader = get_loader(
        image_root=train_image_root,
        gt_root=train_gt_root,
        batchsize=opt.batchsize,
        trainsize=opt.img_size,
        shuffle=True,
        augmentation=True
    )

    print('\nStarting full-supervised training...\n')

    model_name = 'full_supervision'

    for epoch in range(1, opt.epoch + 1):
        train_one_epoch(
            train_loader=train_loader,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            opt=opt,
            model_name=model_name,
            device=device
        )

        scheduler.step()

    print('\nTraining finished.')

    best_path = os.path.join(
        opt.train_save,
        f'{model_name}-best.pth'
    )

    # Final test using best validation checkpoint
    if os.path.isfile(best_path):
        print(f'\nLoading best checkpoint:\n  {best_path}')

        state_dict = torch.load(best_path, map_location=device)
        model.load_state_dict(state_dict)

    test_dice, test_iou, n_test = evaluate(
        model=model,
        data_path=opt.test_path,
        opt=opt,
        device=device,
        split_name='Test'
    )

    hours = int(total_train_time // 3600)
    minutes = int((total_train_time % 3600) // 60)

    print('\n========== Final Results ==========')
    print(f'Training time  : {hours}h {minutes}m')
    print(f'Best Val Dice  : {best_val_dice:.4f}')
    print(f'Test Dice      : {test_dice:.4f}')
    print(f'Test IoU       : {test_iou:.4f}')
    print(f'Test images    : {n_test}')
    print('===================================')
