#train_polyp_hup.py  —  HUPAnno training script with multi-phase loss  (CORRECTED v2)
#   Phase 1  (0  → 30%):  L = L_c
#   Phase 2  (30 → 70%):  L = L_c + λ1·L_PCL(easy) + λ2·L_ce + λ_conf·L_conf
#   Phase 3  (70 → 100%): L = L_c + λ1·L_PCL(all)  + λ2·L_ce + λ3·L_patch + λ_conf·L_conf
#
# CORRECTIONS APPLIED (cumulative — see ambiguity table shared separately):
#   1. loss_certain: l_in / l_out / l_rf are MASKED to their confident
#      region only (multiply-by-mask-before-BCE, same pattern BPAnno's
#      dual_mask_loss uses). Ω_Δ (the uncertain ring) receives NO direct
#      gradient from L_in / L_out / L_RF.
#   2. loss_ce is now MASKED to omega_delta (the ring) only, matching
#      BPAnno's classification_loss_ccg exactly — previously it ran
#      densely over the whole image, leaking spatial boundary location
#      information into the ring that stricter weak supervision would
#      not provide.
#   3. Phase 2 / Phase 3 are mutually exclusive (elif), so l_pcl is
#      computed exactly once per step, not twice-and-discarded.
#   4. L_conf is computed AND included in the Phase 3 loss sum (was
#      previously computed but silently dropped from the gradient).
#   5. Single, unambiguous loss assembly after the phase branches —
#      no stray unconditional re-assignment.
#   6. --pretrained_dir is now actually threaded into EMCADNet(...),
#      with a safe fallback if the installed networks_hup.py doesn't
#      yet accept that kwarg (see NOTE at the bottom of __main__).

import os
import time
import argparse
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.ndimage import distance_transform_edt

from lib.networks_hup import EMCADNet
from utils.dataloader_hup import get_loader
from utils.utils import clip_gradient, AvgMeter

LAMBDA_PCL   = 0.1
LAMBDA_CE    = 0.3
LAMBDA_PATCH = 0.2
LAMBDA_CONF  = 0.05

MU_HARD = 0.3    # stricter inside LRP  (LOWER threshold)
MU_EASY = 0.5    # tolerant outside     (HIGHER threshold)

RHO_LRP  = 0.85  # aggressive sampling for annotator-verified LRP pixels
RHO_EASY = 0.5   # standard sampling for entropy-selected easy pixels

PCL_TEMP     = 0.07
PHASE2_START = 0.30
PHASE3_START = 0.70
EMA_DECAY    = 0.99
WARMUP_EPOCHS = 3

best             = 0.0
total_train_time = 0.0
epoch_history    = []


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def warmup_lr(optimizer, epoch, warmup_epochs, base_lr):
    lr = base_lr * (epoch / warmup_epochs)
    for pg in optimizer.param_groups:
        pg['lr'] = lr


def dice_coefficient(pred_bin, gt_bin, eps=1e-6):
    p = pred_bin.view(-1).float()
    g = gt_bin.view(-1).float()
    return (2.0 * (p * g).sum() + eps) / (p.sum() + g.sum() + eps)


def iou_metric(pred_bin, gt_bin, eps=1e-6):
    p = pred_bin.view(-1).float()
    g = gt_bin.view(-1).float()
    i = (p * g).sum()
    return (i + eps) / (p.sum() + g.sum() - i + eps)


def hd95_metric(pred_np, gt_np):
    pred = pred_np.astype(bool)
    gt   = gt_np.astype(bool)
    if not gt.any():
        return 0.0   # no GT annotation — skip this sample
    if not pred.any():
        # model predicted nothing — use max distance from GT surface
        d = distance_transform_edt(~gt)
        return float(np.percentile(d[d > 0], 95)) if (d > 0).any() else 0.0
    d1 = distance_transform_edt(~pred)[gt]
    d2 = distance_transform_edt(~gt)[pred]
    return float(np.percentile(np.concatenate([d1, d2]), 95))


# =========================================================================== #
# structure_loss — kept for reference / potential ablation use, but no
# longer used inside loss_certain (see masked version below).
# =========================================================================== #

def structure_loss(pred, mask):
    """Standard Dice + BCE, computed densely over the full image."""
    if pred.dim() == 4:
        pred = pred.squeeze(1)

    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')

    prob  = torch.sigmoid(pred)
    inter = (prob * mask).sum(dim=(-2, -1))
    union = prob.sum(dim=(-2, -1)) + mask.sum(dim=(-2, -1))
    dice  = 1.0 - (2.0 * inter + 1.0) / (union + 1.0)
    dice  = dice.mean()

    return bce + dice


# =========================================================================== #
# Loss 1 — L_c  (README §1)  — CORRECTED, MASKED VERSION
#
# L_c = L_in(p, y_in) + L_out(p, y_out) + L_RF(p, y_RF) + L_RB(p, y_RB)
#
# Each term is masked to the region it claims to supervise (same
# multiply-by-mask-before-BCE trick as BPAnno's dual_mask_loss), so
# Ω_Δ (the uncertain ring) receives NO direct gradient from L_in/L_out.
# =========================================================================== #

def loss_certain(pred, y_in, y_out, lrp_fg, lrp_bg, eps=1e-6):
    if pred.dim() == 4:
        pred = pred.squeeze(1)   # (B, H, W)

    device = pred.device
    prob   = torch.sigmoid(pred)

    # ── L_in : confident foreground (Ω_I) → predict 1, masked ───────────────
    in_area = y_in.sum() + eps
    l_in = F.binary_cross_entropy(
        (prob * y_in).clamp(eps, 1 - eps),
        y_in,
        reduction='sum'
    ) / in_area

    # ── L_out : confident background (OUTSIDE the outer envelope) ───────────
    # y_out == 1 means "inside outer envelope", so true confident bg = 1-y_out
    outside_mask = (1.0 - y_out)
    out_area     = outside_mask.sum() + eps
    l_out = F.binary_cross_entropy(
        (prob * outside_mask).clamp(eps, 1 - eps),
        torch.zeros_like(prob),
        reduction='sum'
    ) / out_area

    # ── L_RF : LRP-resolved foreground → predict 1, masked ───────────────────
    if lrp_fg.any():
        rf_area = lrp_fg.sum() + eps
        l_rf = F.binary_cross_entropy(
            (prob * lrp_fg).clamp(eps, 1 - eps),
            lrp_fg,
            reduction='sum'
        ) / rf_area
    else:
        l_rf = torch.tensor(0.0, device=device)

    # ── L_RB : LRP-resolved background → predict 0 (already masked) ─────────
    if lrp_bg.any():
        logits_rb = pred[lrp_bg.bool()]
        target_rb = torch.zeros_like(logits_rb)
        l_rb = F.binary_cross_entropy_with_logits(
            logits_rb, target_rb, reduction='mean'
        )
    else:
        l_rb = torch.tensor(0.0, device=device)

    return l_in + l_out + l_rf + l_rb


# =========================================================================== #
# Loss 2 — L_ce  (README §3)  — CORRECTED, MASKED VERSION
#
# CCG classification loss, now restricted to omega_delta (the ring) only,
# matching BPAnno's classification_loss_ccg exactly. Previously this ran
# densely over the entire image, which — even though it didn't leak the
# raw fg/bg label — did leak dense spatial information about exactly
# where the boundary lives, which stricter weak supervision withholds.
# =========================================================================== #

def loss_ce(cls_logits, y_c, omega_delta):
    """
    cls_logits  : (B, C, H, W)
    y_c         : (B, H, W)  long
    omega_delta : (B, H, W)  float {0,1} — 1 inside the uncertain ring
    """
    B, C, H, W  = cls_logits.shape
    logits_flat = cls_logits.permute(0, 2, 3, 1).reshape(-1, C)
    labels_flat = y_c.reshape(-1)
    ring_mask   = omega_delta.reshape(-1).bool()

    if ring_mask.sum() == 0:
        return torch.tensor(0.0, device=cls_logits.device, requires_grad=True)

    return F.cross_entropy(logits_flat[ring_mask], labels_flat[ring_mask])


# =========================================================================== #
# Loss 3 — L_conf  (README §3)
# =========================================================================== #

def loss_conf(conf_map, lrp_mask, mu_hard=MU_HARD, mu_easy=MU_EASY):
    target = torch.full_like(conf_map.squeeze(1), mu_easy)
    target[lrp_mask.bool()] = mu_hard
    return F.mse_loss(conf_map.squeeze(1), target)


# =========================================================================== #
# Loss 4 — L_PCL  (README §2)
# =========================================================================== #

def sample_anchors_lrp(emb, lrp_unc_mask, rho=RHO_LRP, max_n=256):
    idx = lrp_unc_mask.nonzero(as_tuple=False)
    if idx.shape[0] == 0:
        return None
    n    = min(max(1, int(idx.shape[0] * rho)), max_n)
    perm = torch.randperm(idx.shape[0], device=emb.device)[:n]
    return emb[:, idx[perm, 0], idx[perm, 1]].T


def sample_anchors_easy(emb, pred_single, easy_unc_mask,
                        rho=RHO_EASY, max_n=256):
    idx = easy_unc_mask.nonzero(as_tuple=False)
    if idx.shape[0] == 0:
        return None
    prob    = torch.sigmoid(pred_single.squeeze())
    entropy = -(prob * (prob + 1e-6).log() +
                (1 - prob) * (1 - prob + 1e-6).log())
    ent_vals = entropy[idx[:, 0], idx[:, 1]]
    n        = min(max(1, int(idx.shape[0] * rho)), max_n)
    topk     = torch.topk(ent_vals, n).indices
    sel      = idx[topk]
    return emb[:, sel[:, 0], sel[:, 1]].T


def loss_pcl(embeddings, pred, y_in, y_out,
             omega_delta, lrp_mask, lrp_uncertain,
             neg_queue, model, phase,
             temp=PCL_TEMP, max_pos=128, max_neg=128):
    B, D, H, W = embeddings.shape
    device     = embeddings.device
    total_loss = torch.tensor(0.0, device=device)
    n_valid    = 0

    for b in range(B):
        emb   = embeddings[b]
        m_pos = y_in[b].bool()
        m_neg = (y_out[b] == 0)

        if not m_pos.any() or not m_neg.any():
            continue

        anchor_list = []
        easy_unc    = (omega_delta[b] == 1) & ~lrp_mask[b].bool()

        if phase == 2:
            a = sample_anchors_easy(emb, pred[b:b+1], easy_unc)
            if a is not None:
                anchor_list.append(a)

        elif phase >= 3:
            a_lrp = sample_anchors_lrp(emb, lrp_uncertain[b].bool())
            if a_lrp is not None:
                anchor_list.append(a_lrp)
            a_easy = sample_anchors_easy(emb, pred[b:b+1], easy_unc)
            if a_easy is not None:
                anchor_list.append(a_easy)

        if not anchor_list:
            continue

        anch  = torch.cat(anchor_list, dim=0)

        p_idx = m_pos.nonzero(as_tuple=False)
        pp    = torch.randperm(p_idx.shape[0], device=device)[:max_pos]
        pos   = emb[:, p_idx[pp, 0], p_idx[pp, 1]].T.mean(0, keepdim=True)

        n_idx    = m_neg.nonzero(as_tuple=False)
        np_      = torch.randperm(n_idx.shape[0], device=device)[:max_neg]
        neg_live = emb[:, n_idx[np_, 0], n_idx[np_, 1]].T
        negs     = torch.cat([neg_live, neg_queue.T], dim=0)

        sim_pos = (anch * pos).sum(1, keepdim=True) / temp
        sim_neg = (anch @ negs.T) / temp
        logits  = torch.cat([sim_pos, sim_neg], dim=1)
        labels  = torch.zeros(logits.shape[0],
                              dtype=torch.long, device=device)

        total_loss += F.cross_entropy(logits, labels)
        n_valid    += 1

        with torch.no_grad():
            model.update_queue(neg_live.detach())

    return total_loss / max(n_valid, 1)


# =========================================================================== #
# Loss 5 — L_patch  (README §4)
# =========================================================================== #

def loss_patch(pred, lrp_fg, lrp_bg, lrp_uncertain,
               proto_fg, proto_bg, eps=1e-6):
    B          = pred.shape[0]
    prob       = torch.sigmoid(pred.squeeze(1))
    total_loss = torch.tensor(0.0, device=pred.device)
    n_valid    = 0

    for b in range(B):
        unc_mask = lrp_uncertain[b].bool()
        if not unc_mask.any():
            continue

        fg_inv = (255 - lrp_fg[b].cpu().numpy() * 255).astype(np.uint8)
        bg_inv = (255 - lrp_bg[b].cpu().numpy() * 255).astype(np.uint8)

        dist_to_fg = torch.from_numpy(
            cv2.distanceTransform(fg_inv, cv2.DIST_L2, 5)
        ).to(pred.device)
        dist_to_bg = torch.from_numpy(
            cv2.distanceTransform(bg_inv, cv2.DIST_L2, 5)
        ).to(pred.device)

        closer_fg = (dist_to_fg <= dist_to_bg) & unc_mask
        closer_bg = (dist_to_fg >  dist_to_bg) & unc_mask
        p_unc     = prob[b].clamp(eps, 1 - eps)
        loss_b    = torch.tensor(0.0, device=pred.device)

        if closer_fg.any():
            ps = p_unc[closer_fg]
            fg = proto_fg.clamp(eps, 1 - eps)
            kl = fg*(fg.log()-ps.log()) + (1-fg)*((1-fg).log()-(1-ps).log())
            loss_b = loss_b + kl.mean()

        if closer_bg.any():
            ps = p_unc[closer_bg]
            bg = proto_bg.clamp(eps, 1 - eps)
            kl = bg*(bg.log()-ps.log()) + (1-bg)*((1-bg).log()-(1-ps).log())
            loss_b = loss_b + kl.mean()

        total_loss += loss_b
        n_valid    += 1

    return total_loss / max(n_valid, 1)


# =========================================================================== #
# Prototype EMA
# =========================================================================== #

def update_prototypes(proto_fg, proto_bg, pred, y_in, y_out,
                      decay=EMA_DECAY):
    prob    = torch.sigmoid(pred.squeeze(1)).detach()
    fg_mask = y_in.bool()
    bg_mask = (y_out == 0)
    if fg_mask.any():
        proto_fg = decay * proto_fg + (1 - decay) * prob[fg_mask].mean()
    if bg_mask.any():
        proto_bg = decay * proto_bg + (1 - decay) * prob[bg_mask].mean()
    return proto_fg, proto_bg


# =========================================================================== #
# Phase helper
# =========================================================================== #

def get_phase(epoch, total_epochs):
    p = epoch / total_epochs
    if p < PHASE2_START:  return 1
    elif p < PHASE3_START: return 2
    else:                  return 3


# =========================================================================== #
# Evaluation
# =========================================================================== #

def test(model, image_root, mask_root, opt, threshold=0.5):
    model.eval()
    loader = get_loader(
        image_root   = image_root,
        gt_root      = mask_root,
        batchsize    = opt.test_batchsize,
        trainsize    = opt.img_size,
        shuffle      = False,
        num_workers  = 2,
        pin_memory   = True,
        augmentation = False,
        K            = opt.K
    )
    DSC = IOU = HD = total = hd_valid = 0.0
    with torch.no_grad():
        for batch in loader:
            images = batch[0].cuda()
            gt     = batch[-1].cuda().float()
            preds  = model(images, mode='test')
            if isinstance(preds, list):
                preds = preds[-1]
            for i in range(images.shape[0]):
                p_prob = torch.sigmoid(preds[i]).squeeze()
                p_bin  = (p_prob >= threshold).float()
                g_bin  = (gt[i].squeeze() >= 0.5).float()

                if g_bin.sum() == 0:
                    continue

                DSC   += dice_coefficient(p_bin, g_bin).item()
                IOU   += iou_metric(p_bin, g_bin).item()

                hd = hd95_metric(
                    p_bin.cpu().numpy().astype(np.uint8),
                    g_bin.cpu().numpy().astype(np.uint8)
                )
                HD       += hd
                hd_valid += 1
                total    += 1

    n  = max(total, 1)
    nh = max(hd_valid, 1)
    return DSC/n, IOU/n, HD/nh, int(total)


# =========================================================================== #
# Summary table
# =========================================================================== #

def print_summary_table():
    print(f'\n{"="*72}')
    print(f'  TRAINING SUMMARY')
    print(f'{"="*72}')
    print(f'  {"Ep":>4}  {"Ph"}  {"LR":>9}  {"Loss":>8}  '
          f'{"Dice":>8}  {"IoU":>8}  {"HD95":>8}')
    print(f'  {"─"*4}  {"──"}  {"─"*9}  {"─"*8}  '
          f'{"─"*8}  {"─"*8}  {"─"*8}')
    for r in epoch_history:
        marker = ' ✓' if r['best'] else ''
        print(f'  {r["epoch"]:>4}  Ph{r["phase"]}  '
              f'{r["lr"]:>9.2e}  '
              f'{r["loss"]:>8.4f}  '
              f'{r["dice"]:>8.4f}  '
              f'{r["iou"]:>8.4f}  '
              f'{r["hd95"]:>8.2f}'
              f'{marker}')
    print(f'{"="*72}\n')


# =========================================================================== #
# Training — one epoch
# =========================================================================== #

def train(train_loader, model, optimizer, epoch, opt, model_name,
          proto_fg, proto_bg):
    global best, total_train_time, epoch_history

    model.train()
    phase       = get_phase(epoch, opt.epoch)
    epoch_start = time.time()
    loss_record = AvgMeter()
    size_rates  = [0.75, 1.0, 1.25]
    total_step  = len(train_loader)
    current_lr  = get_lr(optimizer)

    print(f'\n{"="*65}')
    print(f'  Epoch {epoch}/{opt.epoch}  |  Phase {phase}  |  '
          f'LR {current_lr:.2e}  |  '
          f'{datetime.now().strftime("%H:%M:%S")}')
    print(f'{"="*65}')

    for step, batch in enumerate(train_loader, start=1):
        (images,
         y_in, y_out, omega_delta,
         lrp_fg, lrp_bg, lrp_uncertain, lrp_mask,
         y_c, gt) = batch

        images        = images.cuda()
        y_in          = y_in.cuda()
        y_out         = y_out.cuda()
        omega_delta   = omega_delta.cuda()
        lrp_fg        = lrp_fg.cuda()
        lrp_bg        = lrp_bg.cuda()
        lrp_uncertain = lrp_uncertain.cuda()
        lrp_mask      = lrp_mask.cuda()
        y_c           = y_c.cuda()

        for rate in size_rates:
            optimizer.zero_grad()

            if rate != 1.0:
                sz       = int(round(opt.img_size * rate / 32) * 32)
                images_r = F.interpolate(images, size=(sz, sz),
                                         mode='bilinear', align_corners=True)
            else:
                images_r = images

            preds, cls_logits, embeddings, conf_map = model(
                images_r, mode='train'
            )
            pred_size = preds[-1].shape[2:]

            def rsz(m, long=False):
                out = F.interpolate(
                    m.unsqueeze(1).float(), size=pred_size, mode='nearest'
                ).squeeze(1)
                return out.long() if long else out

            y_in_r        = rsz(y_in)
            y_out_r       = rsz(y_out)
            omega_delta_r = rsz(omega_delta)
            lrp_fg_r      = rsz(lrp_fg)
            lrp_bg_r      = rsz(lrp_bg)
            lrp_unc_r     = rsz(lrp_uncertain)
            lrp_mask_r    = rsz(lrp_mask)
            y_c_r         = rsz(y_c, long=True)

            cls_r  = F.interpolate(cls_logits, size=pred_size,
                                   mode='bilinear', align_corners=False)
            conf_r = F.interpolate(conf_map, size=pred_size,
                                   mode='bilinear', align_corners=False)
            emb_r  = F.normalize(
                F.interpolate(embeddings, size=pred_size,
                              mode='bilinear', align_corners=False), dim=1
            )
            p_finest = preds[-1]   # (B,1,H,W)

            # ── Phase 1 base term: L_c, always on, summed over deep supervision ──
            l_c = sum(
                loss_certain(p, y_in_r, y_out_r, lrp_fg_r, lrp_bg_r)
                for p in preds
            )

            l_ce    = torch.tensor(0.0, device=images.device)
            l_conf  = torch.tensor(0.0, device=images.device)
            l_pcl   = torch.tensor(0.0, device=images.device)
            l_patch = torch.tensor(0.0, device=images.device)

            # ── Phase 2: L = L_c + λ1·PCL(easy) + λ2·Lce + λ_conf·Lconf ──────
            if phase == 2:
                l_ce   = loss_ce(cls_r, y_c_r, omega_delta_r)
                l_conf = loss_conf(conf_r, lrp_mask_r)
                l_pcl  = loss_pcl(
                    emb_r, p_finest,
                    y_in_r, y_out_r,
                    omega_delta_r, lrp_mask_r, lrp_unc_r,
                    model.neg_queue.detach(), model, phase=2
                )

            # ── Phase 3: L = L_c + λ1·PCL(all) + λ2·Lce + λ3·Lpatch + λ_conf·Lconf ──
            elif phase >= 3:
                l_ce    = loss_ce(cls_r, y_c_r, omega_delta_r)
                l_conf  = loss_conf(conf_r, lrp_mask_r)
                l_pcl   = loss_pcl(
                    emb_r, p_finest,
                    y_in_r, y_out_r,
                    omega_delta_r, lrp_mask_r, lrp_unc_r,
                    model.neg_queue.detach(), model, phase=3
                )
                l_patch = loss_patch(
                    p_finest,
                    lrp_fg_r, lrp_bg_r, lrp_unc_r,
                    proto_fg, proto_bg
                )

            # Single, unambiguous loss assembly for every phase.
            # Phase 1: l_ce/l_conf/l_pcl/l_patch are all 0.0 tensors, so
            # this reduces exactly to loss = l_c.
            loss = (l_c
                    + LAMBDA_PCL   * l_pcl
                    + LAMBDA_CE    * l_ce
                    + LAMBDA_PATCH * l_patch
                    + LAMBDA_CONF  * l_conf)

            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            proto_fg, proto_bg = update_prototypes(
                proto_fg, proto_bg,
                p_finest.detach(), y_in_r, y_out_r
            )

            if rate == 1.0:
                loss_record.update(loss.item(), opt.batchsize)

        if step % 50 == 0 or step == total_step:
            print(
                f'  {datetime.now().strftime("%H:%M:%S")}  '
                f'Step [{step:>4}/{total_step}]  '
                f'Loss {loss_record.show():.4f}  '
                f'Lc {l_c.item():.3f}  '
                f'Lce {l_ce.item():.3f}  '
                f'Lpcl {l_pcl.item():.3f}  '
                f'Lpatch {l_patch.item():.3f}  '
                f'Lconf {l_conf.item():.4f}'
            )

    epoch_time        = time.time() - epoch_start
    total_train_time += epoch_time

    os.makedirs(opt.train_save, exist_ok=True)
    torch.save(model.state_dict(),
               os.path.join(opt.train_save, f'{model_name}-last.pth'))

    model.eval()
    d_dice, d_iou, d_hd95, n_samples = test(
        model, opt.val_image_root, opt.val_mask_root, opt
    )

    is_best = d_dice > best
    marker  = ' ← best' if is_best else ''

    print(f'\n  ┌─ Epoch {epoch:>3}/{opt.epoch}  Ph{phase}  '
          f'LR {current_lr:.2e}  ({epoch_time/60:.1f} min) ───────')
    print(f'  │  Train Loss : {loss_record.show():.4f}')
    print(f'  │  Val  Dice  : {d_dice:.4f}{marker}')
    print(f'  │  Val  IoU   : {d_iou:.4f}')
    print(f'  │  Val  HD95  : {d_hd95:.2f} px')
    print(f'  │  proto_fg={proto_fg.item():.3f}  '
          f'proto_bg={proto_bg.item():.3f}  n={n_samples}')
    print(f'  └{"─"*55}')

    if is_best:
        best = d_dice
        torch.save(model.state_dict(),
                   os.path.join(opt.train_save, f'{model_name}-best.pth'))
        print(f'  ✓  Best  Dice={best:.4f}  IoU={d_iou:.4f}  '
              f'HD95={d_hd95:.2f}')

    epoch_history.append({
        'epoch' : epoch,
        'phase' : phase,
        'lr'    : current_lr,
        'loss'  : loss_record.show(),
        'dice'  : d_dice,
        'iou'   : d_iou,
        'hd95'  : d_hd95,
        'best'  : is_best,
    })

    return proto_fg, proto_bg


# =========================================================================== #
# Entry point
# =========================================================================== #

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch',            type=int,   default=200)
    parser.add_argument('--lr',               type=float, default=1e-4)
    parser.add_argument('--batchsize',        type=int,   default=8)
    parser.add_argument('--test_batchsize',   type=int,   default=8)
    parser.add_argument('--img_size',         type=int,   default=352)
    parser.add_argument('--clip',             type=float, default=0.5)
    parser.add_argument('--K',                type=int,   default=2)
    parser.add_argument('--train_save',       type=str,
                        default='./model_pth_hup/')
    parser.add_argument('--train_image_root', type=str,   required=True)
    parser.add_argument('--train_mask_root',  type=str,   required=True)
    parser.add_argument('--val_image_root',   type=str,   required=True)
    parser.add_argument('--val_mask_root',    type=str,   required=True)
    parser.add_argument('--pretrained_dir',   type=str,
                        default='./pretrained_pth/pvt/',
                        help='Directory containing pvt_v2_b2.pth')
    opt = parser.parse_args()

    for label, path in [
        ('train_image_root', opt.train_image_root),
        ('train_mask_root',  opt.train_mask_root),
        ('val_image_root',   opt.val_image_root),
        ('val_mask_root',    opt.val_mask_root),
    ]:
        if not os.path.isdir(path):
            raise FileNotFoundError(f'--{label} not found: {path}')
    print('✓  All data paths verified.')

    device           = torch.device('cuda')
    best             = 0.0
    total_train_time = 0.0
    epoch_history    = []
    proto_fg         = torch.tensor(0.8, device=device)
    proto_bg         = torch.tensor(0.2, device=device)

    # NOTE: EMCADNet(...) is only given pretrained_dir=opt.pretrained_dir if
    # the installed lib/networks_hup.py actually accepts that keyword.
    # If your networks_hup.py has a hardcoded internal checkpoint path
    # instead of a constructor argument, this try/except falls back to the
    # old call so training doesn't crash — but in that case you still need
    # to either (a) place pvt_v2_b2.pth at whatever path networks_hup.py
    # hardcodes, or (b) edit networks_hup.py to accept pretrained_dir.
    try:
        model = EMCADNet(
            num_classes=1, kernel_sizes=[1,3,5],
            expansion_factor=2, dw_parallel=True, add=True,
            lgag_ks=3, activation='relu6',
            encoder='pvt_v2_b2', pretrain=True,
            pretrained_dir=opt.pretrained_dir
        ).to(device)
    except TypeError:
        print('  [warning] lib/networks_hup.py EMCADNet does not accept '
              '"pretrained_dir" — falling back to its hardcoded internal '
              'path. --pretrained_dir is being ignored.')
        model = EMCADNet(
            num_classes=1, kernel_sizes=[1,3,5],
            expansion_factor=2, dw_parallel=True, add=True,
            lgag_ks=3, activation='relu6',
            encoder='pvt_v2_b2', pretrain=True
        ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=opt.lr, weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=max(1, opt.epoch - WARMUP_EPOCHS),
        eta_min=1e-6
    )

    train_loader = get_loader(
        image_root   = opt.train_image_root,
        gt_root      = opt.train_mask_root,
        batchsize    = opt.batchsize,
        trainsize    = opt.img_size,
        shuffle      = True,
        num_workers  = 4,
        pin_memory   = True,
        augmentation = True,
        K            = opt.K
    )

    p2_ep = int(opt.epoch * PHASE2_START)
    p3_ep = int(opt.epoch * PHASE3_START)
    print(f'\nHUPAnno  |  {opt.epoch} epochs  |  K={opt.K}')
    print(f'  Phase 1  ep 1–{p2_ep}       : L = L_c')
    print(f'  Phase 2  ep {p2_ep+1}–{p3_ep}  : '
          f'L = L_c + PCL(easy) + Lce(ring) + Lconf')
    print(f'  Phase 3  ep {p3_ep+1}–{opt.epoch}  : '
          f'L = L_c + PCL(LRP+easy) + Lce(ring) + Lpatch + Lconf')
    print(f'  ρ_LRP={RHO_LRP}  ρ_easy={RHO_EASY}')
    print(f'  μ_hard={MU_HARD} (LRP stricter)  μ_easy={MU_EASY} (tolerant)')
    print(f'  Train: {opt.train_image_root}')
    print(f'  Val  : {opt.val_image_root}\n')

    for epoch in range(1, opt.epoch + 1):
        if epoch <= WARMUP_EPOCHS:
            warmup_lr(optimizer, epoch, WARMUP_EPOCHS, opt.lr)
        else:
            scheduler.step()

        proto_fg, proto_bg = train(
            train_loader, model, optimizer, epoch, opt,
            'ws_hupanno', proto_fg, proto_bg
        )

    print_summary_table()
    print(f'Total time : {total_train_time/3600:.2f} h')
    print(f'Best Dice  : {best:.4f}')