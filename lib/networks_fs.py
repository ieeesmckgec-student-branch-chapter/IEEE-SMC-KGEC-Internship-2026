import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.pvtv2 import pvt_v2_b2
from lib.decoders import EMCAD


class EMCADNet(nn.Module):
    def __init__(self, num_classes=1, kernel_sizes=[1, 3, 5],
                 expansion_factor=2, dw_parallel=True, add=True,
                 lgag_ks=3, activation='relu6', encoder='pvt_v2_b2', pretrain=True):
        super().__init__()

        # Grayscale -> RGB conversion (for 1-channel inputs)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )

        self.backbone = pvt_v2_b2()
        channels = [512, 320, 128, 64]

        if pretrain:
            ckpt = torch.load(
                './pretrained_pth/pvt/pvt_v2_b2.pth',
                map_location='cpu'
            )
            self.backbone.load_state_dict(ckpt, strict=False)
            print("Loaded PVTv2-B2 pretrained weights.")

        self.decoder = EMCAD(
            channels=channels,
            kernel_sizes=kernel_sizes,
            expansion_factor=expansion_factor,
            dw_parallel=dw_parallel,
            add=add,
            lgag_ks=lgag_ks,
            activation=activation
        )

        self.out_head4 = nn.Conv2d(channels[0], num_classes, 1)
        self.out_head3 = nn.Conv2d(channels[1], num_classes, 1)
        self.out_head2 = nn.Conv2d(channels[2], num_classes, 1)
        self.out_head1 = nn.Conv2d(channels[3], num_classes, 1)

    def forward(self, x, mode='train'):
        # Handle grayscale input
        if x.size(1) == 1:
            x = self.conv(x)

        orig_h, orig_w = x.shape[2], x.shape[3]

        # Encoder (PVTv2 backbone)
        x1, x2, x3, x4 = self.backbone(x)

        # Decoder (EMCAD)
        dec_outs = self.decoder(x4, [x3, x2, x1])

        # Upsample all predictions back to input resolution
        p4 = F.interpolate(
            self.out_head4(dec_outs[0]),
            size=(orig_h, orig_w),
            mode='bilinear',
            align_corners=False
        )
        p3 = F.interpolate(
            self.out_head3(dec_outs[1]),
            size=(orig_h, orig_w),
            mode='bilinear',
            align_corners=False
        )
        p2 = F.interpolate(
            self.out_head2(dec_outs[2]),
            size=(orig_h, orig_w),
            mode='bilinear',
            align_corners=False
        )
        p1 = F.interpolate(
            self.out_head1(dec_outs[3]),
            size=(orig_h, orig_w),
            mode='bilinear',
            align_corners=False
        )

        if mode == 'test':
            # Return finest prediction first for inference
            return [p1, p2, p3, p4]
        
        # During training return all predictions for deep supervision
        # Order: coarsest -> finest  [p4, p3, p2, p1]
        return [p4, p3, p2, p1]