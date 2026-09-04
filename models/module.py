import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import sys
sys.path.append("..")
from utils import local_pcd

def init_bn(module):
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return


def init_uniform(module, init_method):
    if module.weight is not None:
        if init_method == "kaiming":
            nn.init.kaiming_uniform_(module.weight)
        elif init_method == "xavier":
            nn.init.xavier_uniform_(module.weight)
    return

class Conv2d(nn.Module):
    """Applies a 2D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class Deconv2d(nn.Module):
    """Applies a 2D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv2d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        y = self.conv(x)
        if self.stride == 2:
            h, w = list(x.size())[2:]
            y = y[:, :, :2 * h, :2 * w].contiguous()
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)

class Conv3d(nn.Module):
    """Applies a 3D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Conv3d, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)

class Deconv3d(nn.Module):
    """Applies a 3D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv3d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        y = self.conv(x)
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)



class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvBn3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = ConvBnReLU(in_channels, out_channels, kernel_size=3, stride=stride, pad=1)
        self.conv2 = ConvBn(out_channels, out_channels, kernel_size=3, stride=1, pad=1)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        return out


class Hourglass3d(nn.Module):
    def __init__(self, channels):
        super(Hourglass3d, self).__init__()

        self.conv1a = ConvBnReLU3D(channels, channels * 2, kernel_size=3, stride=2, pad=1)
        self.conv1b = ConvBnReLU3D(channels * 2, channels * 2, kernel_size=3, stride=1, pad=1)

        self.conv2a = ConvBnReLU3D(channels * 2, channels * 4, kernel_size=3, stride=2, pad=1)
        self.conv2b = ConvBnReLU3D(channels * 4, channels * 4, kernel_size=3, stride=1, pad=1)

        self.dconv2 = nn.Sequential(
            nn.ConvTranspose3d(channels * 4, channels * 2, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(channels * 2))

        self.dconv1 = nn.Sequential(
            nn.ConvTranspose3d(channels * 2, channels, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(channels))

        self.redir1 = ConvBn3D(channels, channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = ConvBn3D(channels * 2, channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1b(self.conv1a(x))
        conv2 = self.conv2b(self.conv2a(conv1))
        dconv2 = F.relu(self.dconv2(conv2) + self.redir2(conv1), inplace=True)
        dconv1 = F.relu(self.dconv1(dconv2) + self.redir1(x), inplace=True)
        return dconv1


def homo_warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth] o [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth,
                                                                                            -1)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy

    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                   padding_mode='zeros')
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

    return warped_src_fea

class DeConv2dFuse(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, relu=True, bn=True,
                 bn_momentum=0.1):
        super(DeConv2dFuse, self).__init__()

        self.deconv = Deconv2d(in_channels, out_channels, kernel_size, stride=2, padding=1, output_padding=1,
                               bn=True, relu=relu, bn_momentum=bn_momentum)

        self.conv = Conv2d(2*out_channels, out_channels, kernel_size, stride=1, padding=1,
                           bn=bn, relu=relu, bn_momentum=bn_momentum)

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x_pre, x):
        x = self.deconv(x)
        x = torch.cat((x, x_pre), dim=1)
        x = self.conv(x)
        return x


class FeatureNet(nn.Module):
    def __init__(self, base_channels, num_stage=3, stride=4, arch_mode="unet"):
        super(FeatureNet, self).__init__()
        assert arch_mode in ["unet", "fpn"], print("mode must be in 'unet' or 'fpn', but get:{}".format(arch_mode))
        print("*************feature extraction arch mode:{}****************".format(arch_mode))
        self.arch_mode = arch_mode
        self.stride = stride
        self.base_channels = base_channels
        self.num_stage = num_stage

        self.conv0 = nn.Sequential(
            Conv2d(3, base_channels, 3, 1, padding=1),
            Conv2d(base_channels, base_channels, 3, 1, padding=1),
        )

        self.conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )

        self.conv2 = nn.Sequential(
            Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        )

        self.out1 = nn.Conv2d(base_channels * 4, base_channels * 4, 1, bias=False)
        self.out_channels = [4 * base_channels]

        if self.arch_mode == 'unet':
            if num_stage == 3:
                self.deconv1 = DeConv2dFuse(base_channels * 4, base_channels * 2, 3)
                self.deconv2 = DeConv2dFuse(base_channels * 2, base_channels, 3)

                self.out2 = nn.Conv2d(base_channels * 2, base_channels * 2, 1, bias=False)
                self.out3 = nn.Conv2d(base_channels, base_channels, 1, bias=False)
                self.out_channels.append(2 * base_channels)
                self.out_channels.append(base_channels)

            elif num_stage == 2:
                self.deconv1 = DeConv2dFuse(base_channels * 4, base_channels * 2, 3)

                self.out2 = nn.Conv2d(base_channels * 2, base_channels * 2, 1, bias=False)
                self.out_channels.append(2 * base_channels)
        elif self.arch_mode == "fpn":
            final_chs = base_channels * 4
            if num_stage == 3:
                self.inner1 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)
                self.inner2 = nn.Conv2d(base_channels * 1, final_chs, 1, bias=True)

                self.out2 = nn.Conv2d(final_chs, base_channels * 2, 3, padding=1, bias=False)
                self.out3 = nn.Conv2d(final_chs, base_channels, 3, padding=1, bias=False)
                self.out_channels.append(base_channels * 2)
                self.out_channels.append(base_channels)

            elif num_stage == 2:
                self.inner1 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)

                self.out2 = nn.Conv2d(final_chs, base_channels, 3, padding=1, bias=False)
                self.out_channels.append(base_channels)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)

        intra_feat = conv2
        outputs = {}
        out = self.out1(intra_feat)
        outputs["stage1"] = out
        if self.arch_mode == "unet":
            if self.num_stage == 3:
                intra_feat = self.deconv1(conv1, intra_feat)
                out = self.out2(intra_feat)
                outputs["stage2"] = out

                intra_feat = self.deconv2(conv0, intra_feat)
                out = self.out3(intra_feat)
                outputs["stage3"] = out

            elif self.num_stage == 2:
                intra_feat = self.deconv1(conv1, intra_feat)
                out = self.out2(intra_feat)
                outputs["stage2"] = out

        elif self.arch_mode == "fpn":
            if self.num_stage == 3:
                intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(conv1)
                out = self.out2(intra_feat)
                outputs["stage2"] = out

                intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner2(conv0)
                out = self.out3(intra_feat)
                outputs["stage3"] = out

            elif self.num_stage == 2:
                intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(conv1)
                out = self.out2(intra_feat)
                outputs["stage2"] = out

        return outputs

class CostRegNet(nn.Module):
    def __init__(self, in_channels, base_channels):
        super(CostRegNet, self).__init__()
        self.conv0 = Conv3d(in_channels, base_channels, padding=1)

        self.conv1 = Conv3d(base_channels, base_channels * 2, stride=2, padding=1)
        self.conv2 = Conv3d(base_channels * 2, base_channels * 2, padding=1)

        self.conv3 = Conv3d(base_channels * 2, base_channels * 4, stride=2, padding=1)
        self.conv4 = Conv3d(base_channels * 4, base_channels * 4, padding=1)

        self.conv5 = Conv3d(base_channels * 4, base_channels * 8, stride=2, padding=1)
        self.conv6 = Conv3d(base_channels * 8, base_channels * 8, padding=1)

        self.conv7 = Deconv3d(base_channels * 8, base_channels * 4, stride=2, padding=1, output_padding=1)

        self.conv9 = Deconv3d(base_channels * 4, base_channels * 2, stride=2, padding=1, output_padding=1)

        self.conv11 = Deconv3d(base_channels * 2, base_channels * 1, stride=2, padding=1, output_padding=1)

        self.prob = nn.Conv3d(base_channels, 1, 3, stride=1, padding=1, bias=False)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x)
        return x

class RefineNet(nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()
        self.conv1 = ConvBnReLU(4, 32)
        self.conv2 = ConvBnReLU(32, 32)
        self.conv3 = ConvBnReLU(32, 32)
        self.res = ConvBnReLU(32, 1)

    def forward(self, img, depth_init):
        concat = F.cat((img, depth_init), dim=1)
        depth_residual = self.res(self.conv3(self.conv2(self.conv1(concat))))
        depth_refined = depth_init + depth_residual
        return depth_refined


class NormalHead(nn.Module):
    def __init__(self, in_channels, hidden_channels=32):
        super(NormalHead, self).__init__()
        self.conv1 = Conv2d(in_channels, hidden_channels, 3, 1, padding=1)
        self.conv2 = Conv2d(hidden_channels, hidden_channels, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(hidden_channels, 3, 3, stride=1, padding=1, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return F.normalize(x, p=2, dim=1, eps=1e-6)


def depth_regression(p, depth_values):
    if depth_values.dim() <= 2:
        # print("regression dim <= 2")
        depth_values = depth_values.view(*depth_values.shape, 1, 1)
    depth = torch.sum(p * depth_values, 1)

    return depth


def gradient_x(img):
    return img[..., :, 1:] - img[..., :, :-1]


def gradient_y(img):
    return img[..., 1:, :] - img[..., :-1, :]


def get_depth_normals(depth):
    dzdx = gradient_x(depth)
    dzdy = gradient_y(depth)
    dzdx = torch.cat((dzdx, dzdx[:, :, -1:]), dim=2)
    dzdy = torch.cat((dzdy, dzdy[:, -1:, :]), dim=1)
    normal = torch.stack((-dzdx, -dzdy, torch.ones_like(depth)), dim=1)
    return F.normalize(normal, p=2, dim=1, eps=1e-6)


def compute_normal_from_depth(depth, intrinsics):
    if depth.dim() == 4:
        depth = depth.squeeze(1)
    batch, height, width = depth.shape
    fx = intrinsics[:, 0, 0].view(-1, 1, 1).clamp(min=1e-6)
    fy = intrinsics[:, 1, 1].view(-1, 1, 1).clamp(min=1e-6)
    cx = intrinsics[:, 0, 2].view(-1, 1, 1)
    cy = intrinsics[:, 1, 2].view(-1, 1, 1)
    x = torch.arange(width, dtype=depth.dtype, device=depth.device).view(1, 1, width)
    y = torch.arange(height, dtype=depth.dtype, device=depth.device).view(1, height, 1)
    x = x.expand(batch, height, width)
    y = y.expand(batch, height, width)
    points = torch.stack(((x - cx) / fx * depth,
                          (y - cy) / fy * depth,
                          depth), dim=1)
    tangent_x = gradient_x(points)
    tangent_y = gradient_y(points)
    tangent_x = torch.cat((tangent_x, tangent_x[:, :, :, -1:]), dim=3)
    tangent_y = torch.cat((tangent_y, tangent_y[:, :, -1:, :]), dim=2)
    normal = torch.cross(tangent_x, tangent_y, dim=1)
    return F.normalize(normal, p=2, dim=1, eps=1e-6)


def image_gradient_magnitude(ref_img, target_size):
    ref_img = F.interpolate(ref_img, size=target_size, mode='bilinear', align_corners=False)
    img_dx = gradient_x(ref_img).abs().mean(dim=1)
    img_dy = gradient_y(ref_img).abs().mean(dim=1)
    grad = ref_img.new_zeros(ref_img.size(0), target_size[0], target_size[1])
    grad[:, :, 1:] = torch.max(grad[:, :, 1:], img_dx)
    grad[:, :, :-1] = torch.max(grad[:, :, :-1], img_dx)
    grad[:, 1:, :] = torch.max(grad[:, 1:, :], img_dy)
    grad[:, :-1, :] = torch.max(grad[:, :-1, :], img_dy)
    return grad


def sobel_gradient_magnitude(ref_img, target_size):
    ref_img = F.interpolate(ref_img, size=target_size, mode='bilinear', align_corners=False)
    gray = ref_img.mean(dim=1, keepdim=True)
    kernel_x = gray.new_tensor([[-1.0, 0.0, 1.0],
                                [-2.0, 0.0, 2.0],
                                [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
    kernel_y = gray.new_tensor([[-1.0, -2.0, -1.0],
                                [0.0, 0.0, 0.0],
                                [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
    grad_x = F.conv2d(gray, kernel_x, padding=1)
    grad_y = F.conv2d(gray, kernel_y, padding=1)
    return torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-12).squeeze(1)


def depth_gradient_magnitude(depth, valid_mask):
    if depth.dim() == 4:
        depth = depth.squeeze(1)
    if valid_mask.dim() == 4:
        valid_mask = valid_mask.squeeze(1)
    valid_mask = valid_mask > 0.5
    depth_mean = masked_mean(depth, valid_mask).detach()
    depth_norm = depth / (depth_mean + 1e-6)
    grad = depth.new_zeros(depth.shape)
    depth_dx = gradient_x(depth_norm).abs()
    depth_dy = gradient_y(depth_norm).abs()
    if depth_dx.numel() > 0:
        grad = torch.max(grad, F.pad(depth_dx, (1, 0)))
        grad = torch.max(grad, F.pad(depth_dx, (0, 1)))
    if depth_dy.numel() > 0:
        grad = torch.max(grad, F.pad(depth_dy, (0, 0, 1, 0)))
        grad = torch.max(grad, F.pad(depth_dy, (0, 0, 0, 1)))
    return grad


def curvature_magnitude(depth, valid_mask):
    if depth.dim() == 4:
        depth = depth.squeeze(1)
    if valid_mask.dim() == 4:
        valid_mask = valid_mask.squeeze(1)
    valid_mask = valid_mask > 0.5
    depth_mean = masked_mean(depth, valid_mask).detach()
    depth_norm = depth / (depth_mean + 1e-6)
    curv = depth.new_zeros(depth.shape)
    if depth.size(2) >= 3:
        curv_x = (depth_norm[:, :, 2:] - 2 * depth_norm[:, :, 1:-1] + depth_norm[:, :, :-2]).abs()
        curv = torch.max(curv, F.pad(curv_x, (1, 1)))
    if depth.size(1) >= 3:
        curv_y = (depth_norm[:, 2:, :] - 2 * depth_norm[:, 1:-1, :] + depth_norm[:, :-2, :]).abs()
        curv = torch.max(curv, F.pad(curv_y, (0, 0, 1, 1)))
    return curv


def build_smooth_mask(ref_img, valid_mask, confidence, depth_normal_conf_threshold,
                      edge_grad_threshold, target_size):
    if valid_mask.dim() == 4:
        valid_mask = valid_mask.squeeze(1)
    if valid_mask.shape[-2:] != target_size:
        valid_mask = F.interpolate(valid_mask.float().unsqueeze(1), size=target_size,
                                   mode='nearest').squeeze(1)
    valid_mask = valid_mask > 0.5
    if confidence.dim() == 4:
        confidence = confidence.squeeze(1)
    if confidence.shape[-2:] != target_size:
        confidence = F.interpolate(confidence.unsqueeze(1), size=target_size,
                                   mode='bilinear', align_corners=False).squeeze(1)
    high_confidence_mask = confidence > depth_normal_conf_threshold
    edge_grad = image_gradient_magnitude(ref_img, target_size)
    non_edge_mask = edge_grad < edge_grad_threshold
    return valid_mask & high_confidence_mask & non_edge_mask


def build_geometry_weight(ref_img, valid_mask, confidence, target_size,
                          conf_mid=0.65, k_conf=10.0,
                          edge_mid=0.25, k_edge=10.0, w_min=0.05):
    if valid_mask.dim() == 4:
        valid_mask = valid_mask.squeeze(1)
    if valid_mask.shape[-2:] != target_size:
        valid_mask = F.interpolate(valid_mask.float().unsqueeze(1), size=target_size,
                                   mode='nearest').squeeze(1)
    valid_weight = (valid_mask > 0.5).float()

    if confidence.dim() == 4:
        confidence = confidence.squeeze(1)
    confidence = confidence.detach()
    if confidence.shape[-2:] != target_size:
        confidence = F.interpolate(confidence.unsqueeze(1), size=target_size,
                                   mode='bilinear', align_corners=False).squeeze(1)

    edge_grad = image_gradient_magnitude(ref_img, target_size)
    w_conf = torch.sigmoid(k_conf * (confidence - conf_mid))
    w_edge = torch.sigmoid(k_edge * (edge_mid - edge_grad))
    geometry_weight = valid_weight * (w_min + (1.0 - w_min) * w_conf * w_edge)

    valid_sum = valid_weight.sum() + 1e-6
    metrics = {
        "geometry_weight_mean": geometry_weight.mean().detach(),
        "geometry_weight_valid_mean": (geometry_weight.sum() / valid_sum).detach(),
        "high_weight_ratio": (geometry_weight > 0.7).float().mean().detach(),
        "low_weight_ratio": (geometry_weight < 0.2).float().mean().detach(),
    }
    return geometry_weight, metrics


def build_dual_region_geometry(ref_img, valid_mask, depth, confidence, target_size,
                               threshold_edge=0.25, threshold_depth=0.02,
                               threshold_curv=0.02, conf_mid=0.65,
                               k_conf=10.0, smooth_k=2.0):
    if valid_mask.dim() == 4:
        valid_mask = valid_mask.squeeze(1)
    if valid_mask.shape[-2:] != target_size:
        valid_mask = F.interpolate(valid_mask.float().unsqueeze(1), size=target_size,
                                   mode='nearest').squeeze(1)
    valid_mask = valid_mask > 0.5

    if depth.dim() == 4:
        depth = depth.squeeze(1)
    if depth.shape[-2:] != target_size:
        depth = F.interpolate(depth.unsqueeze(1), size=target_size,
                              mode='bilinear', align_corners=False).squeeze(1)
    depth_for_mask = depth.detach()

    if confidence.dim() == 4:
        confidence = confidence.squeeze(1)
    confidence = confidence.detach()
    if confidence.shape[-2:] != target_size:
        confidence = F.interpolate(confidence.unsqueeze(1), size=target_size,
                                   mode='bilinear', align_corners=False).squeeze(1)

    edge_map = sobel_gradient_magnitude(ref_img, target_size)
    depth_grad = depth_gradient_magnitude(depth_for_mask, valid_mask)
    curv_map = curvature_magnitude(depth_for_mask, valid_mask)

    edge_mask = edge_map > threshold_edge
    depth_edge_mask = depth_grad > threshold_depth
    joint_mask = edge_mask & depth_edge_mask
    high_curv_mask = curv_map > threshold_curv
    region_a = valid_mask & (joint_mask | high_curv_mask)
    region_b = valid_mask & (~region_a)

    confidence_weight = torch.sigmoid(k_conf * (confidence - conf_mid))
    smooth_weight = torch.exp(-smooth_k * edge_map)
    weight_b = region_b.float() * confidence_weight * smooth_weight

    metrics = {
        "region_A_ratio": region_a.float().mean().detach(),
        "region_B_ratio": region_b.float().mean().detach(),
    }
    return region_a, weight_b, metrics


def non_edge_depth_grad_mean(depth, smooth_mask):
    if depth.dim() == 4:
        depth = depth.squeeze(1)
    depth_dx = gradient_x(depth).abs()
    depth_dy = gradient_y(depth).abs()
    mask_x = smooth_mask[:, :, 1:] & smooth_mask[:, :, :-1]
    mask_y = smooth_mask[:, 1:, :] & smooth_mask[:, :-1, :]
    return 0.5 * (masked_mean(depth_dx, mask_x) + masked_mean(depth_dy, mask_y))


def scale_intrinsics(intrinsics, source_size, target_size):
    if source_size == target_size:
        return intrinsics
    scaled = intrinsics.clone()
    scale_y = float(target_size[0]) / float(source_size[0])
    scale_x = float(target_size[1]) / float(source_size[1])
    scaled[:, 0, :] *= scale_x
    scaled[:, 1, :] *= scale_y
    return scaled


def depth_normal_consistency_loss(normal_pred, depth_pred, intrinsics, ref_img, valid_mask,
                                  confidence, depth_normal_conf_threshold=0.8,
                                  edge_grad_threshold=0.05):
    target_size = depth_pred.shape[-2:]
    if normal_pred.shape[-2:] != target_size:
        normal_pred = F.interpolate(normal_pred, size=target_size, mode='bilinear', align_corners=False)
        normal_pred = F.normalize(normal_pred, p=2, dim=1, eps=1e-6)
    smooth_mask = build_smooth_mask(ref_img, valid_mask, confidence,
                                    depth_normal_conf_threshold,
                                    edge_grad_threshold, target_size)
    intrinsics = scale_intrinsics(intrinsics, ref_img.shape[-2:], target_size)
    normal_depth = compute_normal_from_depth(depth_pred, intrinsics)
    cos = torch.abs((normal_pred * normal_depth).sum(dim=1)).clamp(0.0, 1.0)
    loss = masked_mean(1.0 - cos, smooth_mask)
    metrics = {
        "normal_depth_cos": masked_mean(cos, smooth_mask).detach(),
        "smooth_mask_ratio": smooth_mask.float().mean().detach(),
        "non_edge_depth_grad_mean": non_edge_depth_grad_mean(depth_pred.detach(), smooth_mask).detach(),
    }
    return loss, metrics, smooth_mask


def masked_mean(value, mask):
    mask = mask.float()
    return (value * mask).sum() / (mask.sum() + 1e-6)


def normal_smooth_loss(depth, mask):
    normal = get_depth_normals(depth)
    mask_x = mask[:, :, 1:] & mask[:, :, :-1]
    mask_y = mask[:, 1:, :] & mask[:, :-1, :]
    loss_x = 1 - (normal[..., :, 1:] * normal[..., :, :-1]).sum(dim=1)
    loss_y = 1 - (normal[..., 1:, :] * normal[..., :-1, :]).sum(dim=1)
    return masked_mean(loss_x, mask_x) + masked_mean(loss_y, mask_y)


def curvature_loss(depth, mask):
    depth_mean = masked_mean(depth, mask).detach()
    depth_norm = depth / (depth_mean + 1e-6)
    curv_x = depth_norm[:, :, 2:] - 2 * depth_norm[:, :, 1:-1] + depth_norm[:, :, :-2]
    curv_y = depth_norm[:, 2:, :] - 2 * depth_norm[:, 1:-1, :] + depth_norm[:, :-2, :]
    mask_x = mask[:, :, 2:] & mask[:, :, 1:-1] & mask[:, :, :-2]
    mask_y = mask[:, 2:, :] & mask[:, 1:-1, :] & mask[:, :-2, :]
    return masked_mean(curv_x.abs(), mask_x) + masked_mean(curv_y.abs(), mask_y)


def weighted_mean(value, weight):
    return (value * weight).sum() / (weight.sum() + 1e-6)


def soft_curvature_loss(depth, weight):
    if depth.dim() == 4:
        depth = depth.squeeze(1)
    if weight.dim() == 4:
        weight = weight.squeeze(1)
    weight = weight.float()
    depth_mean = weighted_mean(depth, weight).detach()
    depth_norm = depth / (depth_mean + 1e-6)
    curv_x = depth_norm[:, :, 2:] - 2 * depth_norm[:, :, 1:-1] + depth_norm[:, :, :-2]
    curv_y = depth_norm[:, 2:, :] - 2 * depth_norm[:, 1:-1, :] + depth_norm[:, :-2, :]
    weight_x = torch.min(torch.min(weight[:, :, 2:], weight[:, :, 1:-1]), weight[:, :, :-2])
    weight_y = torch.min(torch.min(weight[:, 2:, :], weight[:, 1:-1, :]), weight[:, :-2, :])
    return weighted_mean(curv_x.abs(), weight_x) + weighted_mean(curv_y.abs(), weight_y)


def dual_region_curvature_loss(depth, region_a, weight_b, lambda_a=1.5, lambda_b=1.0):
    if depth.dim() == 4:
        depth = depth.squeeze(1)
    if region_a.dim() == 4:
        region_a = region_a.squeeze(1)
    if weight_b.dim() == 4:
        weight_b = weight_b.squeeze(1)
    region_a = region_a > 0.5
    weight_b = weight_b.float()
    valid_support = (region_a | (weight_b > 0.0)).float()
    depth_mean = weighted_mean(depth, valid_support).detach()
    depth_norm = depth / (depth_mean + 1e-6)
    curv_x = depth_norm[:, :, 2:] - 2 * depth_norm[:, :, 1:-1] + depth_norm[:, :, :-2]
    curv_y = depth_norm[:, 2:, :] - 2 * depth_norm[:, 1:-1, :] + depth_norm[:, :-2, :]
    region_a_x = region_a[:, :, 2:] & region_a[:, :, 1:-1] & region_a[:, :, :-2]
    region_a_y = region_a[:, 2:, :] & region_a[:, 1:-1, :] & region_a[:, :-2, :]
    weight_b_x = torch.min(torch.min(weight_b[:, :, 2:], weight_b[:, :, 1:-1]), weight_b[:, :, :-2])
    weight_b_y = torch.min(torch.min(weight_b[:, 2:, :], weight_b[:, 1:-1, :]), weight_b[:, :-2, :])
    loss_a = masked_mean(curv_x.abs(), region_a_x) + masked_mean(curv_y.abs(), region_a_y)
    loss_b = weighted_mean(curv_x.abs(), weight_b_x) + weighted_mean(curv_y.abs(), weight_b_y)
    return lambda_a * loss_a + lambda_b * loss_b, loss_a, loss_b


def edge_aware_smooth_loss(depth, ref_img, mask):
    ref_img = F.interpolate(ref_img, size=depth.shape[-2:], mode='bilinear', align_corners=False)
    depth_mean = masked_mean(depth, mask).detach()
    depth_norm = depth / (depth_mean + 1e-6)
    depth_dx = gradient_x(depth_norm).abs()
    depth_dy = gradient_y(depth_norm).abs()
    img_dx = gradient_x(ref_img).abs().mean(dim=1)
    img_dy = gradient_y(ref_img).abs().mean(dim=1)
    weight_x = torch.exp(-img_dx)
    weight_y = torch.exp(-img_dy)
    mask_x = mask[:, :, 1:] & mask[:, :, :-1]
    mask_y = mask[:, 1:, :] & mask[:, :-1, :]
    return masked_mean(depth_dx * weight_x, mask_x) + masked_mean(depth_dy * weight_y, mask_y)


def bounded_residual_target(depth_raw, depth_gt, depth_interval,
                            max_residual_ratio=0.25):
    """Build a detached, interval-bounded target for the SGER proposal."""
    interval = torch.as_tensor(
        depth_interval, dtype=depth_raw.dtype,
        device=depth_raw.device).detach()
    max_residual = float(max_residual_ratio) * interval
    return (depth_gt - depth_raw.detach()).clamp(
        min=-max_residual, max=max_residual).detach()


def residual_benefit_target(depth_raw, raw_residual, depth_gt,
                            depth_interval, margin_ratio=0.05):
    """Label proposals that reduce raw error by an interval-scaled margin."""
    raw_error = (depth_raw.detach() - depth_gt).abs()
    proposal_error = (
        depth_raw.detach() + raw_residual.detach() - depth_gt).abs()
    interval = torch.as_tensor(
        depth_interval, dtype=depth_raw.dtype,
        device=depth_raw.device).detach()
    margin = float(margin_ratio) * interval
    return (proposal_error + margin < raw_error).detach()


def balanced_binary_cross_entropy(prediction, target, mask):
    prediction = prediction.clamp(1e-6, 1.0 - 1e-6)
    target = target.float()
    valid_target = target[mask]
    valid_prediction = prediction[mask]
    if valid_target.numel() == 0:
        return prediction.sum() * 0.0
    positive_fraction = valid_target.mean().detach()
    if positive_fraction.item() <= 0.0 or positive_fraction.item() >= 1.0:
        return F.binary_cross_entropy(valid_prediction, valid_target)
    weights = torch.where(
        valid_target > 0.5,
        1.0 - positive_fraction,
        positive_fraction)
    losses = F.binary_cross_entropy(
        valid_prediction, valid_target, reduction="none")
    return (losses * weights).sum() / (weights.sum() + 1e-6)


def cas_mvsnet_loss(inputs, depth_gt_ms, mask_ms, **kwargs):
    depth_loss_weights = kwargs.get("dlossw", None)
    raw_depth_loss_weight = kwargs.get("raw_depth_loss_weight", 0.5)
    refined_depth_loss_weight = kwargs.get("refined_depth_loss_weight", 1.0)
    residual_loss_weight = kwargs.get("residual_loss_weight", 0.001)
    gate_loss_weight = kwargs.get("gate_loss_weight", 0.0)
    safe_refine_loss_weight = kwargs.get("safe_refine_loss_weight", 0.0)
    safe_refine_margin = kwargs.get("safe_refine_margin", 0.0)
    residual_target_loss_weight = kwargs.get(
        "residual_target_loss_weight", 0.0)
    gate_benefit_loss_weight = kwargs.get("gate_benefit_loss_weight", 0.0)
    residual_target_ratio = kwargs.get("residual_target_ratio", 0.25)
    benefit_margin_ratio = kwargs.get("benefit_margin_ratio", 0.05)
    residual_target_loss_scale = float(kwargs.get(
        "residual_target_loss_scale", 1.0))
    gate_benefit_loss_scale = float(kwargs.get(
        "gate_benefit_loss_scale", 1.0))
    sger_loss_scale = float(kwargs.get("sger_loss_scale", 1.0))
    if not math.isfinite(sger_loss_scale) or not 0.0 <= sger_loss_scale <= 1.0:
        raise ValueError("sger_loss_scale must be finite and within [0, 1]")
    effective_refined_depth_loss_weight = (
        sger_loss_scale * refined_depth_loss_weight)
    effective_residual_loss_weight = sger_loss_scale * residual_loss_weight
    effective_gate_loss_weight = sger_loss_scale * gate_loss_weight
    effective_safe_refine_loss_weight = (
        sger_loss_scale * safe_refine_loss_weight)
    effective_residual_target_loss_weight = (
        residual_target_loss_scale * residual_target_loss_weight)
    effective_gate_benefit_loss_weight = (
        gate_benefit_loss_scale * gate_benefit_loss_weight)
    imgs = kwargs.get("imgs", None)
    proj_matrices = kwargs.get("proj_matrices", None)
    normal_smooth_loss_weight = kwargs.get("normal_smooth_loss_weight", 0.0)
    curv_loss_weight = kwargs.get("curv_loss_weight", 0.0)
    edge_smooth_loss_weight = kwargs.get("edge_smooth_loss_weight", 0.0)
    depth_normal_loss_weight = kwargs.get("depth_normal_loss_weight", 0.0)
    depth_normal_conf_threshold = kwargs.get("depth_normal_conf_threshold", 0.8)
    edge_grad_threshold = kwargs.get("edge_grad_threshold", 0.05)
    geometry_conf_mid = kwargs.get("geometry_conf_mid", 0.65)
    geometry_k_conf = kwargs.get("geometry_k_conf", 10.0)
    geometry_edge_mid = kwargs.get("geometry_edge_mid", 0.25)
    geometry_k_edge = kwargs.get("geometry_k_edge", 10.0)
    geometry_w_min = kwargs.get("geometry_w_min", 0.05)
    use_dual_region_curvature = kwargs.get("use_dual_region_curvature", False)
    region_lambda_a = kwargs.get("region_lambda_a", 1.5)
    region_lambda_b = kwargs.get("region_lambda_b", 1.0)
    region_edge_threshold = kwargs.get("region_edge_threshold", 0.25)
    region_depth_threshold = kwargs.get("region_depth_threshold", 0.02)
    region_curv_threshold = kwargs.get("region_curv_threshold", 0.02)
    region_smooth_k = kwargs.get("region_smooth_k", 2.0)
    return_extra = kwargs.get("return_extra", False)

    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    extra = {}
    depth_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    normal_smooth_loss_final = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    curvature_loss_final = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    edge_aware_smooth_loss_final = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    curvature_loss_raw_final = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    curvature_loss_weighted_final = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    geometry_metric_accum = {}
    geometry_metric_count = 0
    region_metric_accum = {}
    region_metric_count = 0
    curv_loss_a_final = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    curv_loss_b_final = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    curv_loss_total_final = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)

    stage_keys = [k for k in inputs.keys()
                  if k.startswith("stage") and k.replace("stage", "").isdigit()]
    stage_keys = sorted(stage_keys, key=lambda k: int(k.replace("stage", "")))

    for stage_key in stage_keys:
        stage_inputs = inputs[stage_key]
        depth_est = stage_inputs["depth"]
        depth_raw = stage_inputs.get("depth_raw", None)
        depth_gt = depth_gt_ms[stage_key]
        mask = mask_ms[stage_key] > 0.5
        stage_idx = int(stage_key.replace("stage", "")) - 1

        refined_depth_loss = F.smooth_l1_loss(
            depth_est[mask], depth_gt[mask], reduction='mean')
        depth_loss = refined_depth_loss
        if depth_raw is not None:
            extra["sger_residual_scale"] = total_loss.new_tensor(
                sger_loss_scale)
            extra["effective_refined_depth_loss_weight"] = (
                total_loss.new_tensor(effective_refined_depth_loss_weight))
            extra["effective_residual_loss_weight"] = total_loss.new_tensor(
                effective_residual_loss_weight)
            extra["effective_gate_loss_weight"] = total_loss.new_tensor(
                effective_gate_loss_weight)
            extra["effective_safe_refine_loss_weight"] = (
                total_loss.new_tensor(effective_safe_refine_loss_weight))
            extra["effective_residual_target_loss_weight"] = (
                total_loss.new_tensor(effective_residual_target_loss_weight))
            extra["effective_gate_benefit_loss_weight"] = (
                total_loss.new_tensor(effective_gate_benefit_loss_weight))
            raw_depth_loss = F.smooth_l1_loss(
                depth_raw[mask], depth_gt[mask], reduction='mean')
            stage_depth_loss = (
                effective_refined_depth_loss_weight * refined_depth_loss
                + raw_depth_loss_weight * raw_depth_loss)
            extra["{}/raw_depth_loss".format(stage_key)] = raw_depth_loss.detach()
            extra["{}/refined_depth_loss".format(stage_key)] = refined_depth_loss.detach()
            raw_abs_error = masked_mean((depth_raw - depth_gt).abs(), mask)
            refined_abs_error = masked_mean((depth_est - depth_gt).abs(), mask)
            extra["{}/raw_abs_error".format(stage_key)] = (
                raw_abs_error.detach())
            extra["{}/refined_abs_error".format(stage_key)] = (
                refined_abs_error.detach())
            extra["{}/raw_to_refined_error_delta".format(stage_key)] = (
                raw_abs_error - refined_abs_error).detach()
            improved = (
                (depth_est - depth_gt).abs()
                < (depth_raw - depth_gt).abs())
            extra["{}/refined_improved_pixel_ratio".format(stage_key)] = (
                masked_mean(improved.float(), mask).detach())
            worsened = (
                (depth_est - depth_gt).abs()
                > (depth_raw - depth_gt).abs())
            extra["{}/refined_worsened_pixel_ratio".format(stage_key)] = (
                masked_mean(worsened.float(), mask).detach())
            if effective_safe_refine_loss_weight > 0:
                safe_refine_loss = masked_mean(
                    F.relu(
                        (depth_est - depth_gt).abs()
                        - (depth_raw - depth_gt).abs().detach()
                        + safe_refine_margin),
                    mask)
                total_loss += (
                    effective_safe_refine_loss_weight * safe_refine_loss)
                extra["{}/safe_refine_loss".format(stage_key)] = (
                    safe_refine_loss.detach())
        else:
            stage_depth_loss = refined_depth_loss

        if "depth_residual" in stage_inputs:
            extra["{}/mean_abs_depth_residual".format(stage_key)] = masked_mean(
                stage_inputs["depth_residual"].abs(), mask).detach()
        if "raw_depth_residual" in stage_inputs:
            extra["{}/mean_abs_raw_residual".format(stage_key)] = masked_mean(
                stage_inputs["raw_depth_residual"].abs(), mask).detach()
        if "geometry_gate" in stage_inputs:
            extra["{}/geometry_gate_mean".format(stage_key)] = masked_mean(
                stage_inputs["geometry_gate"], mask).detach()
            if effective_gate_loss_weight > 0:
                gate_loss = masked_mean(stage_inputs["geometry_gate"], mask)
                total_loss += effective_gate_loss_weight * gate_loss
                extra["{}/gate_loss".format(stage_key)] = gate_loss.detach()
        benefit_gate = stage_inputs.get(
            "benefit_gate", stage_inputs.get("geometry_gate", None))
        if benefit_gate is not None:
            extra["{}/benefit_gate_mean".format(stage_key)] = masked_mean(
                benefit_gate, mask).detach()
        raw_residual = stage_inputs.get("raw_depth_residual", None)
        depth_interval = stage_inputs.get("depth_interval", None)
        if (depth_raw is not None and raw_residual is not None
                and benefit_gate is not None and depth_interval is not None):
            residual_target = bounded_residual_target(
                depth_raw, depth_gt, depth_interval, residual_target_ratio)
            benefit_target = residual_benefit_target(
                depth_raw, raw_residual, depth_gt, depth_interval,
                benefit_margin_ratio)
            residual_target_loss = F.smooth_l1_loss(
                raw_residual[mask], residual_target[mask], reduction="mean")
            gate_benefit_loss = balanced_binary_cross_entropy(
                benefit_gate, benefit_target, mask)
            total_loss += (
                effective_residual_target_loss_weight
                * residual_target_loss)
            total_loss += (
                effective_gate_benefit_loss_weight * gate_benefit_loss)
            extra["{}/residual_target_loss".format(stage_key)] = (
                residual_target_loss.detach())
            extra["{}/gate_benefit_loss".format(stage_key)] = (
                gate_benefit_loss.detach())
            nonzero_target = residual_target.abs() > 1e-6
            sign_correct = (
                torch.sign(raw_residual) == torch.sign(residual_target))
            sign_mask = mask & nonzero_target
            extra["{}/residual_sign_accuracy".format(stage_key)] = (
                masked_mean(sign_correct.float(), sign_mask).detach())
            actual_improved = (
                (depth_est - depth_gt).abs()
                < (depth_raw - depth_gt).abs())
            actual_worsened = (
                (depth_est - depth_gt).abs()
                > (depth_raw - depth_gt).abs())
            extra["{}/gate_on_improved_mean".format(stage_key)] = (
                masked_mean(benefit_gate, mask & actual_improved).detach())
            extra["{}/gate_on_worsened_mean".format(stage_key)] = (
                masked_mean(benefit_gate, mask & actual_worsened).detach())
            predicted_benefit = benefit_gate >= 0.5
            true_positive = mask & predicted_benefit & benefit_target
            predicted_positive = mask & predicted_benefit
            target_positive = mask & benefit_target
            extra["{}/benefit_target_positive_ratio".format(stage_key)] = (
                masked_mean(benefit_target.float(), mask).detach())
            extra["{}/gate_precision".format(stage_key)] = (
                true_positive.float().sum()
                / (predicted_positive.float().sum() + 1e-6)).detach()
            extra["{}/gate_recall".format(stage_key)] = (
                true_positive.float().sum()
                / (target_positive.float().sum() + 1e-6)).detach()
        for uncertainty_key in (
                "probability_entropy", "depth_variance",
                "top1_top2_margin"):
            if uncertainty_key in stage_inputs:
                extra["{}/{}_mean".format(
                    stage_key, uncertainty_key)] = masked_mean(
                        stage_inputs[uncertainty_key], mask).detach()
        if "region_a" in stage_inputs:
            extra["{}/region_A_ratio".format(stage_key)] = masked_mean(
                stage_inputs["region_a"].float(), mask).detach()
        if "region_b_weight" in stage_inputs:
            extra["{}/region_B_weight_mean".format(stage_key)] = masked_mean(
                stage_inputs["region_b_weight"], mask).detach()

        if depth_loss_weights is not None:
            total_loss += depth_loss_weights[stage_idx] * stage_depth_loss
        else:
            total_loss += stage_depth_loss
        #  if int(stage_key.replace("stage", "")) > 2:
        geometry_mask = None
        geometry_weight = None
        if imgs is not None and "photometric_confidence" in stage_inputs:
            geometry_mask = build_smooth_mask(imgs[:, 0], mask, stage_inputs["photometric_confidence"],
                                              depth_normal_conf_threshold,
                                              edge_grad_threshold,
                                              depth_est.shape[-2:])
            extra["geometry_mask_ratio"] = geometry_mask.float().mean().detach()
            geometry_weight, geometry_metrics = build_geometry_weight(
                imgs[:, 0],
                mask,
                stage_inputs["photometric_confidence"],
                depth_est.shape[-2:],
                geometry_conf_mid,
                geometry_k_conf,
                geometry_edge_mid,
                geometry_k_edge,
                geometry_w_min)
            geometry_metric_count += 1
            for metric_key, metric_value in geometry_metrics.items():
                extra["{}/{}".format(stage_key, metric_key)] = metric_value
                if metric_key not in geometry_metric_accum:
                    geometry_metric_accum[metric_key] = metric_value
                else:
                    geometry_metric_accum[metric_key] = geometry_metric_accum[metric_key] + metric_value
        if normal_smooth_loss_weight > 0:
            normal_smooth_loss_final = normal_smooth_loss(depth_est, mask)
            total_loss += (normal_smooth_loss_weight/pow(2, stage_idx)) * normal_smooth_loss_final
            extra["normal_smooth_loss"] = normal_smooth_loss_final.detach()
        if curv_loss_weight > 0:
            curv_stage_weight = 0.5 * float(stage_idx + 1)
            if use_dual_region_curvature and imgs is not None and "photometric_confidence" in stage_inputs:
                region_a, weight_b, region_metrics = build_dual_region_geometry(
                    imgs[:, 0],
                    mask,
                    depth_est,
                    stage_inputs["photometric_confidence"],
                    depth_est.shape[-2:],
                    region_edge_threshold,
                    region_depth_threshold,
                    region_curv_threshold,
                    geometry_conf_mid,
                    geometry_k_conf,
                    region_smooth_k)
                curv_loss_total, curv_loss_a, curv_loss_b = dual_region_curvature_loss(
                    depth_est, region_a, weight_b, region_lambda_a, region_lambda_b)
                region_b = mask & (~region_a)
                acc_a = masked_mean((depth_est - depth_gt).abs().detach(), region_a)
                acc_b = masked_mean((depth_est - depth_gt).abs().detach(), region_b)
                total_loss += curv_loss_weight * curv_stage_weight * curv_loss_total
                curvature_loss_final = curvature_loss_final + curv_stage_weight * curv_loss_total
                curv_loss_a_final = curv_loss_a_final + curv_stage_weight * curv_loss_a
                curv_loss_b_final = curv_loss_b_final + curv_stage_weight * curv_loss_b
                curv_loss_total_final = curv_loss_total_final + curv_stage_weight * curv_loss_total
                stage_region_metrics = {
                    "curv_loss_A": curv_loss_a.detach(),
                    "curv_loss_B": curv_loss_b.detach(),
                    "curv_loss_total": curv_loss_total.detach(),
                    "acc_region_A": acc_a.detach(),
                    "acc_region_B": acc_b.detach(),
                }
                stage_region_metrics.update(region_metrics)
                region_metric_count += 1
                for metric_key, metric_value in stage_region_metrics.items():
                    extra["{}/{}".format(stage_key, metric_key)] = metric_value
                    if metric_key not in region_metric_accum:
                        region_metric_accum[metric_key] = metric_value
                    else:
                        region_metric_accum[metric_key] = region_metric_accum[metric_key] + metric_value
            else:
                curvature_loss_raw = curvature_loss(depth_est, mask)
                curvature_loss_weighted = soft_curvature_loss(depth_est, geometry_weight if geometry_weight is not None else mask.float())
                curvature_loss_final = curvature_loss_final + curv_stage_weight * curvature_loss_weighted
                curvature_loss_raw_final = curvature_loss_raw_final + curv_stage_weight * curvature_loss_raw
                curvature_loss_weighted_final = curvature_loss_weighted_final + curv_stage_weight * curvature_loss_weighted
                total_loss += curv_loss_weight * curv_stage_weight * curvature_loss_weighted
                extra["{}/curvature_loss_raw".format(stage_key)] = curvature_loss_raw.detach()
                extra["{}/curvature_loss_weighted".format(stage_key)] = curvature_loss_weighted.detach()
                extra["{}/curv_loss".format(stage_key)] = curvature_loss_weighted.detach()
        if edge_smooth_loss_weight > 0 and imgs is not None:
            edge_aware_smooth_loss_final = edge_aware_smooth_loss(depth_est, imgs[:, 0], mask)
            total_loss += (edge_smooth_loss_weight/pow(2, stage_idx)) * edge_aware_smooth_loss_final
            extra["edge_smooth_loss"] = edge_aware_smooth_loss_final.detach()
        if (effective_residual_loss_weight > 0
                and "residual_ratio" in stage_inputs):
            residual_loss = masked_mean(stage_inputs["residual_ratio"], mask)
            total_loss += (
                effective_residual_loss_weight / pow(2, stage_idx)
            ) * residual_loss
            extra["{}/residual_loss".format(stage_key)] = residual_loss.detach()
        if (depth_normal_loss_weight > 0 and imgs is not None and proj_matrices is not None
                and stage_key in proj_matrices
                and "normal" in stage_inputs and "photometric_confidence" in stage_inputs):
            intrinsics = proj_matrices[stage_key][:, 0, 1, :3, :3]
            loss_depth_normal, dn_metrics, smooth_mask = depth_normal_consistency_loss(
                stage_inputs["normal"],
                depth_est,
                intrinsics,
                imgs[:, 0],
                mask,
                stage_inputs["photometric_confidence"],
                depth_normal_conf_threshold,
                edge_grad_threshold)
            total_loss += (depth_normal_loss_weight / pow(2, stage_idx)) * loss_depth_normal
            extra["depth_normal_loss"] = loss_depth_normal.detach()
            extra.update(dn_metrics)
            extra["smooth_mask"] = smooth_mask.detach()

    if geometry_metric_count > 0:
        for metric_key, metric_value in geometry_metric_accum.items():
            extra[metric_key] = (metric_value / geometry_metric_count).detach()
    if curv_loss_weight > 0:
        extra["curv_loss"] = curvature_loss_final.detach()
        if use_dual_region_curvature and region_metric_count > 0:
            for metric_key, metric_value in region_metric_accum.items():
                if metric_key in ("curv_loss_A", "curv_loss_B", "curv_loss_total"):
                    continue
                extra[metric_key] = (metric_value / region_metric_count).detach()
            extra["curv_loss_A"] = curv_loss_a_final.detach()
            extra["curv_loss_B"] = curv_loss_b_final.detach()
            extra["curv_loss_total"] = curv_loss_total_final.detach()
        else:
            extra["curvature_loss_raw"] = curvature_loss_raw_final.detach()
            extra["curvature_loss_weighted"] = curvature_loss_weighted_final.detach()

    if return_extra:
            return total_loss, depth_loss, normal_smooth_loss_final, curvature_loss_final, edge_aware_smooth_loss_final, extra
    return total_loss, depth_loss, normal_smooth_loss_final, curvature_loss_final, edge_aware_smooth_loss_final


def get_cur_depth_range_samples(cur_depth, ndepth, depth_inteval_pixel, shape, max_depth=192.0, min_depth=0.0):
    #shape, (B, H, W)
    #cur_depth: (B, H, W)
    #return depth_range_values: (B, D, H, W)
    cur_depth_min = (cur_depth - ndepth / 2 * depth_inteval_pixel)  # (B, H, W)
    cur_depth_max = (cur_depth + ndepth / 2 * depth_inteval_pixel)
    # cur_depth_min = (cur_depth - ndepth / 2 * depth_inteval_pixel).clamp(min=0.0)   #(B, H, W)
    # cur_depth_max = (cur_depth_min + (ndepth - 1) * depth_inteval_pixel).clamp(max=max_depth)

    assert cur_depth.shape == torch.Size(shape), "cur_depth:{}, input shape:{}".format(cur_depth.shape, shape)
    new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1)  # (B, H, W)

    depth_range_samples = cur_depth_min.unsqueeze(1) + (torch.arange(0, ndepth, device=cur_depth.device,
                                                                  dtype=cur_depth.dtype,
                                                                  requires_grad=False).reshape(1, -1, 1,
                                                                                               1) * new_interval.unsqueeze(1))

    return depth_range_samples


def get_depth_range_samples(cur_depth, ndepth, depth_inteval_pixel, device, dtype, shape,
                           max_depth=192.0, min_depth=0.0):
    #shape: (B, H, W)
    #cur_depth: (B, H, W) or (B, D)
    #return depth_range_samples: (B, D, H, W)
    if cur_depth.dim() == 2:
        cur_depth_min = cur_depth[:, 0]  # (B,)
        cur_depth_max = cur_depth[:, -1]
        new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1)  # (B, )

        depth_range_samples = cur_depth_min.unsqueeze(1) + (torch.arange(0, ndepth, device=device, dtype=dtype,
                                                                       requires_grad=False).reshape(1, -1) * new_interval.unsqueeze(1)) #(B, D)

        depth_range_samples = depth_range_samples.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, shape[1], shape[2]) #(B, D, H, W)

    else:

        depth_range_samples = get_cur_depth_range_samples(cur_depth, ndepth, depth_inteval_pixel, shape, max_depth, min_depth)

    return depth_range_samples



if __name__ == "__main__":
    # some testing code, just IGNORE it
    import sys
    sys.path.append("../")
    from datasets import find_dataset_def
    from torch.utils.data import DataLoader
    import numpy as np
    import cv2
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    # MVSDataset = find_dataset_def("colmap")
    # dataset = MVSDataset("../data/results/ford/num10_1/", 3, 'test',
    #                      128, interval_scale=1.06, max_h=1250, max_w=1024)

    MVSDataset = find_dataset_def("dtu_yao")
    num_depth = 48
    dataset = MVSDataset("../data/DTU/mvs_training/dtu/", '../lists/dtu/train.txt', 'train',
                         3, num_depth, interval_scale=1.06 * 192 / num_depth)

    dataloader = DataLoader(dataset, batch_size=1)
    item = next(iter(dataloader))

    imgs = item["imgs"][:, :, :, ::4, ::4]  #(B, N, 3, H, W)
    # imgs = item["imgs"][:, :, :, :, :]
    proj_matrices = item["proj_matrices"]   #(B, N, 2, 4, 4) dim=N: N view; dim=2: index 0 for extr, 1 for intric
    proj_matrices[:, :, 1, :2, :] = proj_matrices[:, :, 1, :2, :]
    # proj_matrices[:, :, 1, :2, :] = proj_matrices[:, :, 1, :2, :] * 4
    depth_values = item["depth_values"]     #(B, D)

    imgs = torch.unbind(imgs, 1)
    proj_matrices = torch.unbind(proj_matrices, 1)
    ref_img, src_imgs = imgs[0], imgs[1:]
    ref_proj, src_proj = proj_matrices[0], proj_matrices[1:][0]  #only vis first view

    src_proj_new = src_proj[:, 0].clone()
    src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
    ref_proj_new = ref_proj[:, 0].clone()
    ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])

    warped_imgs = homo_warping(src_imgs[0], src_proj_new, ref_proj_new, depth_values)

    ref_img_np = ref_img.permute([0, 2, 3, 1])[0].detach().cpu().numpy()[:, :, ::-1] * 255
    cv2.imwrite('../tmp/ref.png', ref_img_np)
    cv2.imwrite('../tmp/src.png', src_imgs[0].permute([0, 2, 3, 1])[0].detach().cpu().numpy()[:, :, ::-1] * 255)

    for i in range(warped_imgs.shape[2]):
        warped_img = warped_imgs[:, :, i, :, :].permute([0, 2, 3, 1]).contiguous()
        img_np = warped_img[0].detach().cpu().numpy()
        img_np = img_np[:, :, ::-1] * 255

        alpha = 0.5
        beta = 1 - alpha
        gamma = 0
        img_add = cv2.addWeighted(ref_img_np, alpha, img_np, beta, gamma)
        cv2.imwrite('../tmp/tmp{}.png'.format(i), np.hstack([ref_img_np, img_np, img_add])) #* ratio + img_np*(1-ratio)]))
