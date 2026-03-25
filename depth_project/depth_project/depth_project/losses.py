import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIM(nn.Module):
    def __init__(self):
        super().__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)
        self.refl = nn.ReflectionPad2d(1)
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)
        sigma_x = self.sig_x_pool(x * x) - mu_x * mu_x
        sigma_y = self.sig_y_pool(y * y) - mu_y * mu_y
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y
        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x * mu_x + mu_y * mu_y + self.C1) * (sigma_x + sigma_y + self.C2)
        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def disp_to_depth(disp, min_depth=0.1, max_depth=20.0):
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return depth


def get_smooth_loss(disp, img):
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


def rot_from_axisangle(vec):
    angle = torch.norm(vec, 2, 1, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[:, 0:1]
    y = axis[:, 1:2]
    z = axis[:, 2:3]

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4), device=vec.device)
    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1
    return rot


def transformation_from_parameters(axisangle, translation):
    T = torch.zeros((axisangle.shape[0], 4, 4), device=axisangle.device)
    T[:, 3, 3] = 1
    R = rot_from_axisangle(axisangle)
    T[:, :3, :3] = R[:, :3, :3]
    T[:, :3, 3] = translation
    return T


def backproject(depth, inv_K):
    B, _, H, W = depth.shape
    device = depth.device

    meshgrid = torch.meshgrid(
        torch.arange(W, device=device),
        torch.arange(H, device=device),
        indexing='xy'
    )
    id_coords = torch.stack(meshgrid, dim=0).float()
    ones = torch.ones(B, 1, H * W, device=device)
    pix_coords = torch.unsqueeze(torch.stack([
        id_coords[0].reshape(-1),
        id_coords[1].reshape(-1)
    ], 0), 0).repeat(B, 1, 1)

    cam_points = torch.matmul(inv_K[:, :3, :3], torch.cat([pix_coords, ones], 1))
    cam_points = depth.view(B, 1, -1) * cam_points
    cam_points = torch.cat([cam_points, ones], 1)
    return cam_points


def project(cam_points, K, T, H, W):
    P = torch.matmul(K, T)[:, :3, :]
    cam_points = torch.matmul(P, cam_points)

    pix_coords = cam_points[:, :2, :] / (cam_points[:, 2:3, :] + 1e-7)
    pix_coords = pix_coords.view(-1, 2, H, W)
    pix_coords = pix_coords.permute(0, 2, 3, 1)

    pix_coords[..., 0] /= (W - 1)
    pix_coords[..., 1] /= (H - 1)
    pix_coords = (pix_coords - 0.5) * 2
    return pix_coords


def photometric_reprojection(tgt, src, depth, K, inv_K, T, ssim_fn):
    B, _, H, W = tgt.shape
    cam_points = backproject(depth, inv_K)
    pix_coords = project(cam_points, K, T, H, W)
    warped = F.grid_sample(src, pix_coords, padding_mode='border', align_corners=True)

    l1 = torch.abs(tgt - warped).mean(1, True)
    ssim = ssim_fn(tgt, warped).mean(1, True)
    reprojection = 0.85 * ssim + 0.15 * l1
    return reprojection.mean(), warped


def contrastive_loss(z_t, z_s, temperature=0.1):
    logits = torch.matmul(z_t, z_s.t()) / temperature
    labels = torch.arange(z_t.shape[0], device=z_t.device)
    return F.cross_entropy(logits, labels)
