import os
import torch
from torch.utils.data import DataLoader
from depth_project.dataset_sequence import SequenceDataset
from depth_project.models.depth_net import DepthNet
from depth_project.models.pose_net import PoseNet
from depth_project.losses import (
    SSIM, disp_to_depth, get_smooth_loss,
    transformation_from_parameters, photometric_reprojection,
    contrastive_loss,
)


def main():
    data_dir = os.path.expanduser('~/depth_selfsup_data')
    ckpt_dir = os.path.expanduser('~/ros2_ws/src/depth_project/checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f'data_dir = {data_dir}')
    print(f'ckpt_dir = {ckpt_dir}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device = {device}')

    dataset = SequenceDataset(data_dir, height=192, width=320)
    print(f'dataset length = {len(dataset)}')

    if len(dataset) == 0:
        raise RuntimeError('Dataset is empty. No frame pairs found.')

    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    print(f'number of batches = {len(loader)}')

    depth_net = DepthNet().to(device)
    pose_net = PoseNet().to(device)
    ssim = SSIM().to(device)

    optim = torch.optim.Adam(
        list(depth_net.parameters()) + list(pose_net.parameters()),
        lr=1e-4
    )

    epochs = 3

    for epoch in range(epochs):
        print(f'--- epoch {epoch+1}/{epochs} ---')
        depth_net.train()
        pose_net.train()
        running = 0.0

        for i, batch in enumerate(loader):
            tgt = batch['target'].to(device)
            src = batch['source'].to(device)
            K = batch['K'].to(device)
            inv_K = batch['inv_K'].to(device)

            disp, z_t = depth_net(tgt)
            _, z_s = depth_net(src)

            depth = disp_to_depth(disp)
            axisangle, translation = pose_net(tgt, src)
            T = transformation_from_parameters(axisangle, translation)

            photo_loss, _ = photometric_reprojection(tgt, src, depth, K, inv_K, T, ssim)
            smooth_loss = get_smooth_loss(disp, tgt)
            cont_loss = contrastive_loss(z_t, z_s)

            loss = photo_loss + 1e-3 * smooth_loss + 0.05 * cont_loss

            optim.zero_grad()
            loss.backward()
            optim.step()

            running += loss.item()

            if (i + 1) % 50 == 0:
                print(f'  batch {i+1}/{len(loader)} loss={loss.item():.6f}')

        avg = running / max(len(loader), 1)
        print(f'epoch {epoch+1} average loss = {avg:.6f}')

        ckpt_path = os.path.join(ckpt_dir, 'selfsup_depth_latest.pth')
        torch.save({
            'depth_net': depth_net.state_dict(),
            'pose_net': pose_net.state_dict(),
            'epoch': epoch + 1,
        }, ckpt_path)
        print(f'saved checkpoint: {ckpt_path}')

    print('training finished')


if __name__ == '__main__':
    main()
