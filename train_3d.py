import os
import random

import torch
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.DSTformer import DSTformer
from utils.wild_dataset import MixDataset, load_target

motion = ['bboy_uprock_start', 'breakdance', 'dancing_twerk', 'dodging_right', 'goalkeeper_catch', 'hiphop_dancing',
          'joyful_jump', 'jump', 'twist_dance']
names = ['abe', 'brian', 'drake', 'elizabeth', 'james', 'leonard', 'lewis', 'maynard', 'megan', 'remy', 'shannon',
         'sophie']


def train_epoch(num_epochs, model_pos, train_loader, optimizer, criterion, scheduler):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_pos.train()
    model_pos = model_pos.to(device)
    # criterion = criterion.to(device)
    # model_pos.load_state_dict(torch.load('best.pth', map_location=device))
    min_loss = 100000
    # file_path = "training_losses.txt"
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        avg_loss = 0.0
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}', unit='batches') as pbar:
            for idx, (batch_input, batch_gt) in enumerate(train_loader):
                batch_input = batch_input.to(device)
                targets = []
                standards = []
                for name in batch_gt:
                    random_motion = random.choice(motion)
                    random_name = random.choice(names)
                    standard = load_target(os.path.join('datasets/video', random_motion, random_name, 'output_3D',
                                                        'output_keypoints_3d.npz'))
                    target = load_target(os.path.join('datasets/video', random_motion, name, 'output_3D',
                                                      'output_keypoints_3d.npz'))
                    standards.append(standard)
                    targets.append(target)
                standards = torch.cat(standards, 0)
                targets = torch.cat(targets, 0)
                standards = standards.to(device)
                targets = targets.to(device)
                outputs = model_pos(standards, batch_input)
                loss = criterion(outputs, targets)

                optimizer.zero_grad()  # 清除之前的梯度
                loss.backward()  # 反向传播
                optimizer.step()  # 更新参数
                epoch_loss += loss.item()
                avg_loss = epoch_loss / (idx + 1)
                pbar.set_postfix(loss=f'{avg_loss:.10f}', lr=f'{scheduler.get_last_lr()[0]:.10f}',
                                 min_loss=f'{min_loss:.10f}')
                pbar.update(1)
            if avg_loss < min_loss:
                min_loss = avg_loss
                torch.save(model_pos.state_dict(), 'best.pth')
                # with open(file_path, "a") as file:
                #     file.write(
                #         f"Epoch {epoch + 1}:loss={avg_loss:.3f},lr={scheduler.get_last_lr()[0]:.5f}, min_loss={min_loss:.3f}\n")
            # scheduler.step()


def loss_mpjpe(predicted, target):
    assert predicted.shape == target.shape
    predicted_3d = predicted[:, :, :, :3]
    target_3d = target[:, :, :, :3]
    diff = (predicted_3d - target_3d)
    return torch.mean(torch.norm(diff, dim=-1))


if __name__ == "__main__":
    model_backbone = DSTformer()
    num_parameters = sum(p.numel() for p in model_backbone.parameters())
    print(f"Number of parameters: {num_parameters}")

    pose_dataset = MixDataset(inputs_dir='datasets/video/stands')

    data_loader = DataLoader(pose_dataset, batch_size=4, shuffle=True)

    optimizer = optim.Adam(model_backbone.parameters(), lr=0.001)
    criterion = loss_mpjpe
    num_epochs = 300
    scheduler = CosineAnnealingLR(optimizer, T_max=100)
    people_names = []

    train_epoch(num_epochs, model_backbone, data_loader, optimizer, criterion, scheduler)
