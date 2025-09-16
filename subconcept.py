import argparse
import os

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageNet
from tqdm import tqdm
import numpy as np
from PIL import Image

from text_set_projection import get_text_features
from utils.factory import create_model_and_transforms
from hook_collect import hook_register_neurons_heads
from utils.visualization import show_image


def parse_arguments():
    parser = argparse.ArgumentParser(description="Retrieve top-norm images for a component")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_name', type=str, default='RN50x16')
    parser.add_argument('--num_heads', type=int, default=48)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--sample_stride', type=int, default=1)
    parser.add_argument('--neur_idxs', nargs='+')
    parser.add_argument('--save_dir', type=str)
    return parser.parse_args()


@torch.no_grad()
def main(args):
    model, _, val_preprocess = create_model_and_transforms(args.model_name, pretrained='openai', device=args.device)
    model.visual.attnpool.forward_mode = 'per_head_per_neuron'
    nh_hook = hook_register_neurons_heads(model)

    dataset = ImageNet(root='data', split='val', transform=val_preprocess)
    dataset = Subset(dataset, list(range(0, len(dataset), args.sample_stride)))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    for neur_idx in args.neur_idxs:
        neur_save_mat = np.empty(len(dataset), dtype=np.float32)
        neur_head_save_mat = np.empty((args.num_heads, len(dataset)), dtype=np.float32)
        for im_idx, (images, labels) in enumerate(tqdm(dataloader)):
            model.encode_image(images.to(args.device))
            neur_out = nh_hook.get_out_mat()[:, :, :, neur_idx].sum(dim=1).norm(dim=-1)  # BHOC -> B
            neur_head_out = nh_hook.get_out_mat()[:, :, :, neur_idx].norm(dim=-1)  # BHOC -> BH
            start = im_idx * args.batch_size
            end = start + args.batch_size
            neur_save_mat[start:end] = neur_out.cpu().numpy()
            neur_head_save_mat[:, start:end] = neur_head_out.transpose(-2, -1).cpu().numpy()

        k = 10
        neur_indices = np.argsort(neur_save_mat)[-k:][::-1]
        neur_save_dir = f'{args.save_dir}/{neur_idx}'
        if not os.path.exists(neur_save_dir):
            os.makedirs(neur_save_dir)
        for loop_idx, real_idx in enumerate(neur_indices):
            show_image(dataset[real_idx], 0, f'{neur_save_dir}/{loop_idx}_{real_idx}.png', show_im=False)
        for head_idx in range(args.num_heads):
            neur_head_indices = np.argsort(neur_head_save_mat[head_idx])[-k:][::-1]
            neur_head_save_dir = f'{args.save_dir}/{neur_idx}/{neur_idx}_{head_idx}'
            if not os.path.exists(neur_head_save_dir):
                os.makedirs(neur_head_save_dir)
            for loop_idx, real_idx in enumerate(neur_head_indices):
                show_image(dataset[real_idx], 0, f'{neur_head_save_dir}/{loop_idx}_{real_idx}.png', show_im=False)
        

if __name__ == "__main__":
    args = parse_arguments()
    main(args)  