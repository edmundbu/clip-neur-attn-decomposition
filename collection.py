import os
import argparse

import torch
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import numpy as np
from tqdm import tqdm

from utils.factory import create_model_and_transforms
from utils.test_dataset import ImageNetTest
from utils.get_dims import get_dims
from hook_collect import hook_register_attention, hook_register_neurons_heads, hook_register_tokens, hook_register_conv_maps


def parse_arguments():
    parser = argparse.ArgumentParser(description="Collect various contributions from various components")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_name', type=str, default='RN50x16')
    parser.add_argument('--dataset', type=str, default='imagenet_test')
    parser.add_argument('--data_path', type=str, default='test_data/test')
    parser.add_argument('--sample_stride', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1)
    # (neuron reprs can also be obtained by summing over neuron-head reprs)
    parser.add_argument('--mode', type=str, default='neurons+heads', choices=['attns', 'neuron_acts', 'neurons', 'heads', 'neurons+heads', 
                                                                              'neurons+heads_lens', 'layer4'])  
    parser.add_argument('--mean', type=bool, default=False)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--lens_idx', type=int, default=0)
    parser.add_argument('--num_blocks', type=int, default=8)
    return parser.parse_args()


@torch.no_grad()
def main(args):
    model, _, val_preprocess = create_model_and_transforms(args.model_name, pretrained='openai', device=args.device)
    model.eval()
    num_heads, num_tokens, embed_dim, out_dim = get_dims(model)
    print(f"{num_heads} heads, {embed_dim} neurons, {out_dim} out dim for {args.model_name}.")

    if args.dataset == 'imagenet_test':
        dataset = ImageNetTest(folder=args.data_path, transform=val_preprocess)
    elif args.dataset == 'imagenet_val':
        dataset = ImageNet(root=args.data_path, split='val', transform=val_preprocess)

    dataset = Subset(dataset, list(range(0, len(dataset), args.sample_stride)))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)
    assert len(dataset) % args.batch_size == 0

    if args.mode == 'attns':
        model.visual.attnpool.forward_mode = 'per_head_no_spatial'
        shape = (num_heads, num_tokens, num_tokens)
        hook = hook_register_attention(model, 'post')
    elif args.mode == 'neuron_acts':
        model.visual.attnpool.forward_mode = 'per_head'
        shape = (embed_dim, num_tokens) if args.mean else (len(dataset), embed_dim, num_tokens)
        hook = hook_register_tokens(model, mode='input')
    elif args.mode == 'neurons':
        model.visual.attnpool.forward_mode = 'per_head_per_neuron'
        shape = (embed_dim, out_dim) if args.mean else (len(dataset), embed_dim, out_dim)
        hook = hook_register_neurons_heads(model)
    elif args.mode == 'heads':
        model.visual.attnpool.forward_mode = 'per_head'
        shape = (len(dataset), num_heads, out_dim)
        hook = hook_register_tokens(model, mode='cls_spatial', collapse_heads=False) 
    elif args.mode == 'neurons+heads':
        model.visual.attnpool.forward_mode = 'per_head_per_neuron'
        shape = (num_heads, embed_dim, out_dim) if args.mean else (len(dataset), num_heads, embed_dim, out_dim)
        hook = hook_register_neurons_heads(model)
    elif args.mode == 'neurons+heads_lens':
        shape = (len(dataset), num_heads, embed_dim, out_dim)
        conv_hook = hook_register_conv_maps(model, embed_dim)
        hook = hook_register_neurons_heads(model)
    elif args.mode == 'layer4':
        assert args.mean is True
        shape = (2, args.num_blocks, num_tokens)
        conv_hook = hook_register_conv_maps(model, embed_dim)
        model.visual.attnpool.forward_mode = 'per_head'
        hook = hook_register_tokens(model, mode='cls_spatial', collapse_heads=True)

    ext = os.path.splitext(args.save_path)[1]
    if ext == '.npy':
        print("Saving as .npy!")
        save_mat = np.zeros(shape)
    elif ext == '.dat':
        print("Saving as .dat!")
        if os.path.exists(args.save_path):
            print(f"{args.save_path} already exists! Manually delete to overwrite.")
            return
        else:
            save_mat = np.memmap(args.save_path, dtype=np.float32, mode='w+', shape=shape)

    for idx, itms in enumerate(tqdm(dataloader, desc=f"Collecting {args.mode} samples over {args.dataset}!")):
        images = itms[0] if (isinstance(itms, list)) else itms
        if args.mode == 'neurons+heads_lens' or args.mode == 'layer4':
            model.visual.attnpool.forward_mode = 'skip'
        model.encode_image(images.to(args.device))
        start = idx * args.batch_size
        end = start + args.batch_size

        if args.mode == 'attns':
            if args.mean:
                save_mat += np.sum(hook.get_out_mat(as_np=True), axis=0) / len(dataset)
        elif args.mode == 'neuron_acts':
            assert args.mean is False
            start = idx * args.batch_size
            end = start + args.batch_size
            save_mat[start:end, :, :] = hook.get_out_mat(as_np=True).transpose(0, 2, 1)  # BNC -> BCN
        elif args.mode == 'neurons':
            if args.mean:
                save_mat += hook.get_out_mat().sum(dim=[0, 1]) / len(dataset)
            else:
                out = hook.get_out_mat(as_np=True).transpose(0, 1, 3, 2).sum(axis=1)  
                save_mat[start:end, ...] = out
        elif args.mode == 'heads':
            assert args.mean is False
            out = hook.get_out_mat().sum(dim=1)  # BMHO -> BHO
            save_mat[start:end, ...] = out.cpu().numpy()
        elif args.mode == 'neurons+heads':
            if args.mean:
                save_mat += hook.get_out_mat().sum(dim=0) / len(dataset)
            else:
                out = hook.get_out_mat(as_np=True).transpose(0, 1, 3, 2)  # BHCO
                save_mat[start:end, ...] = out
        elif args.mode == 'neurons+heads_lens':
            conv_maps = conv_hook.get_out_mat()
            conv_hook.reset_conv_maps()
            model.visual.attnpool.forward_mode = 'per_head_per_neuron'
            model.visual.attnpool(conv_maps[args.lens_idx])
            out = hook.get_out_mat(as_np=True).transpose(0, 1, 3, 2)  # BHCO
            save_mat[start:end, ...] = out
        elif args.mode == 'layer4':
            conv_maps = conv_hook.get_out_mat()  # BCHW
            conv_hook.reset_conv_maps()
            model.visual.attnpool.forward_mode = 'per_head'
            for j in range(args.num_blocks):
                conv_map = conv_maps[j]
                save_conv = conv_map.reshape(args.batch_size, embed_dim, -1).permute(0, 2, 1)
                save_conv = torch.cat([torch.full((args.batch_size, 1, embed_dim), float('inf'), 
                                                 device=save_conv.device), save_conv], dim=1)  # BNC
                save_mat[0, j] += (save_conv.norm(dim=-1).sum(dim=0) / len(dataset)).cpu().numpy()
                model.visual.attnpool(conv_map)
                out = hook.get_out_mat(as_np=False)  # BNO
                save_mat[1, j] += (out.norm(dim=-1).sum(dim=0) / len(dataset)).cpu().numpy()

    if ext == '.npy':
        np.save(args.save_path, save_mat)
    elif ext == '.dat':
        save_mat.flush()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
