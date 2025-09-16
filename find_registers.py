import argparse

import torch
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from utils.factory import create_model_and_transforms
from utils.get_dims import get_dims
from utils.test_dataset import ImageNetTest
from utils.cars_dataset import CustomStanfordCars
from hook_collect import hook_register_neurons_heads, hook_register_attention
from hook_replace import hook_register_attn_slice_last_column, hook_register_neurons_zero


def parse_arguments():
    parser = argparse.ArgumentParser(description="Visualize token similarity")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_name', type=str, default='RN50x16')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='cars_test')
    parser.add_argument('--sample_stride', type=int, default=11)
    parser.add_argument('--mode', type=str, default='activations', choices=['activations', 'contributions'])  
    parser.add_argument('--diff_save_path', type=str, default='my_caches/attn_stuff/x4_sink_diff_D=1000.npy')
    parser.add_argument('--load', type=bool, default=True)
    return parser.parse_args()


@torch.no_grad()
def main(args):
    model_zero, _, val_preprocess = create_model_and_transforms(args.model_name, pretrained='openai', device=args.device)
    model_zero.eval()
    model_zero.to(args.device)
    model_dflt, _, val_preprocess = create_model_and_transforms(args.model_name, pretrained='openai', device=args.device)
    model_dflt.eval()
    model_dflt.to(args.device)
    if args.dataset == 'imagenet_val':
        dataset = ImageNet(root='data', split='val', transform=val_preprocess)
    elif args.dataset == 'imagenet_test':
        dataset = ImageNetTest(folder='test_data/test', transform=val_preprocess)
    elif args.dataset == 'cars_test':
        dataset = CustomStanfordCars(root='cars', split='test', transform=val_preprocess)
    dataset = Subset(dataset, list(range(0, len(dataset), args.sample_stride)))
    assert len(dataset) % args.batch_size == 0
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    num_heads, num_tokens, embed_dim, out_dim = get_dims(model_zero)

    if args.mode == 'activations':  # zero out neuron activations and check sink difference
        model_zero.visual.attnpool.forward_mode = 'per_head_no_spatial'
        model_dflt.visual.attnpool.forward_mode = 'per_head_no_spatial'
        replace_hook_zero = hook_register_neurons_zero(model_zero, args.device)
        collect_hook_zero = hook_register_attention(model_zero, 'post')
        collect_hook_dflt = hook_register_attention(model_dflt, 'post')
        save_mat = np.zeros(shape=(len(dataset) // args.batch_size, embed_dim), dtype=np.float32)
    elif args.mode == 'contributions':  # zero out sink and check (neuron, head) contribution difference
        model_zero.visual.attnpool.forward_mode = 'per_head_per_neuron'
        model_dflt.visual.attnpool.forward_mode = 'per_head_per_neuron'
        hook_register_attn_slice_last_column(model_zero, args.device, col=torch.zeros(num_tokens).to(args.device))
        collect_hook_zero = hook_register_neurons_heads(model_zero, args.device)  
        collect_hook_dflt = hook_register_neurons_heads(model_dflt, args.device)
        save_mat = np.zeros(shape=(num_heads, embed_dim), dtype=np.float32)

    if not args.load:
        for batch_idx, itms in enumerate(tqdm(dataloader, desc=f"Finding difference in norms when zeroing out!")):
            images = itms[0] if isinstance(itms, tuple) or isinstance(itms, list) else itms
            if args.mode == 'activations':
                for neuron in tqdm(range(embed_dim), "Inner loop!"):
                    mask = torch.ones(embed_dim, dtype=torch.float32, device=args.device)
                    mask[neuron] = 0.0
                    replace_hook_zero.set_neurons_zero_mask(mask)
                    model_zero.encode_image(images.to(args.device))
                    model_dflt.encode_image(images.to(args.device))
                    sink_zero = collect_hook_zero.get_out_mat()[:, :, 0, -1].mean(dim=1)  # BHNN -> B1
                    sink_dflt = collect_hook_dflt.get_out_mat()[:, :, 0, -1].mean(dim=1)
                    diff = (sink_dflt - sink_zero).cpu().numpy()  # higher diff means neuron reduces sink more
                    start = batch_idx * args.batch_size
                    end = start + args.batch_size
                    save_mat[start:end, neuron] = diff
            elif args.mode == 'contributions':
                model_zero.encode_image(images.to(args.device))  # post-softmax attn last column is zeroed out
                model_dflt.encode_image(images.to(args.device))
                out_zero = collect_hook_zero.get_out_mat()  # BHOC
                out_dflt = collect_hook_dflt.get_out_mat()
                diff = (out_zero.norm(dim=-2) - out_dflt.norm(dim=-2)).abs().cpu().numpy()
                save_mat += diff.sum(axis=0) / len(dataset)  # HC
        np.save(args.diff_save_path, save_mat)
    else:
        if args.mode == 'activations':
            diff_mat = torch.from_numpy(np.load(args.diff_save_path))
            print(f"Diff mat of shape {diff_mat.shape}.")
            mean_diff = diff_mat.abs().mean(axis=0)  # compute abs as neurons can be set to zero from both directions (pos embed)
            print(torch.topk(mean_diff, 50))


if __name__ == "__main__":
    args = parse_arguments()
    main(args)