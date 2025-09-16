import argparse
import json
import itertools

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageNet
from tqdm import tqdm
from scipy.stats import kurtosis
import numpy as np
import matplotlib.pyplot as plt


def parse_arguments():
    parser = argparse.ArgumentParser(description="Find the top text for a given direction")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_name', type=str, default='RN50x16')
    parser.add_argument('--mode', type=str, default='n', choices=['n', 'nh'])
    parser.add_argument('--all_norms_path', type=str, default='huge_cache/nh_val.dat')
    parser.add_argument('--mask_path', type=str, default=None)
    parser.add_argument('--num_top_norms', default=10)
    parser.add_argument('--idx_to_super_path', type=str, default='utils/wordnet/idx_to_super.json')
    parser.add_argument('--save_dir', type=str, default='caches/poly')
    parser.add_argument('--load', type=bool, default=True)
    return parser.parse_args()


def main(args):

    if args.mask_path is not None:
        mask = np.load(args.mask_path)  # HC
        mask = mask.astype(bool, copy=False)
        # np.multiply(all_norms, mask[None, :, :], out=all_norms)

    if not args.load:
        with open(args.idx_to_super_path, 'r') as f:
            idx_to_super = json.load(f)
        
        if args.mode == 'n':
            all_norms = np.load(args.all_norms_path)  # BC
            kurtosis_track = np.zeros(args.shape[1], dtype=np.float32)
            values_track = np.zeros((args.shape[1], args.num_top_norms), dtype=np.float32)
            indices_track = np.zeros(args.shape[1], dtype=np.float32)
        elif args.mode == 'nh':
            all_norms = np.load(args.all_norms_path)  # BHC
            kurtosis_track = np.zeros((args.shape[1], args.shape[-1]), dtype=np.float32)
            values_track = np.zeros((args.shape[1], args.shape[-1], args.num_top_norms), dtype=np.float32)
            indices_track = np.zeros((args.shape[1], args.shape[-1]), dtype=np.float32)
            
        if args.mode == 'n':
            for neuron in tqdm(range(all_norms.shape[1])):
                norms = all_norms[1]
                kurtosis_track[neuron] = kurtosis(norms)
                indices = np.argsort(norms)[-args.num_top_norms:][::-1]
                values = norms[indices]
                values_track[neuron] = values
                # map indices to superset and then find number of unique values
                supers = [idx_to_super.get(str(orig_idx), None) for orig_idx in indices]
                indices_track[neuron] = len(set(supers))
        elif args.mode == 'nh':
            for head, neuron in tqdm(itertools.product(range(all_norms.shape[1]), range(all_norms.shape[-1]))):
                norms = all_norms[:, head, neuron]
                kurtosis_track[head, neuron] = kurtosis(norms)
                indices = np.argsort(norms)[-args.num_top_norms:][::-1]
                values = norms[indices]
                values_track[head, neuron] = values
                # map indices to superset and then find number of unique values
                supers = [idx_to_super.get(str(orig_idx), None) for orig_idx in indices]
                indices_track[head, neuron] = len(set(supers))

        np.save(f'{args.save_dir}/{args.mode}_kurtosis_track_{args.num_top_norms}.npy', kurtosis_track)
        np.save(f'{args.save_dir}/{args.mode}_values_track_{args.num_top_norms}.npy', values_track)
        np.save(f'{args.save_dir}/{args.mode}_unique_supers_count_{args.num_top_norms}.npy', indices_track)
    
    kurtosis_track = np.load(f'{args.save_dir}/{args.mode}_kurtosis_track_{args.num_top_norms}.npy')
    values_track = np.load(f'{args.save_dir}/{args.mode}_values_track_{args.num_top_norms}.npy')
    indices_track = np.load(f'{args.save_dir}/{args.mode}_unique_supers_count_{args.num_top_norms}.npy')
    print(f"Average kurtosis: {kurtosis_track.mean()}")
    print(f"Average classes: {indices_track.mean()}")
    
    y_vals = list(indices_track.flatten())
    sorted_indices = sorted(range(len(y_vals)), key=lambda i: y_vals[i], reverse=True)
    new_y = [y_vals[i] for i in sorted_indices]
    plt.bar(range(len(y_vals)), new_y)
    plt.title('Kurtosis per component')
    plt.show()

    # Thinner fonts + strokes (Times-like), now doubled sizes
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L", "STIX Two Text", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.weight": "regular",
        "font.size": 24,        # was 9
        "axes.labelsize": 24,   # was 9
        "xtick.labelsize": 20,  # was 8
        "ytick.labelsize": 20,  # was 8
        "legend.fontsize": 20,  # was 8
        "axes.linewidth": 0.6, 
    })


    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False, constrained_layout=True)

    # First histogram (neurons) → orange
    ax1.hist(new_y, bins=10, label="Neurons", color="C1")
    ax1.legend()

    # Reload values for neuron-attention
    indices_track = np.load(f'{args.save_dir}/nh_unique_supers_count_{args.num_top_norms}.npy')
    y_vals = list(indices_track.flatten())
    sorted_indices = sorted(range(len(y_vals)), key=lambda i: y_vals[i], reverse=False)
    new_y = [y_vals[i] for i in sorted_indices]

    # Second histogram (neuron-attention) → blue
    ax2.hist(new_y, bins=10, label="Neuron-attention", color="C0")
    ax2.legend()

    # Force x-axis ticks to stride by 1
    for ax in (ax1, ax2):
        ax.set_xticks(np.arange(min(new_y), max(new_y)+1, 1))

    # Shared axis labels
    fig.supxlabel("Number of super-classes")
    fig.supylabel("Number of components")

    plt.savefig('figures/_final/poly.pdf')
    plt.show()


  
    


if __name__ == "__main__":
    args = parse_arguments()
    main(args)