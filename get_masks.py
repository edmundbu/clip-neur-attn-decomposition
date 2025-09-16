import argparse

import numpy as np
import matplotlib.pyplot as plt

from utils.get_dims import get_dims
from utils.factory import create_model_and_transforms


def parse_arguments():
    parser = argparse.ArgumentParser(description="Save masks for mean-ablation")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_name', type=str, default='RN50x16')
    parser.add_argument('--mode', type=str, default='neur_head_reprs', choices=['neur_acts', 'neur_reprs', 'neur_head_reprs'])  
    parser.add_argument('--neuron_acts_norms_path', type=str)  
    parser.add_argument('--neurons_norms_path', type=str)  
    parser.add_argument('--neuron_head_norms_path', type=str)  
    parser.add_argument('--top_images', type=int, default=50)
    parser.add_argument('--percentile', type=int, default=90)
    parser.add_argument('--show_hist', type=bool, default=False)
    parser.add_argument('--mask_save_path', type=str)  
    return parser.parse_args()


def main(args):
    model, _, _ = create_model_and_transforms(args.model_name, pretrained='openai', device=args.device)
    model.eval()
    model.visual.attnpool.forward_mode = 'per_head_per_neuron'

    if args.mode == 'neur_acts':
        norms = np.load(args.neuron_acts_norms_path)
    elif args.mode == 'neur_reprs':
        norms = np.load(args.neurons_norms_path)
    elif args.mode == 'neur_head_reprs':
        norms = np.load(args.neuron_head_norms_path)

    norms = np.moveaxis(norms, -1, 0)
    print(np.sort(norms, axis=0))
    topk_vals = np.sort(norms, axis=0)[-args.top_images:, :]  
    topk_avg = np.mean(topk_vals, axis=0)
    print(f"Took means of top {args.top_images} activating images!")
    flat_vals = topk_avg.flatten()

    thresh = np.percentile(flat_vals, args.percentile)
    mask = topk_avg >= thresh
    np.save(args.mask_save_path, mask)
    print(f"{np.sum(mask)} {args.mode} ({100-args.percentile}%) saved!")

    if args.show_hist:
        plt.hist(flat_vals, bins=10000)  
        plt.xlim(0, np.max(flat_vals))
        plt.xlabel('Mean value')
        plt.ylabel('Frequency')
        plt.axvline(x=thresh, color='red', linestyle='--', linewidth=2)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)