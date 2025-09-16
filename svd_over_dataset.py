import argparse
import os
import itertools

import numpy as np
from tqdm import trange, tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(description="SVD over collected neuron or neuron-attention contributions")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_name', type=str, default='RN50x16')
    parser.add_argument('--shape', type=tuple, help="Shape of the mem map")
    parser.add_argument('--collect_save_path', type=str)
    parser.add_argument('--mode', type=str, default='nh', choices=['n', 'nh'])
    parser.add_argument('--mean_save_path', type=str)
    parser.add_argument('--top_k_pca', type=int, default=50)
    parser.add_argument('--out_save_path', type=str)
    parser.add_argument('--pc_idx', type=int, default=0) 
    parser.add_argument('--norm_out_path', type=str)
    return parser.parse_args()


def main(args):
    if args.mode == 'n':
        neurons = np.load(args.collect_save_path)
        print(f"Neuron reprs loaded at {args.collect_save_path}!")  # [images, neurons, d_out]
        if not os.path.isfile(args.mean_save_path):
            neurons_mean = neurons.mean(axis=0)
            np.save(args.mean_save_path, neurons_mean)
            print(f"Mean over dataset saved to {args.mean_save_path}!")
        neurons_mean = np.load(args.mean_save_path)
        pcas = []
        norms = []
        num_neurons = neurons.shape[1]
        for neuron in trange(num_neurons, desc=f"Computing top-{args.pc_idx} PC for each neuron!"):
            current_neurons = neurons[:, neuron] - neurons_mean[neuron]
            important = np.argsort(np.linalg.norm(current_neurons, axis=-1))[
                -args.top_k_pca :
            ]
            current_important_neurons = current_neurons[important]
            norms.append(
                np.sort(np.linalg.norm(current_neurons, axis=-1))[-args.top_k_pca :]
            )
            u, s, vh = np.linalg.svd(
                current_important_neurons, full_matrices=False
            )  # (u * s) @ vh is value
            how_many_positive = len(
                np.nonzero(1.0 * ((current_important_neurons @ vh[args.pc_idx]) > 0))[0]
            )
            # set the direction:
            if how_many_positive > args.top_k_pca // 2:
                pcas.append(vh[args.pc_idx])
            else:
                pcas.append(-vh[args.pc_idx])

        pcas = np.stack(pcas, axis=0)  # [neurons, d_out]
        np.save(args.out_save_path, pcas)
        print(f"Top {args.pc_idx} pc saved to {args.out_save_path}!")
        print(pcas)
        norms = np.stack(norms, axis=0).reshape(num_neurons, -1)  # [heads, neurons, top_k]
        np.save(args.norm_out_path, norms)
        print(f"PCA norms for each neuron saved to {args.norm_out_path}")
        print(norms)

    elif args.mode == 'nh':
        neurons_heads = np.memmap(args.collect_save_path, dtype=np.float32, mode='r', shape=args.shape)
        print(f"Neuron-attn reprs loaded at {args.collect_save_path}!")  # [images, heads, neurons, d_out]
        if not os.path.isfile(args.mean_save_path):
            neurons_heads_mean = neurons_heads.mean(axis=0)
            np.save(args.mean_save_path, neurons_heads_mean)
            print(f"Mean over dataset saved to {args.mean_save_path}!")
        neurons_heads_mean = np.load(args.mean_save_path)

        pcas = []
        norms = []
        num_heads, num_neurons = neurons_heads.shape[1], neurons_heads.shape[2]
        for head, neuron in tqdm(itertools.product(range(num_heads), range(num_neurons)),
                                 desc=f"Computing top-{args.pc_idx} PC for each neuron-head pair!"):
            
            current_neurons_heads = neurons_heads[:, head, neuron] - neurons_heads_mean[head, neuron]
            important = np.argsort(np.linalg.norm(current_neurons_heads, axis=-1))[
                -args.top_k_pca :
            ]
            current_important_neurons = current_neurons_heads[important]
            norms.append(
                np.sort(np.linalg.norm(current_neurons_heads, axis=-1))[-args.top_k_pca :]
            )
            u, s, vh = np.linalg.svd(
                current_important_neurons, full_matrices=False
            )  # (u * s) @ vh is value

            how_many_positive = len(
                np.nonzero(1.0 * ((current_important_neurons @ vh[args.pc_idx]) > 0))[0]
            )
            # set the direction:
            if how_many_positive > args.top_k_pca // 2:
                pcas.append(vh[args.pc_idx])
            else:
                pcas.append(-vh[args.pc_idx])

        saved_pcas = np.stack(pcas, axis=0).reshape(num_heads, num_neurons, -1)  # [heads, neurons, d_out]
        np.save(args.out_save_path, saved_pcas)
        print(f"Top {args.pc_idx} PC saved to {args.out_save_path}!")
        print(saved_pcas)
        norms = np.stack(norms, axis=0).reshape(num_heads, num_neurons, -1)  # [heads, neurons, top_k]
        np.save(args.norm_out_path, norms)
        print(f"PCA norms for each neuron-attention pair saved to {args.norm_out_path}")
        print(norms)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
    
        