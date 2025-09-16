import json
import time
import os
import argparse

import torch
import numpy as np
from tqdm import tqdm, trange
import scipy as sp

from utils.decompose import Decompose


def parse_arguments():
    parser = argparse.ArgumentParser(description='Sparse decomposition for (neuron, head)s')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_name', type=str, default='RN50x16')
    parser.add_argument('--num_components', type=int, default=64)
    parser.add_argument('--text_descs_list_path', type=str, default='utils/text_descriptions/30k.txt')
    parser.add_argument('--text_descs_embeddings_path', type=str)
    parser.add_argument('--pcs_path', type=str)
    parser.add_argument('--descriptions_save_path', type=str)
    parser.add_argument('--decomposition_save_path', type=str)
    return parser.parse_args()


def main(args):
    # load files
    with open(args.text_descs_list_path, 'r') as f:
        lines = [i.replace('\n', '') for i in f.readlines()]
    print("Text descriptions loaded!")
    text_descs = np.load(args.text_descs_embeddings_path)  # must be computed from same textset 
    text_descs = text_descs - text_descs.mean(axis=0)
    text_descs = text_descs / np.linalg.norm(text_descs, axis=0, keepdims=True)
    print("Text descriptions embeddings loaded!")  # [textset, d_out]
    pcs = np.load(args.pcs_path)  
    if pcs.ndim > 2:
        pcs = pcs.reshape((pcs.shape[0]*pcs.shape[1], pcs.shape[2]))
    print(f"PCs flattened to shape {pcs.shape}!")

    # compute sparse decomposition
    if not os.path.isfile(args.decomposition_save_path):
        print("Computing sparse dictionary!")
        before = time.time()
        coder = Decompose(
            text_descs,
            l1_penalty=1.0,
            transform_n_nonzero_coefs=args.num_components,
        )
        decomposition = coder.transform(pcs)  
        print("Done in", (time.time() - before) / (60 * 60), "hours!")
        sparse_matrix = sp.sparse.csc_matrix(decomposition)
        sp.sparse.save_npz(args.decomposition_save_path, sparse_matrix)
    else:
        decomposition = sp.sparse.load_npz(args.decomposition_save_path).toarray()
        # [components, textset]

    # label neurons, and find all neurons that contribute to some concept
    print("Pre")
    sparse_decomposition = decomposition.copy()
    print("Post")
    jsn = {}
    for neuron in trange(decomposition.shape[0], desc="Creating json!"):
        jsn[neuron] = []
        all_addresses = np.argsort(np.abs(decomposition[neuron]))
        addresses = all_addresses[-args.num_components :]
        zero_address = all_addresses[: -args.num_components]
        for j in addresses[::-1]:
            jsn[neuron].append((int(j), float(decomposition[neuron, j]), str(lines[j])))
        sparse_decomposition[neuron, zero_address] = 0

    sorted_data = sorted(jsn[1300], key=lambda x: abs(x[1]), reverse=True)
    top3 = sorted_data[:3]
    print(top3)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)

