import argparse
import os
import itertools

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np

from utils.test_dataset import ImageNetTest
from utils.factory import create_model_and_transforms


def parse_arguments():
    parser = argparse.ArgumentParser(description="Proxy for polysemanticity")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_name', type=str, default='RN50x16')
    parser.add_argument('--mode', type=str, choices=['n', 'nh'])
    parser.add_argument('--norms_path', type=str)  # norms over 1000 images (not top 50)
    parser.add_argument('--embeddings_save_path', type=str)
    parser.add_argument('--scores_save_path', type=str)
    parser.add_argument('--num_top_norms', default=10)
    parser.add_argument('--load', type=bool, default=True)
    return parser.parse_args()


@torch.no_grad()
def cache_image_embeddings(model, device, transform, save_path):
    dataset = ImageNetTest('test_data/test', transform=transform)
    dataset = Subset(dataset, list(range(0, len(dataset), 5)))  # 1000 images total
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    res = []
    for image in tqdm(dataloader):
        out = model.encode_image(image.to(device))
        res.append(out / out.norm(dim=-1))
    np.save(save_path, torch.concatenate(res, dim=0).cpu().numpy())


def main(args):
    if not args.load:
        model, _, val_preprocess = create_model_and_transforms(args.model_name, pretrained='openai', device=args.device)
        if not os.path.isfile(args.embeddings_save_path):
            cache_image_embeddings(model, args.device, val_preprocess, args.embeddings_save_path)
        all_embeddings = np.load(args.embeddings_save_path)
        all_norms = np.load(args.norms_path)
        cluster_vals = []
        
        if args.mode == 'n':
            for neuron in tqdm(range(all_norms.shape[1])):
                norms = all_norms[:, neuron]
                indices = np.argsort(norms)[-args.num_top_norms:][::-1]
                embeddings = np.array([all_embeddings[i] for i in indices])
                score = ((embeddings - embeddings.mean(axis=0))**2).sum()
                cluster_vals.append(score)
        elif args.mode == 'nh':
            for head, neuron in tqdm(itertools.product(range(all_norms.shape[1]), range(all_norms.shape[-1]))):
                norms = all_norms[:, head, neuron]
                indices = np.argsort(norms)[-args.num_top_norms:][::-1]
                embeddings = np.array([all_embeddings[i] for i in indices])
                score = ((embeddings - embeddings.mean(axis=0))**2).sum()
                cluster_vals.append(score)

        np.save(args.scores_save_path, np.array(cluster_vals))

    cluster_vals = np.load(args.scores_save_path)