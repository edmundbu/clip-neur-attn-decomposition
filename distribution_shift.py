import argparse
import json
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr

from utils.cars_dataset import CustomStanfordCars
from utils.factory import create_model_and_transforms, get_tokenizer
from text_set_projection import get_text_features
from hook_collect import hook_register_neurons_heads


def parse_arguments():
    parser = argparse.ArgumentParser(description="Distribution shift")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_name', type=str, default='RN50x16')
    parser.add_argument('--pcs_path', type=str, default='my_caches/pcs/neurons+heads_pc0_D=1000_k=50.npy')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--cars_root', type=str, default='cars')
    parser.add_argument('--refined_cars_csv', type=str, default='cars_refined/refined_train.csv')
    parser.add_argument('--load', type=bool, default=True)
    return parser.parse_args()


def get_topk_nh_pairs(pcs, text_embedding, k):
    num_heads, embed_dim, out_dim = pcs.shape
    pcs = pcs.reshape(num_heads*embed_dim, out_dim)
    dots = torch.matmul(pcs,  text_embedding)
    _, topk_indices = torch.topk(dots, k)
    topk_h = topk_indices // embed_dim
    topk_c = topk_indices % embed_dim
    return topk_h, topk_c


@torch.no_grad()
def main(args):
    model, _, val_preprocess = create_model_and_transforms(args.model_name, pretrained='openai', device=args.device)
    tokenizer = get_tokenizer(args.model_name)
    cars_dataset = CustomStanfordCars(root=args.cars_root, split='train', transform=val_preprocess)

    years = range(1991, 2013)
    yellow_per_year = {year: [] for year in years}  
    convertible_per_year = {year: [] for year in years}  
    total_per_year = {year: 0 for year in years}
    
    refined_df = pd.read_csv(args.refined_cars_csv)
    images = refined_df['image']
    classes = refined_df['Class'] 
    color_idx_map = ['', '', '', '', '', '', '', '', '', 'yellow']
    for idx in range(len(images)):
        class_idx = classes[idx]
        label, color = class_idx.split('_')
        label = int(label) - 1
        label = cars_dataset.get_class_name(label)
        label_split = label.split(' ')
        year = label_split[-1]
        color, year = int(color), int(year)
        color = color_idx_map[color]
        total_per_year[year] += 1
        if color == 'yellow':
            yellow_per_year[year].append(1)
        else:
            yellow_per_year[year].append(0)
        if 'Convertible' in label:
            convertible_per_year[year].append(1)
        else:
            convertible_per_year[year].append(0)

    if not args.load:
        pcs = torch.from_numpy(np.load(args.pcs_path)).to('cuda')
        yellow_embedding = get_text_features(model, tokenizer, ['yellow'], use_templates=True).squeeze(0).to(args.device)
        convertible_embedding = get_text_features(model, tokenizer, ['convertible'], use_templates=True).squeeze(0).to(args.device)
        yellow_heads, yellow_neurons = get_topk_nh_pairs(pcs, yellow_embedding, k=args.k)
        convertible_heads, convertible_neurons = get_topk_nh_pairs(pcs, convertible_embedding, k=args.k)

        dataloader = DataLoader(cars_dataset, batch_size=1, shuffle=False)
        model.visual.attnpool.forward_mode = 'per_head_per_neuron'
        hook = hook_register_neurons_heads(model)
        yellow_model_per_year = {year: [] for year in years}  
        convertible_model_per_year = {year: [] for year in years}  

        image_set = set(images.astype(str).values)
        for img_path, image, label in tqdm(dataloader):
            path = Path(img_path[0]).name 
            if path not in image_set:  # some entries missing from the csv :(
                print(f"Skipped {path}")
                continue

            year = int(cars_dataset.get_class_name(label).split(' ')[-1])
            model.encode_image(image.to(args.device))

            out = hook.get_out_mat().squeeze(0)  # HOC
            total_contrib = out.sum(dim=[0, -1]).norm(dim=-1).item()
            yellow_contrib = out[yellow_heads, :, yellow_neurons].sum(dim=0).norm().item()  
            convertible_contrib = out[convertible_heads, :, convertible_neurons].sum(dim=0).norm().item()  
        
            yellow_model_per_year[year].append(yellow_contrib / total_contrib)
            convertible_model_per_year[year].append(convertible_contrib / total_contrib)

        with open('cars_refined/yellow.json', 'w') as f:
            json.dump(yellow_model_per_year, f)
        with open('cars_refined/convertible.json', 'w') as g:
            json.dump(convertible_model_per_year, g)

    with open('cars_refined/yellow.json', 'r') as f:
        yellow_preds = json.load(f)
        yellow_preds = {int(k): np.array(v) for k, v in yellow_preds.items()}
    with open('cars_refined/convertible.json', 'r') as g:
        convertible_preds = json.load(g)
        convertible_preds = {int(k): np.array(v) for k, v in convertible_preds.items()}