import argparse

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm, trange

from utils.factory import create_model_and_transforms, get_tokenizer
from utils.text_embedding_templates import OPENAI_IMAGENET_TEMPLATES
from utils.classes import imagenet_classes, pascal_context_classes


def parse_arguments():
    parser = argparse.ArgumentParser(description="Tokenize text set into normalized embeddings")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_name', type=str, default='RN50x16')
    parser.add_argument('--classnames', type=str, default='IN', choices=['IN', '30k', 'PC'])
    parser.add_argument('--embeddings_save_path', type=str)
    parser.add_argument('--use_templates', type=bool, default=True)
    return parser.parse_args()


@torch.no_grad()
def get_text_features(model, tokenizer, lines, use_templates=False):
    """Computes normalized text embeddings for each line in lines."""
    zeroshot_weights = []
    device = next(model.parameters()).device
    if use_templates:
        classnames = lines 
        templates = OPENAI_IMAGENET_TEMPLATES  # 75 templates
        for classname in tqdm(classnames, "Computing zero-shot text embeddings!"):
            texts = [template(classname) for template in templates]
            tokens = tokenizer(texts)
            text_features = model.encode_text(tokens.to(device))  
            text_features /= text_features.norm(dim=-1, keepdim=True)  # normalize 
            class_embedding = text_features.mean(dim=0)  # average over all prompts
            class_embedding /= class_embedding.norm()  # normalize again
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=0)
    else:
        for i in trange(0, len(lines), desc="Computing embeddings with no template!"):
            texts = lines[i]
            texts = tokenizer(texts).to(device)  # tokenize
            class_embeddings = model.encode_text(texts)
            class_embeddings = F.normalize(class_embeddings, dim=-1)
            zeroshot_weights.append(class_embeddings.detach().cpu())
        zeroshot_weights = torch.concatenate(zeroshot_weights, dim=0)
    return zeroshot_weights


def main(args):
    model, _, _ = create_model_and_transforms(args.model_name, pretrained='openai', device=args.device)
    tokenizer = get_tokenizer(args.model_name)
    model.to(args.device)
    model.eval()

    if args.classnames == 'IN':
        lines = imagenet_classes
    elif args.classnames == '30k':
        with open('utils/text_descriptions/30k.txt' 'r') as f:
            lines = f.readlines()
    elif args.classnames == 'PC':
        lines = pascal_context_classes

    features = get_text_features(model, tokenizer, lines, args.use_templates)
    torch.save(features, args.embeddings_save_path)  # [|textset|, d_out]


if __name__ == "__main__":
    args = parse_arguments()
    main(args)