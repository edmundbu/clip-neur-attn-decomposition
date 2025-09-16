import argparse

import torch
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader, Subset
import numpy as np
import scipy as sp
from tqdm import tqdm

from hook_replace import hook_register_reconstruct
from utils.factory import create_model_and_transforms
from utils.get_dims import get_dims


def parse_arguments():
    parser = argparse.ArgumentParser(description="Reconstruction")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_name', type=str, default='RN50x16')
    parser.add_argument('--dataset', type=str, default='imagenet_val')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--sample_stride', type=int, default=1)
    parser.add_argument('--imagenet_zeroshot_path', type=str)
    parser.add_argument('--pcs_paths', nargs='+')
    parser.add_argument('--means_path', type=str, help="Means that are computed alongside principal components") 
    parser.add_argument('--text_descs_embeddings_path', type=str)
    parser.add_argument('--decomposition_save_path', type=str, default=None)
    return parser.parse_args()


@torch.no_grad()
def main(args):
    model, _, val_preprocess = create_model_and_transforms(args.model_name, pretrained='openai', device=args.device)
    model.eval()
    model.to(args.device)
    model.visual.attnpool.forward_mode = 'per_head_per_neuron'
    num_heads, num_tokens, embed_dim, out_dim = get_dims(model)
    text_features = torch.load(args.imagenet_zeroshot_path)
    if args.dataset == 'imagenet_val': 
        dataset = ImageNet(root='data', split='val', transform=val_preprocess)
    dataset = Subset(dataset, list(range(0, len(dataset), args.sample_stride)))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    if args.decomposition_save_path is not None:  # sparse
        means = torch.from_numpy(np.load(args.means_path)).transpose(-2, -1).to(args.device)
        decomposition = sp.sparse.load_npz(args.decomposition_save_path).toarray()  # CD or (H*C)D 
        print(decomposition.shape)
        text_descs = np.load(args.text_descs_embeddings_path)  # DO
        text_descs = text_descs - text_descs.mean(axis=0)
        text_descs = text_descs / np.linalg.norm(text_descs, axis=0, keepdims=True)

        if decomposition.shape[0] == embed_dim:  
            pcs_from_sparse_code = torch.from_numpy(decomposition @ text_descs).permute(1, 0).to(args.device)  # O-first
            pcs_from_sparse_code = pcs_from_sparse_code / pcs_from_sparse_code.norm(dim=0, keepdim=True)
        elif decomposition.shape[0] == embed_dim * num_heads:  
            decomposition = decomposition.reshape((num_heads, embed_dim, decomposition.shape[-1]))  # make sure to reshape BEFORE matmul
            pcs_from_sparse_code = torch.from_numpy(decomposition @ text_descs).to(args.device)  # HCO
            pcs_from_sparse_code = pcs_from_sparse_code / pcs_from_sparse_code.norm(dim=-1, keepdim=True)
            pcs_from_sparse_code = pcs_from_sparse_code.transpose(-2, -1)
            print(f"Reshaped reconstructed pcs to shape {pcs_from_sparse_code.shape}")
        pcs = [pcs_from_sparse_code]

    else:
        pcs = [torch.from_numpy(np.load(path)).transpose(-2, -1).to(args.device) for path in args.pcs_paths]
        means = torch.from_numpy(np.load(args.means_path)).transpose(-2, -1).to(args.device)

    hook = hook_register_reconstruct(model, pcs=pcs, means=means)  
    
    correct, accum = 0, 0
    progress_bar = tqdm(dataloader, "Computing reconstruction accuracy!")
    for images, labels in progress_bar:
        model.encode_image(images.to(args.device))
        out = hook.get_output()
        image_features = out / out.norm(dim=-1, keepdim=True)
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        preds = torch.argmax(text_probs, dim=-1)
        correct += (preds == labels.to(args.device)).sum().item()
        accum += labels.shape[0]
        progress_bar.set_postfix({"Accuracy": f"{correct / accum:.4f}"})
    print(f"Accuracy: {correct / len(dataset)}") 


if __name__ == "__main__":
    args = parse_arguments()
    main(args)