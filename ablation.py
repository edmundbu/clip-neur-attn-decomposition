import argparse

import torch
import einops
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm

from utils.factory import create_model_and_transforms
from hook_collect import hook_register_neurons_heads
from hook_replace import hook_register_attn_sink, hook_register_neurons_zero, hook_register_neurons_mean
from utils.get_dims import get_dims


def parse_arguments():
    parser = argparse.ArgumentParser(description="Mean-ablation (excuse the mess!)")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_name', type=str, default='RN50x16')
    parser.add_argument('--dataset', type=str, default='imagenet_val')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--sample_stride', type=int, default=10)
    parser.add_argument('--imagenet_zeroshot_path', type=str)
    parser.add_argument('--mode', type=str, default='neur_head_reprs', 
                        choices=['zero_sink', 'zero_register_neurons', 
                                 'neur_acts', 'neur_reprs', 'neur_head_reprs'])
    parser.add_argument('--means_path', type=str)
    parser.add_argument('--replace_index_path', type=str)
    # attention sink zeroing
    parser.add_argument('--neurons_sink_diffs_path', type=str)
    parser.add_argument('--top_k_sink_neurons', type=int, default=18)
    parser.add_argument('--extra_token', type=bool, default=False)
    return parser.parse_args()


@torch.no_grad()
def main(args):
    model, _, val_preprocess = create_model_and_transforms(args.model_name, pretrained='openai', device=args.device)
    model.eval()
    model.to(args.device)
    text_features = torch.load(args.imagenet_zeroshot_path)
    if args.dataset == 'imagenet_val':
        dataset = ImageNet(root='data', split='val', transform=val_preprocess)
    dataset = Subset(dataset, list(range(0, len(dataset), args.sample_stride)))
    assert len(dataset) % args.batch_size == 0
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)
    
    num_heads, num_tokens, embed_dim, out_dim = get_dims(model)
    means = torch.from_numpy(np.load(args.means_path)).to(args.device).transpose(-2, -1)
    print(f"Means loaded with shape {means.shape}")
    if args.mode == 'zero_sink':  # modifies in place
        model.visual.attnpool.forward_mode = 'per_head'
        hook = hook_register_attn_sink(model, mode='post')
    elif args.dataset == 'zero_register_neurons':
        model.visual.attnpool.forward_mode = 'default'
        hook_register_neurons_zero(model, args.neurons_sink_diffs_path, args.top_k_sink_neurons, args.extra_token)
    elif args.mode == 'neur_acts':
        model.visual.attnpool.forward_mode = 'default'
        replace_mask = torch.from_numpy(np.load(args.replace_index_path)).to(args.device)  # C
        replace_mask = ~replace_mask  # replace False values with means  
        replace_mask = einops.repeat(replace_mask, 'c -> n b c', n=num_tokens, b=args.batch_size)
        means = einops.repeat(means, 'n c -> n b c', b=args.batch_size)
        hook = hook_register_neurons_mean(model, mask=replace_mask, means=means)
    elif args.mode == 'neur_reprs':
        model.visual.attnpool.forward_mode = 'per_head_per_neuron'
        replace_mask = torch.from_numpy(np.load(args.replace_index_path)).to(args.device)
        num = torch.sum(replace_mask.flatten()).item()
        print(f"{num} {args.mode} are replaced ({100.0 * num / embed_dim:.2f}%)!")
        collapse = lambda x : x.sum(axis=[-1]) 
        hook = hook_register_neurons_heads(model)
        replace_mask = einops.repeat(replace_mask, 'c -> o c', o=out_dim)
        means = einops.repeat(means, 'o c -> b o c', b=args.batch_size)
    elif args.mode == 'neur_head_reprs':
        model.visual.attnpool.forward_mode = 'per_head_per_neuron'
        replace_mask = torch.from_numpy(np.load(args.replace_index_path)).to(args.device)
        num = torch.sum(replace_mask.flatten()).item()
        print(f"{num} {args.mode} are replaced ({100.0 * num / (num_heads * embed_dim):.2f}%)!")
        collapse = lambda x : x.sum(axis=[1, -1])  
        hook = hook_register_neurons_heads(model)
        replace_mask = einops.repeat(replace_mask, 'h c -> h o c', o=out_dim)
        means = einops.repeat(means, 'h o c -> b h o c', b=args.batch_size)
 
    correct, accum = 0, 0
    progress_bar = tqdm(dataloader, f"Computing {args.mode} ablation accuracy!")
    for images, labels in progress_bar:
        out = model.encode_image(images.to(args.device))
        if args.mode == 'neur_acts':
            pass  # TODO
        elif args.mode == 'neur_reprs':
            model_out = hook.get_out_mat().sum(dim=1)  # sum over heads
            for i in range(args.batch_size):
                means[i][replace_mask] = model_out[i][replace_mask]
            out = collapse(means)
        elif args.mode == 'neur_head_reprs':
            model_out = hook.get_out_mat()
            for i in range(args.batch_size):
                means[i][replace_mask] = model_out[i][replace_mask]
            out = collapse(means)
        elif args.mode == 'neur_reprs_per_im':
            model_out = hook.get_out_mat().sum(dim=1)  # BOC
            model_out_norm = model_out.norm(dim=-2)  # BC
            for i in range(args.batch_size):
                thresh = torch.quantile(model_out_norm[i], q=(args.threshold_percentile/100))
                replace_mask = model_out_norm[i] >= thresh
                replace_mask = einops.repeat(replace_mask, 'c -> o c', o=out_dim)
                means[i][replace_mask] = model_out[i][replace_mask]
            out = means.sum(dim=-1)

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