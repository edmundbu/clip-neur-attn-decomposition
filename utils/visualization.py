import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torchvision.transforms import functional as F2
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import colorsys

from utils.constants import OPENAI_DATASET_STD, OPENAI_DATASET_MEAN
from utils.classes import pascal_context_classes

ALPHA = 0.6
COLORMAP = {'magma': cv2.COLORMAP_MAGMA}


"""
For visualizing single images.
"""


def squarify(vector, cls_in_quadrant=1):
    assert vector.ndim == 1
    grid = vector
    n = grid.shape[0]
    sqrt_n = int(math.sqrt(n))
    if sqrt_n * sqrt_n != n:
        if cls_in_quadrant == 1:
            cls_like = grid[0]
            grid = grid[1:].reshape(sqrt_n, sqrt_n)
        elif cls_in_quadrant == 4:
            cls_like = grid[-1]
            grid = grid[:-1].reshape(sqrt_n, sqrt_n)
    return grid


def visualize_grid(grid, title='', cls_in_quadrant=1, cmap='viridis'):
    """Expects shape [N, N] or [1+N*N]."""
    if isinstance(grid, torch.Tensor):
        grid = grid.cpu().numpy()
    if grid.ndim == 1:
        grid = squarify(grid, cls_in_quadrant)
        
    plt.figure(figsize=(5, 5))
    sns.heatmap(grid, cmap=cmap)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def overlay_heatmap(heatmap, image, title='', cmap='magma', save_dir=None):
    heatmap_np = heatmap.cpu().numpy() if isinstance(heatmap, torch.Tensor) else heatmap
    if isinstance(image, torch.Tensor):
        image_np = image.cpu().clone()
    if heatmap_np.ndim == 1:
        heatmap_np = squarify(heatmap_np)
    assert heatmap_np.ndim == 2
    assert image_np.ndim == 3 and image_np.shape[0] == 3

    image_unnormed = image_np * torch.tensor(OPENAI_DATASET_MEAN).view(3, 1, 1) + torch.tensor(OPENAI_DATASET_STD).view(3, 1, 1)
    image_np = image_unnormed.clamp(0, 1).permute(1, 2, 0).numpy()  # CHW -> HWC
    image_np_uint8 = (image_np * 255).astype(np.uint8)

    heatmap_resized = cv2.resize(heatmap_np, (image_np.shape[1], image_np.shape[0]))
    heatmap_norm = cv2.normalize(heatmap_resized, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_uint8 = heatmap_norm.astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, COLORMAP[cmap])

    image_bgr = cv2.cvtColor(image_np_uint8, cv2.COLOR_RGB2BGR)
    blended_bgr = cv2.addWeighted(image_bgr, 1 - ALPHA, heatmap_color, ALPHA, 0)
    blended_rgb = cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)

    overlay = blended_rgb
    fig, ax = plt.subplots()
    ax.imshow(overlay)
    ax.axis('off')
    vmin = np.min(heatmap_np)
    vmax = np.max(heatmap_np)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array(heatmap_np)
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    plt.title(title)
    plt.show()


def show_image(tensor, index=0, save_path=None, show_im=True):
    img = tensor[index].detach().cpu()
    if img.ndim == 3 and img.shape[0] not in (1, 3): # likely HWC
        img = img.permute(2, 0, 1)  # convert to CHW
    mean = torch.tensor(OPENAI_DATASET_MEAN).view(3,1,1)
    std  = torch.tensor(OPENAI_DATASET_STD).view(3,1,1)
    img = img * std + mean
    img = F2.to_pil_image(img)
    plt.imshow(img)
    plt.axis("off")
    img.save(save_path)
    if show_im:
        plt.show()


"""
For visualizing many images.
"""

mean = torch.tensor([122.771, 116.746, 104.094], dtype=torch.float32)
std = torch.tensor([ 68.501,  66.632,  70.323], dtype=torch.float32)

def to_torch(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x

def denormalize_image(img_norm: torch.Tensor, mean=mean, std=std) -> np.ndarray:
    """
    img_norm: [3,H,W], normalized via (img - mean)/std where img was in 0..255 scale.
    Returns uint8 RGB array [H,W,3] suitable for plt.imshow.
    """
    if img_norm.ndim != 3 or img_norm.shape[0] != 3:
        raise ValueError("img_norm must be [3, H, W].")
    # broadcast: [3,1,1]
    img = img_norm * std[:, None, None].to(img_norm.device) + mean[:, None, None].to(img_norm.device)
    img = img.clamp(0, 255).permute(1, 2, 0)  # [H,W,3]
    return img.byte().cpu().numpy()

def overlay_heatmaps(
    heatmaps, img_norm, mean=mean, std=std,
    alpha=0.45, cmap="magma", cols=8,
    panel_title="All overlays (raw logits)", save_panel_path=None,
    class_arr=pascal_context_classes
):
    heatmaps = to_torch(heatmaps)
    img_norm = to_torch(img_norm)

    if heatmaps.ndim == 2:
        heatmaps = heatmaps.unsqueeze(0)
    elif heatmaps.ndim == 4 and heatmaps.shape[1] == 1:
        heatmaps = heatmaps[:,0]

    N, Hh, Wh = heatmaps.shape
    Hi, Wi = img_norm.shape[-2:]

    # Upsample if heatmap resolution != image resolution
    if (Hh, Wh) != (Hi, Wi):
        heatmaps = F.interpolate(
            heatmaps.unsqueeze(1).float(),
            size=(Hi, Wi),
            mode="bilinear",
            align_corners=False
        ).squeeze(1)

    base = denormalize_image(img_norm, mean, std)  # [Hi,Wi,3] uint8

    rows = math.ceil(N / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3.2, rows*3.2), squeeze=False)
    fig.text(
        0.98, 0.03,                 # x=98% across, y=2% up from bottom
        panel_title,                # your title string
        ha='right', va='bottom',    # align to bottom-right
        fontsize=14,
        fontweight='bold'
    )

    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        ax.axis("off")

        if i >= N:
            continue

        hm = heatmaps[i].float().cpu().numpy()  # raw logits

        # Show raw logits with their true min/max
        im = ax.imshow(base)
        im = ax.imshow(hm, cmap=cmap, alpha=alpha, interpolation="nearest")

        # Add per-heatmap colorbar with raw value scale
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=6)

        ax.set_title(f"{class_arr[i]}", fontsize=9)

    plt.tight_layout(h_pad=3.5)  # increase vertical padding
    plt.subplots_adjust(top=0.75, hspace=0.6, wspace=0.2)

    if save_panel_path is not None:
        fig.savefig(save_panel_path, dpi=150, bbox_inches="tight")
        
    plt.show()


"""Single image segmentation."""

def argmax_segment_and_label(
    logits_PHW: torch.Tensor,         # shape: (P, H, W)
    classnames: list,                 # length P
    image_3HprimeWprime: torch.Tensor,# shape: (3, H', W'), values 0..1 or 0..255
    alpha: float = 0.2,               # overlay strength for colored mask
    font_path: str = None,     # optional path to a .ttf
    base_font_size: int = 14          # will be scaled by patch size
):
    """
    Returns:
        annotated_pil (PIL.Image): original image with colored segmentation overlay + per-patch text labels
        upscaled_labels (torch.LongTensor): (H', W') label map aligned to the original image
        lowres_labels (torch.LongTensor): (H, W) label map before upscaling
    """
    assert logits_PHW.ndim == 3, "logits must be (P, H, W)"
    P, H, W = logits_PHW.shape
    assert len(classnames) == P, "classnames length must match P"

    # 1) Argmax over classes -> (H, W)
    lowres_labels = logits_PHW.argmax(dim=0)  # (H, W), dtype long

    # 2) Upscale to original image size using nearest (keeps integer labels intact)
    assert image_3HprimeWprime.ndim == 3 and image_3HprimeWprime.shape[0] == 3, "image must be (3, H', W')"
    _, Hprime, Wprime = image_3HprimeWprime.shape

    upscaled_labels = F.interpolate(
        lowres_labels.unsqueeze(0).unsqueeze(0).float(),  # (1,1,H,W)
        size=(Hprime, Wprime),
        mode="nearest"
    ).squeeze(0).squeeze(0).to(torch.long)  # (H', W')

    # 3) Build a simple color palette for classes (P x 3) in 0..255
    #    (deterministic distinct-ish colors)
    torch.manual_seed(42)
    palette = (torch.rand(P, 3) * 255).to(torch.uint8)

    # 4) Make a colored overlay image from labels
    color_overlay = palette[upscaled_labels]  # (H', W', 3) uint8

   # 4) Use YOUR denormalization for the base image (no changes to your function)
    base_rgb_uint8 = denormalize_image(image_3HprimeWprime)  # [H', W', 3] uint8
    base_pil = Image.fromarray(base_rgb_uint8, mode="RGB")

    # 6) Blend overlay onto image
    overlay_pil = Image.fromarray(color_overlay.cpu().numpy(), mode="RGB")
    blended = Image.blend(base_pil, overlay_pil, alpha=alpha)

    # 7) Draw per-patch text at the center of each (H x W) patch region
    draw = ImageDraw.Draw(blended)

    # Pick a legible font size relative to patch size
    # (roughly 35% of the smaller dimension of a patch, with a floor)
    patch_h = Hprime / H
    patch_w = Wprime / W
    est_size = int(max(12, min(patch_h, patch_w) * 0.35))
    font_size = max(base_font_size, est_size)
    try:
        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    # For readability, draw text with a stroke
    for i in range(H):
        for j in range(W):
            cls_idx = int(lowres_labels[i, j])
            label = classnames[cls_idx]

            # Center of this patch in the upscaled image
            cx = int(round((j + 0.5) * patch_w))
            cy = int(round((i + 0.5) * patch_h))

            # Keep text within image bounds
            cx = max(0, min(Wprime - 1, cx))
            cy = max(0, min(Hprime - 1, cy))

            # Draw centered text with stroke for contrast
            draw.text(
                (cx, cy),
                label,
                fill=(255, 255, 255),
                font=font,
                anchor="mm",          # center the text at (cx, cy)
                stroke_width=max(1, font_size // 10),
                stroke_fill=(0, 0, 0)
            )

    return blended, upscaled_labels, lowres_labels


"""Draw on labels"""

def _non_green_palette(P: int, seed: int = 42) -> torch.Tensor:
    """
    Deterministic HSV palette that avoids the green hue band (~90°–150°).
    Returns (P,3) uint8 RGB.
    """
    # Define hue ranges to use (in [0,1]): [0,0.25) U (0.45,1.0)
    allowed = []
    # Evenly sample more than needed, then filter
    samples = max(P * 3, 64)
    for k in range(samples):
        h = (k / samples) % 1.0
        if (h < 0.25) or (h > 0.45):  # skip 0.25–0.45 (~90–162 deg) = greenish
            allowed.append(h)
    # If P is larger than filtered hues, wrap around
    step = max(1, len(allowed) // P)
    chosen = [allowed[(i * step) % len(allowed)] for i in range(P)]

    # Convert HSV -> RGB, keep saturation/value comfortably high but not neon
    rgb = []
    for h in chosen:
        r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.95)
        rgb.append([int(r * 255), int(g * 255), int(b * 255)])
    return torch.tensor(rgb, dtype=torch.uint8)

def make_segmap_with_legend(
    logits_PHW: torch.Tensor,          # (P, H, W)
    classnames: list,                  # length P
    image_3HprimeWprime: torch.Tensor, # (3, H', W'), values 0..1 or 0..255
    *,
    overlay_alpha: float = 0.35,       # blend strength on top of original image
    palette_rgb255: Optional[torch.Tensor] = None,  # optional (P,3) uint8
    avoid_green: bool = True,          # prefer non-green palette
    seed: int = 42,                    # deterministic palette spacing
    legend_width: int = 240,           # px; side panel width
    legend_bg: tuple = (255, 255, 255),
    legend_text_color: tuple = (20, 20, 20),
    base_font_size: int = 16,
    font_path: Optional[str] = None
) -> Tuple[Image.Image, torch.LongTensor, torch.LongTensor]:
    """
    Returns:
        annotated (PIL.Image): original image blended with colored segmentation + side legend
        upscaled_labels (torch.LongTensor): (H', W') label map aligned to original image
        lowres_labels (torch.LongTensor): (H, W) label map before upscaling

    Notes:
      • Smooth boundaries: we upsample LOGITS with bilinear → argmax at full res.
      • Non-green colors by default (you can pass your own palette if you want).
      • Legend shows color swatches and class names in a side panel.
    """
    # --- sanity checks
    assert logits_PHW.ndim == 3, "logits must be (P, H, W)"
    P, H, W = logits_PHW.shape
    assert len(classnames) == P, "classnames length must match P"
    assert image_3HprimeWprime.ndim == 3 and image_3HprimeWprime.shape[0] == 3, \
        "image must be (3, H', W')"
    _, Hprime, Wprime = image_3HprimeWprime.shape

    # --- 1) Argmax at low-res (for return value)
    lowres_labels = logits_PHW.argmax(dim=0)  # (H, W), long

    # --- 2) Smooth boundaries: upsample LOGITS to (H', W') with bilinear, then argmax
    #      (Operate as NCHW for interpolate)
    up_logits = F.interpolate(
        logits_PHW.unsqueeze(0),  # (1, P, H, W)
        size=(Hprime, Wprime),
        mode="bilinear",
        align_corners=False
    ).squeeze(0)  # (P, H', W')
    upscaled_labels = up_logits.argmax(dim=0).to(torch.long)  # (H', W')

    # --- 3) Palette
    if palette_rgb255 is not None:
        assert palette_rgb255.shape == (P, 3) and palette_rgb255.dtype == torch.uint8, \
            "palette_rgb255 must be (P,3) uint8"
        palette = palette_rgb255
    else:
        palette = _non_green_palette(P, seed=seed) if avoid_green else (
            (torch.rand(P, 3) * 255).to(torch.uint8)  # basic random as fallback
        )

    # --- 4) Colorize labels (H', W', 3) uint8
    seg_rgb_uint8 = palette[upscaled_labels]  # index with (H', W') → (H', W', 3)
    seg_pil = Image.fromarray(seg_rgb_uint8.cpu().numpy(), mode="RGB")

    # --- 5) Blend overlay on original
    base_rgb_uint8 = denormalize_image(image_3HprimeWprime)  # uses YOUR helper
    base_pil = Image.fromarray(base_rgb_uint8, mode="RGB")
    blended = Image.blend(base_pil, seg_pil, alpha=overlay_alpha)

    # --- 6) Build side legend panel
    legend = Image.new("RGB", (legend_width, Hprime), legend_bg)
    d = ImageDraw.Draw(legend)

    # pick font
    try:
        font = ImageFont.truetype(font_path, base_font_size) if font_path else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    # layout parameters
    padding = 14
    swatch_h = max(14, int(base_font_size * 1.1))
    swatch_w = 28
    gap = 10  # gap between swatch and text
    line_gap = max(6, base_font_size // 3)

    y = padding
    for idx, name in enumerate(classnames):
        # swatch
        color = tuple(int(c) for c in palette[idx].tolist())
        d.rectangle([padding, y, padding + swatch_w, y + swatch_h], fill=color, outline=(0,0,0))
        # text (wrap if needed)
        text_x = padding + swatch_w + gap
        d.text((text_x, y), name, fill=legend_text_color, font=font)
        y += swatch_h + line_gap
        if y > Hprime - padding - swatch_h:
            # stop if legend panel is full
            break

    # --- 7) Stitch blended image + legend side-by-side
    out = Image.new("RGB", (Wprime + legend_width, Hprime), (0, 0, 0))
    out.paste(blended, (0, 0))
    out.paste(legend, (Wprime, 0))

    # --- 8) Also print a key to stdout (optional but handy)
    print("Segmentation color key (R,G,B):")
    for idx, name in enumerate(classnames):
        r, g, b = map(int, palette[idx].tolist())
        print(f"  {name:<20s} -> ({r:3d}, {g:3d}, {b:3d})")

    return out, upscaled_labels, lowres_labels
