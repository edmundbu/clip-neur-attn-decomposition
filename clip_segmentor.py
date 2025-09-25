import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
import sys 
sys.path.append("..")

from hook_collect import hook_register_conv_maps, hook_register_tokens
from hook_replace import hook_register_attn_sink, hook_register_neurons_zero
from utils.text_embedding_templates import OPENAI_IMAGENET_TEMPLATES
from utils.factory import create_model_and_transforms, get_tokenizer
from utils.get_dims import get_dims
# from utils.visualization import show_image, visualize_grid, overlay_heatmap
from utils.classes import pascal_context_classes
from segment_utils import get_topk_nh_pairs, interpolate_sink, denorm_for_display, threshold_pairs

from mmseg.models.segmentors import BaseSegmentor
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmengine.structures import PixelData
from mmseg.registry import MODELS


@MODELS.register_module()
class CLIPForSegmentation(BaseSegmentor):  # modified from https://github.com/wangf3014/SCLIP/tree/main
    def __init__(self, clip_path, name_path, device=torch.device('cuda'),
                    pamr_steps=0, pamr_stride=(8, 16), prob_thd=0.0, logit_scale=40, 
                    slide_stride=192, slide_crop=384, area_thd=None,
                    pcs_path=None, text_embed_path=None, text_for_topk=None, k=20000, 
                    conv_map_idx=7, sim_map_idx=7, sink_diff_path=None, use_self_self=False):
        
        data_preprocessor = SegDataPreProcessor(
            mean=[122.771, 116.746, 104.094],
            std=[68.501, 66.632, 70.323],
            rgb_to_bgr=True)
        super().__init__(data_preprocessor=data_preprocessor)

        assert pcs_path is not None and text_embed_path is not None and text_for_topk is not None
        self.net, _, self.val_preprocess = create_model_and_transforms(clip_path, pretrained='openai', device=device)
        self.net.visual.attnpool.forward_mode = 'per_head'
        self.num_heads, self.num_tokens, self.embed_dim, self.out_dim = get_dims(self.net)
        tokenizer = get_tokenizer(clip_path)
        if not use_self_self:
            self.conv_map_hook = hook_register_conv_maps(self.net, embed_dim=self.embed_dim)
            self.cls_map_hook = hook_register_tokens(self.net, mode='cls_spatial', collapse_heads=False)
            if sink_diff_path is not None:
                self.sink_neurons_hook = hook_register_neurons_zero(self.net, sink_diff_path, top_k=32, extra=True)

        query_words, self.query_idx = get_cls_idx(name_path)
        self.num_queries = len(query_words)
        self.num_classes = max(self.query_idx) + 1
        self.query_idx = torch.Tensor(self.query_idx).to(torch.int64).to(device)
        if text_embed_path is None:
            query_features = []
            with torch.no_grad():
                for qw in query_words:
                    query = tokenizer([temp(qw) for temp in OPENAI_IMAGENET_TEMPLATES]).to(device)
                    feature = self.net.encode_text(query)
                    feature /= feature.norm(dim=-1, keepdim=True)
                    feature = feature.mean(dim=0)
                    feature /= feature.norm()
                    query_features.append(feature.unsqueeze(0))
            self.query_features = torch.cat(query_features, dim=0)  # [num_classes, out_dim]
        else:
            self.query_features = torch.load(text_embed_path).to(device).float()
        self.query_features = self.query_features - self.query_features.mean(dim=0)
        self.query_features = self.query_features / self.query_features.norm(dim=-1, keepdim=True)
        self.out_dim = self.query_features.shape[-1]
        
        pcs = torch.from_numpy(np.load(pcs_path)).to(device)
        edit_texts = torch.load(text_for_topk).to(device).float()
        edit_texts = edit_texts - edit_texts.mean(dim=0)
        edit_texts = edit_texts / edit_texts.norm(dim=-1, keepdim=True)
        scores, self.cached_heads, self.cached_neurons = get_topk_nh_pairs(pcs, edit_texts, k=k)
        self.conv_map_idx = conv_map_idx
        self.sim_map_idx = sim_map_idx

        self.dtype = self.query_features.dtype
        self.logit_scale = logit_scale
        self.prob_thd = prob_thd
        self.area_thd = area_thd
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop
        self.align_corners = False

        self.use_self_self = use_self_self

        if pamr_steps > 0:
            raise NotImplementedError()
            # self.pamr = PAMR(pamr_steps, dilations=pamr_stride).to(device)
        else:
            self.pamr = None

    def forward_feature(self, img, logit_size=None):
        if type(img) == list:
            img = img[0]

        if self.use_self_self:
            print("Using self-self attention")
            self.net.visual.attnpool.forward_mode = 'self_self'
            out = self.net.encode_image(img)  
            B, N, O = out.shape
            H, W = img.shape[-2] // 32, img.shape[-1] // 32
            assert H*W == N-1
            out_normalized = F.normalize(out, p=2, dim=-1)
            image_features = torch.einsum('bno,po->bpn', out_normalized, self.query_features)
        else:
            # forward pass through only the ResNet and collect activation maps
            self.net.visual.attnpool.forward_mode = 'skip'
            self.net.encode_image(img)
            self.net.visual.attnpool.forward_mode = 'per_head'
            conv_maps = self.conv_map_hook.get_out_mat()
            self.conv_map_hook.reset_conv_maps()
            B, C, H, W = conv_maps[-1].shape
            N = H*W + 1

            # set up indexing
            repeat_heads = einops.repeat(self.cached_heads, 'p k -> b p n k', b=B, n=N)
            repeat_neurons = einops.repeat(self.cached_neurons, 'p k -> b p n k', b=B, n=N)

            # index into heads
            conv_map = conv_maps[self.sim_map_idx]
            self.net.visual.attnpool(conv_map)
            cls_map = self.cls_map_hook.get_out_mat()
            cls_map = interpolate_sink(cls_map, H, W)
            cls_map = cls_map / cls_map.norm(dim=-1, keepdim=True)
            sim_map = torch.einsum('bnko,po->bpnk', cls_map, self.query_features)
            sim_map_NAP = torch.gather(sim_map, dim=-1, index=repeat_heads)

            # index into neurons
            conv_map = conv_maps[self.conv_map_idx]
            conv_map = conv_map.reshape(B, C, -1).permute(0, 2, 1)
            conv_map = torch.cat([torch.full((B, 1, C), float('inf'), device=conv_map.device), conv_map], dim=1)
            conv_map = interpolate_sink(conv_map, H, W)
            conv_map = einops.repeat(conv_map, 'b n c -> b p n c', p=self.num_classes)
            conv_map_NAP = torch.gather(conv_map, dim=-1, index=repeat_neurons)

            if self.custom_means is None:
                image_features = (conv_map_NAP * sim_map_NAP).mean(dim=-1)  # BPN
            else:
                image_features = (sim_map_NAP * conv_map_NAP).sum(dim=-1) / self.custom_means.view(1, self.num_classes, 1)

        # reshape 
        logits = image_features[:, :, 1:].reshape(B, self.num_classes, H, W)
        if logit_size == None:
            logits = nn.functional.interpolate(logits, size=img.shape[-2:], mode='bilinear')
        else:
            logits = nn.functional.interpolate(logits, size=logit_size, mode='bilinear')
        return logits

    def forward_slide(self, img, img_metas, stride=192, crop_size=384):
        """Inference by sliding-window with overlap.
        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        if type(img) == list:
            img = img[0].unsqueeze(0)
        if type(stride) == int:
            stride = (stride, stride)
        if type(crop_size) == int:
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = img.shape
        out_channels = self.num_queries
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.forward_feature(crop_img)
                preds += nn.functional.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0

        preds = preds / count_mat
        img_size = img_metas[0]['ori_shape'][:2]
        logits = nn.functional.interpolate(preds, size=img_size, mode='bilinear')

        if self.pamr:
            img = nn.functional.interpolate(img, size=img_size, mode='bilinear')
            logits = self.pamr(img, logits.to(img.dtype)).to(self.dtype)

        return logits

    def predict(self, inputs, data_samples):
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]
        
        if self.slide_crop > 0:
            seg_logits = self.forward_slide(inputs, batch_img_metas, self.slide_stride, self.slide_crop)
        else:
            seg_logits = self.forward_feature(inputs, batch_img_metas[0]['ori_shape'])

        return self.postprocess_result(seg_logits, data_samples)
    
    def postprocess_result(self, seg_logits, data_samples):
        batch_size = seg_logits.shape[0]
        for i in range(batch_size):
            seg_logits = seg_logits[i] * self.logit_scale
            seg_logits = seg_logits.softmax(0) # n_queries * w * h

            num_cls, num_queries = max(self.query_idx) + 1, len(self.query_idx)
            if num_cls != num_queries:
                seg_logits = seg_logits.unsqueeze(0)
                cls_index = nn.functional.one_hot(self.query_idx)
                cls_index = cls_index.T.view(num_cls, num_queries, 1, 1)
                seg_logits = (seg_logits * cls_index).max(1)[0]
                seg_pred = seg_logits.argmax(0, keepdim=True)

            if self.area_thd is not None:
                # Force segmentations with area < self.area_thd to 0 (background)
                predictions = nn.functional.one_hot(seg_logits.argmax(0), num_cls).to(seg_logits.dtype)
                area_pred = predictions[:, :, 1:].sum((0, 1), keepdim=True)  # prone background
                area_pred = (area_pred > self.area_thd * area_pred.sum()).to(seg_logits.dtype)          
                seg_logits[1:] *= area_pred.transpose(0, -1)
            
            seg_pred = seg_logits.argmax(0, keepdim=True)
            seg_pred[seg_logits.max(0, keepdim=True)[0] < self.prob_thd] = 0
            
            data_samples[i].set_data({
                'seg_logits':
                PixelData(**{'data': seg_logits}),
                'pred_sem_seg':
                PixelData(**{'data': seg_pred})
            })

        return data_samples
    
    def _forward(data_samples):
        """
        """
    
    def inference(self, img, batch_img_metas):
        """
        """

    def encode_decode(self, inputs, batch_img_metas):
        """
        """
    
    def extract_feat(self, inputs):
        """
        """
    
    def loss(self, inputs, data_samples):
        """
        """

def get_cls_idx(path):
    with open(path, 'r') as f:
        name_sets = f.readlines()
    num_cls = len(name_sets)

    class_names, class_indices = [], []
    for idx in range(num_cls):
        names_i = name_sets[idx].split(', ')
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace('\n', '') for item in class_names]
    return class_names, class_indices