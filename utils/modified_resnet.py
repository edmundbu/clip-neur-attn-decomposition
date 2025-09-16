from typing import Optional
from collections import OrderedDict
import math

import torch
import einops
from torch import nn
from torch.nn import functional as F

from utils.misc import freeze_batch_norm_2d
from utils.hook import HookManager
from utils.visualization import visualize_grid


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, hook: Optional[HookManager]=None):
        super().__init__()
        self.hook_manager = hook or HookManager()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.act3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            # (anti-aliasing)
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

        self.cancel_relu = None

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.hook_manager("post_act", ret=self.act1(self.bn1(self.conv1(x))))
        out = self.hook_manager("post_act", ret=self.act2(self.bn2(self.conv2(out))))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        if self.cancel_relu is not None and out.shape[1] in self.cancel_relu:
            raise NotImplementedError("ReLU is definitely needed...")
        out = self.hook_manager("post_act", ret=self.act3(out))
        return out
    

class MultiHeadAttentionDecomposed(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_out: int, hook: Optional[HookManager]=None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = self.d_model // num_heads
        self.d_out = d_out
        self.hook_manager = hook or HookManager()
        self.extra = False
    
    def _split_weight(self, weight):
        return weight.reshape(self.num_heads, self.d_head, weight.shape[-1])

    def _split_bias(self, bias):
        return bias.reshape(1, self.num_heads, 1, self.d_head)

    def forward_per_head(self, x, W_q, W_k, W_v, W_o, B_q, B_k, B_v, B_o):
        x = x.permute(1, 0, 2)  
        W_q_split, B_q_split = self._split_weight(W_q), self._split_bias(B_q)  # CC -> HDC, C -> 1H1D 
        W_k_split, B_k_split = self._split_weight(W_k), self._split_bias(B_k) 
        W_v_split, B_v_split = self._split_weight(W_v), self._split_bias(B_v)

        q_split = torch.einsum('bnc,hdc->bhnd', x, W_q_split) + B_q_split
        k_split = torch.einsum('bnc,hdc->bhnd', x, W_k_split) + B_k_split
        v_split = torch.einsum('bnc,hdc->bhnd', x, W_v_split) + B_v_split
        q_split = q_split / (math.sqrt(self.d_head) * 1)
        attn = self.hook_manager('attn.pre_softmax', ret=q_split @ k_split.transpose(-2, -1))  # BHNN
        attn = self.hook_manager('attn.post_softmax', ret=F.softmax(attn, dim=-1)) 
        
        if self.extra:
            attn = attn[:, :, :-1, :-1]  # shift out attention sink
            v_split = v_split[:, :, :-1, :]
        
        x = self.hook_manager('extended_attn_v', ret=torch.einsum('bhnm,bhmd->bnmhd', attn, v_split))
        x = self.hook_manager(
            'out.pre_collapse',
            ret=torch.einsum(
                'bnmhd,ohd->bnmho',
                x,
                W_o.reshape(
                    self.d_out, self.num_heads, self.d_head
                )
            )
        )  

        x = self.hook_manager('out.post_collapse', ret=x.sum(axis=[2, 3]))  # BNO
        x += B_o
        return x[:, 0, :]
    
    def forward_per_head_per_neuron(self, x, W_q, W_k, W_v, W_o, B_q, B_k, B_v, B_o):
        """Returns output decomposed over heads, tokens, and neurons (very memory-intensive)."""
        x = x.permute(1, 0, 2) # BNC
        W_q_split, B_q_split = self._split_weight(W_q), self._split_bias(B_q) # CC -> HDC, C -> 1H1D 
        W_k_split, B_k_split = self._split_weight(W_k), self._split_bias(B_k) 

        q_split = torch.einsum('bnc,hdc->bhnd', x, W_q_split) + B_q_split
        k_split = self.hook_manager('key', ret=torch.einsum('bnc,hdc->bhnd', x, W_k_split) + B_k_split)
        q_split = q_split / math.sqrt(self.d_head)
        attn = self.hook_manager('attn.pre_softmax', ret=q_split @ k_split.transpose(-2, -1))  # BHNN
        attn = self.hook_manager('attn.post_softmax', ret=F.softmax(attn, dim=-1))
        attn_cls = attn[:, :, 0, :]  # take row that gives cls output

        W_v_split = self._split_weight(W_v) 
        B_v_split = self._split_bias(B_v) 
        B_v_split_neuron = einops.repeat(
            B_v_split, 'b h n d -> b h n d c', c=self.d_model) / self.d_model
        v_split_neuron = torch.einsum('bnc,hdc->bhndc', x, W_v_split)  # maintain c dim
        v_split_neuron = self.hook_manager('value', ret=v_split_neuron + B_v_split_neuron) 
        x = torch.einsum('bhm,bhmdc->bhdc', attn_cls, v_split_neuron)  

        x = self.hook_manager(
            'out.pre_collapse',
            ret=torch.einsum(
                'bhdc,ohd->bhoc',
                x, W_o.reshape(self.d_out, self.num_heads, self.d_head)
            )
        )
        
        x = self.hook_manager('out.post_head_collapse', ret=x.sum(axis=1))
        x = x.sum(axis=2) 
        x += B_o
        return x
    
    def forward_per_neuron_spatial(self, x, W_q, W_k, W_v, W_o, B_q, B_k, B_v, B_o):
        x = x.permute(1, 0, 2)  
        W_q_split, B_q_split = self._split_weight(W_q), self._split_bias(B_q)  # CC -> HDC, C -> 1H1D 
        W_k_split, B_k_split = self._split_weight(W_k), self._split_bias(B_k) 
        q_split = torch.einsum('bnc,hdc->bhnd', x, W_q_split) + B_q_split
        k_split = torch.einsum('bnc,hdc->bhnd', x, W_k_split) + B_k_split
        q_split = q_split / (math.sqrt(self.d_head) * 1)
        attn = self.hook_manager('attn.pre_softmax', ret=q_split @ k_split.transpose(-2, -1))  # BHNN
        attn = self.hook_manager('attn.post_softmax', ret=F.softmax(attn, dim=-1)) 
        attn = attn[:, :, 0, :]

        W_v_split = self._split_weight(W_v) 
        B_v_split = self._split_bias(B_v) 
        B_v_split_neuron = einops.repeat(
            B_v_split, 'b h n d -> b h n d c', c=self.d_model) / self.d_model
        v_split_neuron = torch.einsum('bnc,hdc->bhndc', x, W_v_split)  # maintain c dim
        v_split_neuron = v_split_neuron + B_v_split_neuron 
        B, _, M, _, C = v_split_neuron.shape
        O = W_o.shape[0]  

        W_o_reshape = W_o.reshape(self.d_out, self.num_heads, self.d_head)
        x = torch.zeros(B, M, C, O, device=attn.device, dtype=attn.dtype)
        for h in range(self.num_heads):
            attn_v = torch.einsum('bm,bmdc->bmdc', attn[:, h, :], v_split_neuron[:, h, :, :, :])
            x += torch.einsum('bmdc,od->bmco', attn_v, W_o_reshape[:, h, :])
        x = self.hook_manager('out.pre_collapse', ret=x)

        x = self.hook_manager('out.post_collapse', ret=x.sum(axis=[1, 2]))  # BO
        x += B_o
        return x
    
    def forward_self_self(self, x, W_q, W_k, W_v, W_o, B_q, B_k, B_v, B_o):
        """Returns all output tokens post-W_o-projection from self-self attention (QQ^T + KK^T)."""
        x = x.permute(1, 0, 2)  # BNC
        t = math.sqrt(self.d_head)
        W_q_split, B_q_split = self._split_weight(W_q), self._split_bias(B_q)  # CC -> HDC, C -> 1H1D 
        W_k_split, B_k_split = self._split_weight(W_k), self._split_bias(B_k) 
        W_v_split, B_v_split = self._split_weight(W_v), self._split_bias(B_v)
        q_split = torch.einsum('bnc,hdc->bhnd', x, W_q_split) + B_q_split
        k_split = torch.einsum('bnc,hdc->bhnd', x, W_k_split) + B_k_split
        v_split = torch.einsum('bnc,hdc->bhnd', x, W_v_split) + B_v_split
        attn_q = (q_split @ q_split.transpose(-2, -1)) / t
        attn_k = (k_split @ k_split.transpose(-2, -1)) / t
        attn = F.softmax(attn_q, dim=-1) + F.softmax(attn_k, dim=-1)

        x = torch.einsum('bhnm,bhmd->bnhd', attn, v_split)
        x = torch.einsum('bnhd,ohd->bnho', x, W_o.reshape(self.d_out, self.num_heads, self.d_head))
        x = x.sum(dim=-2) + B_o
        return x[:, :, :]  # BNO
    
    def forward_values_only(self, x, W_q, W_k, W_v, W_o, B_q, B_k, B_v, B_o):
        x = x.permute(1, 0, 2)  # BNC
        W_v_split, B_v_split = self._split_weight(W_v), self._split_bias(B_v)
        v_split = torch.einsum('bnc,hdc->bhnd', x, W_v_split) + B_v_split
        x = torch.einsum('bhnd,ohd->bno', v_split, W_o.reshape(self.d_out, self.num_heads, self.d_head))
        x = x + B_o
        return x[:, :, :]

    def forward(self, x, mode, W_q, W_k, W_v, W_o, B_q, B_k, B_v, B_o):
        # B := batch, N := seq_len, C := d_embed, H := num_heads, D := d_head, O := d_out
        fwd_method = getattr(self, f'forward_{mode}')
        return fwd_method(x, W_q, W_k, W_v, W_o, B_q, B_k, B_v, B_o)
    
 
class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None, hook: Optional[HookManager]=None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

        self.hook_manager = hook or HookManager()
        self.MHA_decomposed = MultiHeadAttentionDecomposed(embed_dim, num_heads, output_dim, self.hook_manager)
        self.extra = False
        self.forward_mode = 'default'

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # BCHW -> (HW)BC
        cls = self.hook_manager('cls.post_average', ret=x.mean(dim=0, keepdim=True))  # 1BC
        x = torch.cat([cls, x], dim=0)  # (HW+1)BC
        x = self.hook_manager('input_tokens', ret=x)
        x = self.hook_manager('input_tokens_pos_embed', ret=x + self.positional_embedding[:, None, :].to(x.dtype))  # (HW+1)BC

        if self.forward_mode == 'skip': return None
        if self.forward_mode == 'per_head' or self.forward_mode == 'per_head_per_neuron' \
            or self.forward_mode == 'per_neuron_spatial' or self.forward_mode == 'self_self' or self.forward_mode == 'values_only':
            x = self.MHA_decomposed(x, self.forward_mode, 
                                    self.q_proj.weight, self.k_proj.weight, self.v_proj.weight, self.c_proj.weight,
                                    self.q_proj.bias, self.k_proj.bias, self.v_proj.bias, self.c_proj.bias)
        else:
            x, _ = F.multi_head_attention_forward(
                query=x, key=x, value=x,
                embed_dim_to_check=x.shape[-1],
                num_heads=self.num_heads,
                q_proj_weight=self.q_proj.weight,
                k_proj_weight=self.k_proj.weight,
                v_proj_weight=self.v_proj.weight,
                in_proj_weight=None,
                in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
                bias_k=None,
                bias_v=None,
                add_zero_attn=False,
                dropout_p=0.,
                out_proj_weight=self.c_proj.weight,
                out_proj_bias=self.c_proj.bias,
                use_separate_proj_weight=True,
                training=self.training,
                need_weights=False
            )
            x = x[0]
        
        return x
    

class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, image_size=224, width=64, hook: Optional[HookManager]=None):
        super().__init__()
        self.hook_manager = hook or HookManager()

        self.output_dim = output_dim
        self.image_size = image_size

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.act2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.act3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(image_size // 32, embed_dim, heads, output_dim, self.hook_manager)

      
        self.init_parameters()

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride, hook=self.hook_manager)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes, hook=self.hook_manager))

        return nn.Sequential(*layers)

    def init_parameters(self):
        if self.attnpool is not None:
            std = self.attnpool.c_proj.in_features ** -0.5
            nn.init.normal_(self.attnpool.q_proj.weight, std=std)
            nn.init.normal_(self.attnpool.k_proj.weight, std=std)
            nn.init.normal_(self.attnpool.v_proj.weight, std=std)
            nn.init.normal_(self.attnpool.c_proj.weight, std=std)

        for resnet_block in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for name, param in resnet_block.named_parameters():
                if name.endswith("bn3.weight"):
                    nn.init.zeros_(param)

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        assert unlocked_groups == 0, 'partial locking not currently supported for this model'
        for param in self.parameters():
            param.requires_grad = False
        if freeze_bn_stats:
            freeze_batch_norm_2d(self)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        # FIXME support for non-transformer
        pass

    def stem(self, x):
        x = self.hook_manager("post_act", ret=self.act1(self.bn1(self.conv1(x))))
        x = self.hook_manager("post_act", ret=self.act2(self.bn2(self.conv2(x))))
        x = self.hook_manager("post_act", ret=self.act3(self.bn3(self.conv3(x))))
        x = self.avgpool(x)
        return x

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.hook_manager("pre_pooling", ret=self.layer4(x))
        x = self.attnpool(x)
        return x
    

