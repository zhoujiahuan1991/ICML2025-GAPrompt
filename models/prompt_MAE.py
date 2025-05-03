import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import DropPath, trunc_normal_
import numpy as np
from .build import MODELS
from utils import misc
import ipdb
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
from .modules import square_distance, index_points
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from knn_cuda import KNN
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from .GAPrompt import ShiftNet, PointPrompt, Group, propagate, pooling

class PointNetFeaturePropagation(nn.Module):
    def __init__(self):
        super(PointNetFeaturePropagation, self).__init__()

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N] pts
            xyz2: sampled input points position data, [B, C, S] center
            points1: input points data, [B, D, N] pts
            points2: input points data, [B, D, S] x
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        return interpolated_points
    
class Encoder(nn.Module):   ## Embedding module
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n , _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 384 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 384
        return feature_global.reshape(bs, g, self.encoder_channel) # [B, G, 384]



## Transformers
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, require_attn = False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if require_attn:
            return x, attn
        return x


# Block with prompt

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_tokens=10, config=None):
        super().__init__()
        if config is not None:
            self.config = config
        self.norm1 = norm_layer(dim)
        self.dim = dim

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
        self.prompt_dropout = nn.Dropout(0.1)
        self.num_tokens = num_tokens
        self.prompt_embeddings = nn.Parameter(torch.zeros(self.num_tokens, dim))
        self.adapter=None
        self.scaler1 = nn.Parameter(torch.ones([1])*0.5)
        self.scaler2 = nn.Parameter(torch.ones([1])*0.5)
        self.adapter = Adapter(embed_dims=dim, reduction_dims=16)
        self.out_transform = nn.Sequential(
                                nn.BatchNorm1d(dim),
                                nn.GELU()
                            )

    def forward(self, x, global_feature=None, token_position=None, layer_id=None, level1_center=None, level1_index=None, level2_center=None, level2_index=None, batch_idx=None):
        if global_feature is not None and layer_id<self.config.prompt_depth:
            token_prompt = self.prompt_dropout(self.prompt_embeddings.repeat(x.shape[0], 1, 1))
            if self.config.scaler == True:
                token_prompt = token_prompt + self.scaler1*global_feature.repeat([1, self.num_tokens, 1])+token_position.repeat(x.shape[0], 1, 1)
            else:
                token_prompt = token_prompt + 0.5*global_feature.repeat([1, self.num_tokens, 1])+token_position.repeat(x.shape[0], 1, 1)
            x = torch.cat((x[:,0:1], token_prompt, x[:,1:]), 1)
        elif token_position is not None and layer_id<self.config.prompt_depth:
            token_prompt = self.prompt_dropout(self.prompt_embeddings.repeat(x.shape[0], 1, 1))
            token_prompt = token_prompt + token_position.repeat(x.shape[0], 1, 1)
            x = torch.cat((x[:,0:1], token_prompt, x[:,1:]), 1)

        propagation_type = 'replacement_after_attention' # 'replacement'

        if self.config.propagation_type == 'permutation_before_attention':
            B,G,_ = x.shape
            cls_x = x[:,0:1]
            x = x[:,1:]
            G = G-1 # +self.num_tokens
            propagate_range = level1_center.shape[1]
            x_neighborhoods = x.reshape(B*G, -1)[level1_index, :].reshape(B*level2_center.shape[1], -1, self.dim)
            x_centers = x.reshape(B*G, -1)[level2_index, :].reshape(B, level2_center.shape[1], self.dim)
            x_neighborhoods = self.drop_path(x_neighborhoods)+x_neighborhoods
            vis_x = pooling(x_neighborhoods.reshape(B, level2_center.shape[1], -1, self.dim), transform=self.out_transform)+0.3*x_centers
            x[:,-propagate_range:] = propagate(xyz1=level1_center, xyz2=level2_center, points1=x[:,-propagate_range:], points2=vis_x)
            x = torch.concat([cls_x,x], dim=1) 
        elif self.config.propagation_type == 'replacement_before_attention':
            B,G,_ = x.shape
            cls_x = x[:,0:1]
            x = x[:,1:]
            G = G-1 # +self.num_tokens
            propagate_range = level1_center.shape[1]
            if G > propagate_range:
                token_prompt, x = x[:, :-propagate_range], x[:,-propagate_range:]
                replace_x = x.clone()
                x = torch.concat([token_prompt, x], dim=1)
                # replace_x[:, :self.num_tokens] = token_prompt
                replace_x[:, -self.num_tokens:] = token_prompt
                x_neighborhoods = replace_x.reshape(B*propagate_range, -1)[level1_index, :].reshape(B*level2_center.shape[1], -1, self.dim)
                x_centers = replace_x.reshape(B*propagate_range, -1)[level2_index, :].reshape(B, level2_center.shape[1], self.dim)
            else:
                x_neighborhoods = x.reshape(B*propagate_range, -1)[level1_index, :].reshape(B*level2_center.shape[1], -1, self.dim)
                x_centers = x.reshape(B*propagate_range, -1)[level2_index, :].reshape(B, level2_center.shape[1], self.dim)
             
            x_neighborhoods = self.drop_path(x_neighborhoods)+x_neighborhoods
            vis_x = pooling(x_neighborhoods.reshape(B, level2_center.shape[1], -1, self.dim), transform=self.out_transform)+0.3*x_centers
            # de_neighbors=32 for modelnet40
            x[:,-propagate_range:] = propagate(xyz1=level1_center, xyz2=level2_center, points1=x[:,-propagate_range:], points2=vis_x) 
            x = torch.concat([cls_x,x], dim=1) 

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        # visualization = False
        # if visualization:
        #     if layer_id == 11:
        #         _, attn_weight = self.attn(self.norm1(x), require_attn=True)
        #         task = 'attention'
        #         import os
        #         os.makedirs(f'./visualization/{task}',exist_ok=True)
        #         np.save(f'./visualization/{task}/attn-weight-{batch_idx}', attn_weight.detach().cpu().numpy())
        
        if self.config.propagation_type == 'permutation_after_attention':
            B,G,_ = x.shape
            cls_x = x[:,0:1]
            x = x[:,1:]
            G = G-1 # +self.num_tokens
            propagate_range = level1_center.shape[1]
            x_neighborhoods = x.reshape(B*G, -1)[level1_index, :].reshape(B*level2_center.shape[1], -1, self.dim)
            x_centers = x.reshape(B*G, -1)[level2_index, :].reshape(B, level2_center.shape[1], self.dim)
            x_neighborhoods = self.drop_path(x_neighborhoods)+x_neighborhoods
            vis_x = pooling(x_neighborhoods.reshape(B, level2_center.shape[1], -1, self.dim), transform=self.out_transform)+0.3*x_centers
            x[:,-propagate_range:] = propagate(xyz1=level1_center, xyz2=level2_center, points1=x[:,-propagate_range:], points2=vis_x)
        elif self.config.propagation_type == 'replacement_after_attention':
            B,G,_ = x.shape
            cls_x = x[:,0:1]
            x = x[:,1:]
            G = G-1 # +self.num_tokens
            propagate_range = level1_center.shape[1]
            if G > propagate_range:
                token_prompt, x = x[:, :-propagate_range], x[:,-propagate_range:]
                replace_x = x.clone()
                x = torch.concat([token_prompt, x], dim=1)
                # replace_x[:, :self.num_tokens] = token_prompt
                replace_x[:, -self.num_tokens:] = token_prompt
                x_neighborhoods = replace_x.reshape(B*propagate_range, -1)[level1_index, :].reshape(B*level2_center.shape[1], -1, self.dim)
                x_centers = replace_x.reshape(B*propagate_range, -1)[level2_index, :].reshape(B, level2_center.shape[1], self.dim)
            else:
                x_neighborhoods = x.reshape(B*propagate_range, -1)[level1_index, :].reshape(B*level2_center.shape[1], -1, self.dim)
                x_centers = x.reshape(B*propagate_range, -1)[level2_index, :].reshape(B, level2_center.shape[1], self.dim)
             
            x_neighborhoods = self.drop_path(x_neighborhoods)+x_neighborhoods
            vis_x = pooling(x_neighborhoods.reshape(B, level2_center.shape[1], -1, self.dim), transform=self.out_transform)+0.3*x_centers
            x[:,-propagate_range:] = propagate(xyz1=level1_center, xyz2=level2_center, points1=x[:,-propagate_range:], points2=vis_x)
        elif 'before' in self.config.propagation_type:
            cls_x = x[:,0:1]
            x = x[:,1:]
        
        if self.adapter is not None:
            if global_feature is not None and layer_id<6:
                if self.config.scaler == True:
                    x = x + self.adapter(x+global_feature*self.scaler2)
                else:
                    x = x + self.adapter(x+global_feature*0.5)
            else:
                x = x + self.adapter(x)
        if layer_id<self.config.prompt_depth:
            x = torch.concat([cls_x, x[:,self.num_tokens:,:]], dim=1)
        else:
            x = torch.concat([cls_x, x], dim=1)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., config=None):
        super().__init__()
        if config is not None:
            self.config = config
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                num_tokens=config.num_tokens,
                config=config
                )
            for i in range(depth)])

    def forward(self, x, pos, global_feature=None, token_position=None, level1_center=None, level1_index=None, level2_center=None, level2_index=None, batch_idx=None):
        for idx, block in enumerate(self.blocks):
            if idx < self.config.prompt_depth and token_position is not None:
                x = block(x + pos, global_feature=global_feature, token_position=token_position, layer_id=idx, level1_center=level1_center, level1_index=level1_index, level2_center=level2_center, level2_index=level2_index)
            else:
                x = block(x + pos, layer_id=idx, level1_center=level1_center, level1_index=level1_index, level2_center=level2_center, level2_index=level2_index, batch_idx=batch_idx)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim=384, depth=4, num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos, return_token_num):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens predict pixel
        return x


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class Adapter(nn.Module):
    def __init__(self,
                 embed_dims,
                 reduction_dims,
                 drop_rate_adapter=0.1
                ):
        super(Adapter, self).__init__()
        self.embed_dims = embed_dims
        self.super_reductuion_dim = reduction_dims     
        self.dropout = nn.Dropout(p=drop_rate_adapter)

        if self.super_reductuion_dim > 0:
            self.layer_norm = nn.LayerNorm(self.embed_dims)
            self.scale = nn.Linear(self.embed_dims, 1)
            self.ln1 = nn.Linear(self.embed_dims, self.super_reductuion_dim)
            self.activate = QuickGELU()
            self.ln2 = nn.Linear(self.super_reductuion_dim, self.embed_dims)
            self.init_weights()
        
    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=5**0.5)
                nn.init.normal_(m.bias, std=1e-6)
        self.apply(_init_weights)

    def set_sample_config(self, sample_embed_dim):
        self.sample_embed_dim = sample_embed_dim
        self.sampled_weight_0 = self.ln1.weight[:self.sample_embed_dim,:]
        self.sampled_bias_0 =  self.ln1.bias[:self.sample_embed_dim]
        self.sampled_weight_1 = self.ln2.weight[:, :self.sample_embed_dim]
        self.sampled_bias_1 =  self.ln2.bias

    def forward(self, x):
        x = self.layer_norm(x)
        # scale = F.relu(self.scale(x))
        scale = 0.7
        out = self.ln1(x)
        out = self.activate(out)
        out = self.dropout(out)
        out = self.ln2(out)
        return out*scale

    def calc_sampled_param_num(self):
        return  self.sampled_weight_0.numel() + self.sampled_bias_0.numel() + self.sampled_weight_1.numel() + self.sampled_bias_1.numel()

    def get_complexity(self, sequence_length):
        total_flops = 0
        if self.sampled_bias is not None:
             total_flops += self.sampled_bias.size(0)
        total_flops += sequence_length * np.prod(self.sampled_weight.size())
        return total_flops
    
# prompt tuning model with point promt, token prompt, and shiftnet
@MODELS.register_module()
class PointTransformer_pointtokenprompt(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.level2_group_divider = Group(num_group=self.num_group//2, group_size=self.group_size//2)

        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        #position for prompt token    
        self.prompt_cor = nn.Parameter(torch.zeros(10, 3))
        trunc_normal_(self.prompt_cor, std=.06)
        
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
            config=self.config
        )

        self.norm = nn.LayerNorm(self.trans_dim)
        self.cls_head_finetune = nn.Sequential(
                nn.Linear(self.trans_dim * 3, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.6),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.6),
                nn.Linear(256, self.cls_dim)
            )
        for layer in self.cls_head_finetune:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight, a=5.0**0.5)
        self.point_prompt = None
        if config.point_prompt == True:
            self.point_prompt = PointPrompt(point_number=config.point_number, init_type='uniform', scale=config.scale, factor=config.factor)
        self.shift_net = None
        if config.shift_net == True:
            self.shift_net = ShiftNet(3,3, hidden_dimesion=config.encoder_dims, perturbation=config.perturbation, num_group=config.num_group, group_size=config.group_size)
        if self.shift_net is not None:
            self.shape_feature_mlp = nn.Sequential(
                nn.Linear(self.shift_net.num_group//2*self.shift_net.top_center_dim, 16),
                nn.GELU(),
                nn.Dropout(0.25),
                nn.Linear(16, self.encoder_dims)
            )
            for layer in self.shape_feature_mlp:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight, a=5.0**0.5)
        self.build_loss_func()

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)

    def build_loss_func(self):
        # frequency = np.load('ScanObjectNN-weight.npy')
        # inverse_weight = (1/frequency)/sum(1/frequency)
        # manual_weight = torch.Tensor(inverse_weight)
        # self.loss_ce = nn.CrossEntropyLoss(weight=manual_weight, label_smoothing=0) #0.2
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder') :
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Transformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Transformer'
                )

            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
        else:
            print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts, batch_idx=None):
        visualize = False

        shape_feature = None
        if self.shift_net:
            if visualize == True:
                task='shift'
                import os
                os.makedirs(f'./visualization/{task}',exist_ok=True)
                np.save(f'./visualization/{task}/before_points-{batch_idx}', pts.detach().cpu().numpy())
            pts, shape_feature = self.shift_net(pts, require_global_feature=True)
            shape_feature = self.shape_feature_mlp(shape_feature)

            if visualize == True:
                task='shift'
                import os
                os.makedirs(f'./visualization/{task}',exist_ok=True)
                np.save(f'./visualization/{task}/after_points-{batch_idx}', pts.detach().cpu().numpy())
                
                np.save(f'./visualization/{task}/shape_feature-{batch_idx}', shape_feature.detach().cpu().numpy())
            shape_feature = shape_feature[:,None,:]
        if self.point_prompt:
            pts = self.point_prompt(pts) # [batch_size, 2048+20, 3]

            if visualize == True:
                task='attention'
                import os
                os.makedirs(f'./visualization/{task}',exist_ok=True)
                np.save(f'./visualization/{task}/prompt-{batch_idx}', self.point_prompt.points.detach().cpu().numpy())

        neighborhood, center = self.group_divider(pts)

        if visualize == True:
            np.save(f'./visualization/{task}/center-{batch_idx}', center.detach().cpu().numpy())
            np.save(f'./visualization/{task}/neighborhood-{batch_idx}', neighborhood.detach().cpu().numpy())
            # np.save(f'./visualization/{task}/prompt-{batch_idx}', self.point_prompt.points.detach().cpu().numpy())

        group_input_tokens = self.encoder(neighborhood)  # B G N
        
        level2_neighborhood, level2_center, level1_idx, level2_idx = self.level2_group_divider(center, require_index=True)

        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)
        token_pos = self.pos_embed(self.prompt_cor)
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = self.blocks(x, pos, global_feature = shape_feature, token_position = token_pos, level1_center=center, level1_index=level1_idx, level2_center=level2_center, level2_index=level2_idx, batch_idx=batch_idx)
        x = self.norm(x)
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0], shape_feature[:,0]], dim=-1)
        ret = self.cls_head_finetune(concat_f)
        return ret