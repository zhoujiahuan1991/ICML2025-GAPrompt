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
from .PointPrompt import ShiftNet, PointPrompt, Group, propagate, pooling
from .Point_MAE_segment import PointNetFeaturePropagation, Encoder, Mlp, Attention, get_loss
from .prompt_MAE import QuickGELU, Adapter


# Block with prompt
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_tokens=10, config=None, use_prompt=False, use_adapter=False):
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
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.prompt_embeddings = None
        if use_prompt:
            self.prompt_dropout = nn.Dropout(0.1)
            self.num_tokens = num_tokens
            self.prompt_embeddings = nn.Parameter(torch.zeros(self.num_tokens, dim))
        self.adapter=None
        if use_adapter:
            self.adapter = Adapter(embed_dims=dim, reduction_dims=16)
        self.out_transform = nn.Sequential(
                                nn.BatchNorm1d(dim),
                                nn.GELU()
                            )

    def forward(self, x, global_feature=None, token_position=None, layer_id=None, level1_center=None, level1_index=None, level2_center=None, level2_index=None, batch_idx=None):
        if global_feature is not None and self.prompt_embeddings is not None:
            token_prompt = self.prompt_dropout(self.prompt_embeddings.repeat(x.shape[0], 1, 1))
            token_prompt = token_prompt + 0.5*global_feature.repeat([1, self.num_tokens, 1]) + token_position.repeat(x.shape[0], 1, 1)
            x = torch.cat((token_prompt, x), 1)
        # elif token_position is not None and layer_id<self.config.prompt_depth:
        #     token_prompt = self.prompt_dropout(self.prompt_embeddings.repeat(x.shape[0], 1, 1))
        #     token_prompt = token_prompt + token_position.repeat(x.shape[0], 1, 1)
        #     x = torch.cat((token_prompt, x), 1)

        if self.config.propagation_type == 'permutation_before_attention':
            B,G,_ = x.shape
            propagate_range = level1_center.shape[1]
            x_neighborhoods = x.reshape(B*G, -1)[level1_index, :].reshape(B*level2_center.shape[1], -1, self.dim)
            x_centers = x.reshape(B*G, -1)[level2_index, :].reshape(B, level2_center.shape[1], self.dim)
            x_neighborhoods = self.drop_path(x_neighborhoods)+x_neighborhoods
            vis_x = pooling(x_neighborhoods.reshape(B, level2_center.shape[1], -1, self.dim), transform=self.out_transform)+0.3*x_centers
            x[:,-propagate_range:] = propagate(xyz1=level1_center, xyz2=level2_center, points1=x[:,-propagate_range:], points2=vis_x)
 
        elif self.config.propagation_type == 'replacement_before_attention':
            B,G,_ = x.shape
            propagate_range = level1_center.shape[1]
            if G > propagate_range:
                token_prompt, x = x[:, :-propagate_range], x[:,-propagate_range:]
                replace_x = x.clone()
                x = torch.concat([token_prompt, x], dim=1)
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

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if self.config.propagation_type == 'permutation_after_attention':
            B,G,_ = x.shape
            propagate_range = level1_center.shape[1]
            x_neighborhoods = x.reshape(B*G, -1)[level1_index, :].reshape(B*level2_center.shape[1], -1, self.dim)
            x_centers = x.reshape(B*G, -1)[level2_index, :].reshape(B, level2_center.shape[1], self.dim)
            x_neighborhoods = self.drop_path(x_neighborhoods)+x_neighborhoods
            vis_x = pooling(x_neighborhoods.reshape(B, level2_center.shape[1], -1, self.dim), transform=self.out_transform)+0.3*x_centers
            x[:,-propagate_range:] = propagate(xyz1=level1_center, xyz2=level2_center, points1=x[:,-propagate_range:], points2=vis_x)
        elif self.config.propagation_type == 'replacement_after_attention':
            B,G,_ = x.shape
            propagate_range = level1_center.shape[1]
            if G > propagate_range:
                token_prompt, x = x[:, :-propagate_range], x[:,-propagate_range:]
                replace_x = x.clone()
                x = torch.concat([token_prompt, x], dim=1)
                replace_x[:, -self.num_tokens:] = token_prompt
                x_neighborhoods = replace_x.reshape(B*propagate_range, -1)[level1_index, :].reshape(B*level2_center.shape[1], -1, self.dim)
                x_centers = replace_x.reshape(B*propagate_range, -1)[level2_index, :].reshape(B, level2_center.shape[1], self.dim)
            else:
                x_neighborhoods = x.reshape(B*propagate_range, -1)[level1_index, :].reshape(B*level2_center.shape[1], -1, self.dim)
                x_centers = x.reshape(B*propagate_range, -1)[level2_index, :].reshape(B, level2_center.shape[1], self.dim)
             
            x_neighborhoods = self.drop_path(x_neighborhoods)+x_neighborhoods
            vis_x = pooling(x_neighborhoods.reshape(B, level2_center.shape[1], -1, self.dim), transform=self.out_transform)+0.3*x_centers
            x[:,-propagate_range:] = propagate(xyz1=level1_center, xyz2=level2_center, points1=x[:,-propagate_range:], points2=vis_x)
        
        if self.adapter is not None:
            if global_feature is not None:
                x = x + self.adapter(x+global_feature*0.5)
            else:
                x = x + self.adapter(x)
        if self.prompt_embeddings is not None:
            prompt_after_attn = x[:,:self.num_tokens,:]
            x = x[:,self.num_tokens:,:]
            return prompt_after_attn, x
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
                config=config,
                use_prompt= i in [3, 7, 11], 
                use_adapter= True, #i in [3, 7, 11]
                )
            for i in range(depth)])

    def forward(self, x, pos, global_feature=None, token_position=None, level1_center=None, level1_index=None, level2_center=None, level2_index=None, batch_idx=None):
        feature_list = []
        prompt_list = []
        fetch_idx = [3, 7, 11]

        for idx, block in enumerate(self.blocks):
            if idx in fetch_idx:
                prompt_after_attn, x = block(x + pos, global_feature=global_feature, token_position=token_position, layer_id=idx, level1_center=level1_center, level1_index=level1_index, level2_center=level2_center, level2_index=level2_index)
                feature_list.append(x)
                prompt_list.append(prompt_after_attn)
            else:
                x = block(x + pos, global_feature=global_feature, token_position=token_position, layer_id=idx, level1_center=level1_center, level1_index=level1_index, level2_center=level2_center, level2_index=level2_index)
        return feature_list, prompt_list
    
# prompt tuning model with point promt, token prompt, and shiftnet
@MODELS.register_module()
class PointTransformer_pointtokenprompt_sem_seg(nn.Module):
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
        self.prompt_cor = nn.Parameter(torch.zeros(config.num_tokens, 3))
        trunc_normal_(self.prompt_cor, std=.06)
        
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
            config=self.config
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        # self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=True),
        #                            nn.BatchNorm1d(64),
        #                            nn.LeakyReLU(0.2),
        #                            nn.Conv1d(64, 128, kernel_size=1, bias=True),
        #                            nn.BatchNorm1d(128),
        #                            nn.LeakyReLU(0.2),
        #                            )
        
        self.propagation_0 = PointNetFeaturePropagation(in_channel=1152 + 3, mlp=[384*4, 1024], interpolate_neighbors=3)
        # self.group_divider1 = Group(num_group=self.num_group*3, group_size=self.group_size//2)
        # self.propagation_1 = PointNetFeaturePropagation(in_channel=384 + 75, mlp=[384, 384], interpolate_neighbors=5)
        

        self.seg_head = nn.Sequential(
            nn.Conv1d(1024+384*6, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, self.cls_dim, 1)
        )

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)
        self.get_loss = get_loss().cuda()

        self.norm = nn.LayerNorm(self.trans_dim)
        
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

    def forward(self, pts):
        B, N, _ = pts.shape
        shape_feature = None
        if self.shift_net:
            pts, shape_feature = self.shift_net(pts, require_global_feature=True)
            shape_feature = self.shape_feature_mlp(shape_feature)
            shape_feature = shape_feature[:,None,:]
        if self.point_prompt:
            pts = self.point_prompt(pts) # [batch_size, 2048+20, 3]
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N
        x = group_input_tokens
        level2_neighborhood, level2_center, level1_idx, level2_idx = self.level2_group_divider(center, require_index=True)

        # transformer
        pos = self.pos_embed(center)
        token_pos = self.pos_embed(self.prompt_cor)
        feature_list, prompt_list = self.blocks(x, pos, global_feature = shape_feature, token_position = token_pos, level1_center=center, level1_index=level1_idx, level2_center=level2_center, level2_index=level2_idx)
        feature_list = [self.norm(x).contiguous() for x in feature_list]
        prompt_list = [self.norm(prompt).contiguous() for prompt in prompt_list]
        x = torch.cat((feature_list[0],feature_list[1],feature_list[2]), dim=-1) #1152
        prompt = torch.cat((prompt_list[0],prompt_list[1],prompt_list[2]), dim=-1) #1152
        prompt_feature = prompt.repeat(1, N, 1)
        x_max = torch.max(x,1)[0]
        x_avg = torch.mean(x,1)
        x_max_feature = x_max.view(B, -1).unsqueeze(-2).repeat(1, N, 1)
        x_avg_feature = x_avg.view(B, -1).unsqueeze(-2).repeat(1, N, 1)
        # cls_label_one_hot = cls_label.view(B, 16, 1)
        # cls_label_feature = self.label_conv(cls_label_one_hot).transpose(-1,-2).repeat(1, N, 1) # 128
        # x_global_feature = torch.cat((x_max_feature, x_avg_feature, cls_label_feature), -1) #1152*2 + 128
        x_global_feature = torch.cat((x_max_feature, prompt_feature), -1) #1152*2 + 128
        torch.cuda.empty_cache()
        if self.point_prompt:
            pts = pts[:,:-self.point_prompt.point_number]
        f_level_0 = self.propagation_0(pts, center, pts, x)
        x = torch.cat((f_level_0, x_global_feature), -1)
        x = self.seg_head(x.transpose(-1,-2))
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x