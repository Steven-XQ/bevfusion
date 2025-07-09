import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet.models import BACKBONES
from torch.nn import functional as F
from mmcv.cnn import ConvModule

__all__ = ["MoENetwork"]

class AttentionRouter(nn.Module):
    def __init__(self, in_channels, num_experts, embed_dim=128, num_heads=4):
        super(AttentionRouter, self).__init__()

        self.conv1 = ConvModule(in_channels, 32, 3, stride=2, padding=1, conv_cfg=dict(type='Conv2d'), norm_cfg=dict(type='BN2d'), act_cfg=dict(type='PReLU'))

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = ConvModule(32, 64, 3, stride=2, padding=1, conv_cfg=dict(type='Conv2d'), norm_cfg=dict(type='BN2d'), act_cfg=dict(type='PReLU'))

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = ConvModule(64, embed_dim, 3, stride=2, padding=1, conv_cfg=dict(type='Conv2d'), norm_cfg=dict(type='BN2d'), act_cfg=dict(type='PReLU'))

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.norm = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.PReLU(),

            nn.Linear(embed_dim, embed_dim // 2),
            nn.PReLU(),

            nn.Linear(embed_dim // 2, num_experts)
        )

    def forward(self, x):
        # print(f"H: {x.size(2)}, W: {x.size(3)}")
        # x = self.proj(x)  # [B, embed_dim, H=256, W=704]

        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.pool3(x)
        # print(f"X: {x.size()}")

        # x = F.adaptive_avg_pool2d(x, (16, 16))

        x = x.flatten(2).transpose(1, 2)  # [B, 256, embed_dim]

        # print(f"X: {x.size()}")

        attn_out, _ = self.attn(x, x, x)  # [B, 256, embed_dim]

        attn_out = self.norm(attn_out)

        pooled = attn_out.mean(dim=1)  # [B, embed_dim]

        out = self.mlp(pooled)  # [B, num_experts]

        return F.softmax(out, dim=-1)


class BasicRouter(nn.Module):
    
    def __init__(self, in_channels, num_experts):
        super(BasicRouter, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.mlp = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.LayerNorm(256),

            nn.Linear(256, num_experts)
        )
    
    def forward(self, x):
        out = self.encoder(x)

        out = out.view(out.size(0), -1)

        out = self.mlp(out)

        return F.softmax(out, dim=-1)


class MoE(BaseModule):
    def __init__(self, num_experts, in_channels, experts_cfg, router, init_cfg=None):
        super(MoE, self).__init__(init_cfg)

        # 创建experts
        self.experts = nn.ModuleList([BACKBONES.build(cfg) for cfg in experts_cfg])

        # 创建router
        if router['type'] == 'BasicRouter':
            self.router = BasicRouter(in_channels, num_experts)
        elif router['type'] == 'AttentionRouter':
            self.router = AttentionRouter(in_channels, num_experts, router['embed_dim'], router['num_heads'])
        else:
            raise ValueError(f"Unknown router type: {router['type']}")

        self.k = router['k']
        
        # 做卷积，让experts输出channel数保持一致
        self.conv_swint_1 = nn.Conv2d(192, 512, kernel_size=1, stride=1, padding=0)
        self.conv_swint_2 = nn.Conv2d(384, 1024, kernel_size=1, stride=1, padding=0)
        self.conv_swint_3 = nn.Conv2d(768, 2048, kernel_size=1, stride=1, padding=0)

        self.conv_pvt_1 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)
        self.conv_pvt_2 = nn.Conv2d(320, 1024, kernel_size=1, stride=1, padding=0)
        self.conv_pvt_3 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        B = x.size(0)
        final_output_1 = torch.zeros((B, 512, 32, 88), device=x.device)
        final_output_2 = torch.zeros((B, 1024, 16, 44), device=x.device)
        final_output_3 = torch.zeros((B, 2048, 8, 22), device=x.device)

        # 获取专家权重
        routing_probs = self.router(x)  # Shape: [batch_size, num_experts]
        print(routing_probs)
        self.last_routing_probs = routing_probs

        if self.training:
            # Use all experts
            weights = routing_probs
            indices = torch.arange(routing_probs.size(1), device=routing_probs.device).repeat(routing_probs.size(0), 1)
        else:
            # Use top-k experts
            weights, indices = torch.topk(routing_probs, k=self.k, dim=-1)
            weights = weights / weights.sum(dim=-1, keepdim=True)

        expert_indices, counts = torch.unique(indices, return_counts=True)
        for i, count in zip(expert_indices, counts):
            print(f"Expert {i}: {count} times")
        print()
        
        for i, expert in enumerate(self.experts):
            idx, top = torch.where(indices == i)

            if idx.numel() == 0:
                continue
            
            # 获取专家输出
            expert_output = expert(x[idx])
            
            # 调整输出的维度
            expert_type = type(expert).__name__
            if expert_type == "SwinTransformer":
                expert_output_1 = self.conv_swint_1(expert_output[0])
                expert_output_2 = self.conv_swint_2(expert_output[1])
                expert_output_3 = self.conv_swint_3(expert_output[2])
            elif expert_type == "ResNet":
                  expert_output_1 = expert_output[0]
                  expert_output_2 = expert_output[1]
                  expert_output_3 = expert_output[2]
            elif expert_type == "PyramidVisionTransformer":
                expert_output_1 = self.conv_pvt_1(expert_output[0])
                expert_output_2 = self.conv_pvt_2(expert_output[1])
                expert_output_3 = self.conv_pvt_3(expert_output[2])
            else:
                raise ValueError(f"Unsupported expert type {expert_type}")
            
            # 专家输出与权重相乘
            w = weights[idx, top].view(-1, 1, 1, 1)
            final_output_1[idx] += expert_output_1 * w
            final_output_2[idx] += expert_output_2 * w
            final_output_3[idx] += expert_output_3 * w

        # 返回最终结果
        return [final_output_1, final_output_2, final_output_3]

@BACKBONES.register_module()
class MoENetwork(BaseModule):
    def __init__(self, num_experts, in_channels, experts_cfg, router, init_cfg=None):
        super(MoENetwork, self).__init__(init_cfg)
        self.moe = MoE(num_experts, in_channels, experts_cfg, router)
    
    def forward(self, x):
        return self.moe(x)
    
    def get_routing_probs(self):
        return self.moe.last_routing_probs
