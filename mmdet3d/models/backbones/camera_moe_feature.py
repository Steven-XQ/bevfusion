import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet.models import BACKBONES
from torch.nn import functional as F

__all__ = ["MoENetwork"]

class AttentionRouter(nn.Module):
    def __init__(self, in_channels, num_experts, embed_dim=128, num_heads=4):
        super(AttentionRouter, self).__init__()

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.norm = nn.LayerNorm(embed_dim)

        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(embed_dim // 2),

            nn.Linear(embed_dim // 2, num_experts)
        )

    def forward(self, x):
        x = self.proj(x)  # [B, embed_dim, H, W]

        x = x.flatten(2).transpose(1, 2)  # [B, H*W, embed_dim]

        x = self.norm(x)

        attn_out, _ = self.attn(x, x, x)  # [B, H*W, embed_dim]

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
        
        # 做卷积，让resnet在三个阶段的输出与swin保持一致
        self.conv_resnet_1 = nn.Conv2d(512, 192, kernel_size=1, stride=1, padding=0)
        self.conv_resnet_2 = nn.Conv2d(1024, 384, kernel_size=1, stride=1, padding=0)
        self.conv_resnet_3 = nn.Conv2d(2048, 768, kernel_size=1, stride=1, padding=0)

        self.conv_pvt_1 = nn.Conv2d(128, 192, kernel_size=1, stride=1, padding=0)
        self.conv_pvt_2 = nn.Conv2d(320, 384, kernel_size=1, stride=1, padding=0)
        self.conv_pvt_3 = nn.Conv2d(512, 768, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        B = x.size(0)
        final_output_1 = torch.zeros((B, 192, 32, 88), device=x.device)
        final_output_2 = torch.zeros((B, 384, 16, 44), device=x.device)
        final_output_3 = torch.zeros((B, 768, 8, 22), device=x.device)

        # 获取专家权重
        routing_probs = self.router(x)  # Shape: [batch_size, num_experts]
        weights, indices = torch.topk(routing_probs, k=self.k, dim=-1)
        
        for i, expert in enumerate(self.experts):
            idx, top = torch.where(indices == i)

            if idx.numel() == 0:
                continue
            
            # 获取专家输出
            expert_output = expert(x[idx])
            
            # 调整输出的维度
            expert_type = type(expert).__name__
            if expert_type == "SwinTransformer":
                expert_output_1 = expert_output[0]
                expert_output_2 = expert_output[1]
                expert_output_3 = expert_output[2]
            elif expert_type == "ResNet":  # ResNet50 & resnet101
                expert_output_1 = self.conv_resnet_1(expert_output[0])
                expert_output_2 = self.conv_resnet_2(expert_output[1])
                expert_output_3 = self.conv_resnet_3(expert_output[2])
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
