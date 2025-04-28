import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet.models import BACKBONES
from mmcv.cnn import build_norm_layer
from torch.nn import functional as F

__all__ = ["MoENetwork"]

class MLPRouter(nn.Module):
    
    def __init__(self, in_channels, num_experts):
        super(MLPRouter, self).__init__()

        self.conv = nn.Conv2d(in_channels, 512, kernel_size=3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_experts)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.pool(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)

        logits = F.softmax(out, dim=-1)
        return logits

class MoE(BaseModule):
    def __init__(self, num_experts, in_channels, experts_cfg, router=None, init_cfg=None):
        super(MoE, self).__init__(init_cfg)

        # 创建experts
        self.experts = nn.ModuleList([BACKBONES.build(cfg) for cfg in experts_cfg])

        # 创建router
        if router is None:
            self.router = MLPRouter(in_channels, num_experts)
        else:
            self.router = MLPRouter(**router)
        
        # 做卷积，让resnet在三个阶段的输出与swin保持一致
        # self.conv_swin_1 = nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0)
        # self.conv_swin_2 = nn.Conv2d(384, 384, kernel_size=1, stride=1, padding=0)
        # self.conv_swin_3 = nn.Conv2d(768, 768, kernel_size=1, stride=1, padding=0)

        self.conv_resnet_1 = nn.Conv2d(512, 192, kernel_size=1, stride=1, padding=0)  
        self.conv_resnet_2 = nn.Conv2d(1024, 384, kernel_size=1, stride=1, padding=0) 
        self.conv_resnet_3 = nn.Conv2d(2048, 768, kernel_size=1, stride=1, padding=0) 

        self.conv_pvt_1 = nn.Conv2d(128, 192, kernel_size=1, stride=1, padding=0)  
        self.conv_pvt_2 = nn.Conv2d(320, 384, kernel_size=1, stride=1, padding=0) 
        self.conv_pvt_3 = nn.Conv2d(512, 768, kernel_size=1, stride=1, padding=0) 

    def forward(self, x):
        final_output_1 = torch.zeros((6, 192, 32, 88), device=x.device)
        final_output_2 = torch.zeros((6, 384, 16, 44), device=x.device)
        final_output_3 = torch.zeros((6, 768, 8, 22), device=x.device)

        # 获取专家权重
        routing_probs = self.router(x)  # Shape: [batch_size, num_experts]
        weights, indices = torch.topk(routing_probs, k=2, dim=-1)
        # print(f"routing_probs: {routing_probs}")
        
        # expert_outputs = []
        for i, expert in enumerate(self.experts):
            idx, top = torch.where(indices == i)
            # 获取专家输出
            expert_output = expert(x[idx])  
            
            # 调整输出的维度
            if i == 0:  # SwinTransformer 
                # expert_output_1 = self.conv_swin_1(expert_output[0])  
                # expert_output_2 = self.conv_swin_2(expert_output[1]) 
                # expert_output_3 = self.conv_swin_3(expert_output[2]) 
                expert_output_1 = expert_output[0]
                expert_output_2 = expert_output[1]
                expert_output_3 = expert_output[2]
            elif i == 1 or i == 2:  # ResNet50 & resnet101
                expert_output_1 = self.conv_resnet_1(expert_output[0])  
                expert_output_2 = self.conv_resnet_2(expert_output[1])
                expert_output_3 = self.conv_resnet_3(expert_output[2]) 
            elif i == 3:
                expert_output_1 = self.conv_pvt_1(expert_output[0])  
                expert_output_2 = self.conv_pvt_2(expert_output[1])
                expert_output_3 = self.conv_pvt_3(expert_output[2]) 
            
            # 专家输出与权重相乘
            final_output_1[idx] += expert_output_1 * weights[idx, top].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            final_output_2[idx] += expert_output_2 * weights[idx, top].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            final_output_3[idx] += expert_output_3 * weights[idx, top].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            
        
        # # 专家的结果相加
        # final_output_1 = torch.sum([output[0] for output in expert_outputs])
        # final_output_2 = torch.sum([output[1] for output in expert_outputs])
        # final_output_3 = torch.sum([output[2] for output in expert_outputs])

        # 返回最终结果
        return [final_output_1, final_output_2, final_output_3]

@BACKBONES.register_module()
class MoENetwork(BaseModule):
    def __init__(self, num_experts=4, in_channels=3, experts_cfg=None, router=None, init_cfg=None):
        super(MoENetwork, self).__init__(init_cfg)
        self.moe = MoE(num_experts, in_channels, experts_cfg, router)
    
    def forward(self, x):
        return self.moe(x)
