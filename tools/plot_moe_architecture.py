import sys
sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks import *

arch = [
    to_head('..'),
    to_cor(),
    to_begin(),

    # Input image
    to_input('../input.jpg', name="image"),

    # Input feature map
    to_Conv(name='input', xlabel=3, ylabel='H', zlabel='W', offset="(0.8,0,0)", to="(0,0,0)",
            width=4, height=35, depth=35, caption=r"{Input\\{[B, C, H, W]}}"),

    # 1x1 Conv projection to embedding dim
    to_Conv(name='proj', xlabel=128, ylabel='H', zlabel='W', offset="(4,0,0)", to="(input-east)",
            width=12, height=35, depth=35, caption='{1x1 Conv\n→ Embedding\n[B, D, H, W]}'),
    to_connection('input', 'proj'),

    # Flatten + Transpose
    to_Conv(name='flatten', xlabel=128, ylabel='H*W', offset="(3,0,0)", to="(proj-east)", width=12, height=100, depth=1,
               caption='{Flatten + Transpose\n[B, H * W, D]}'),
    to_connection('proj', 'flatten'),

    # LayerNorm
    to_SoftMax(name='norm', offset="(2.5,0,0)", to="(flatten-east)", width=1, height=1, depth=1,
               caption='LayerNorm'),
    to_connection('flatten', 'norm'),

    # Multi-Head Self-Attention
    to_Conv(name='attn', xlabel=128, ylabel='H*W', offset="(2.5,0,0)", to="(norm-east)",
                    width=12, height=100, depth=1, caption=r'{{Multi-Head Self Attention [B, H * W, D]}}'),
    to_connection('norm', 'attn'),

    # Mean Pooling across tokens
    to_Conv(name='pool', ylabel=128, offset="(2.5,0,0)", to="(attn-east)", width=1, height=40, depth=1,
            caption='{Mean Pool\n[B, D]}'),
    to_connection('attn', 'pool'),

    # MLP: Linear → ReLU → LayerNorm → Linear
    to_Conv(name='mlp1', ylabel=64, offset="(2.5,0,0)", to="(pool-east)", width=1, height=20, depth=1,
                      caption='{Linear + ReLU\n[B, D // 2]}'),
    to_connection('pool', 'mlp1'),

    to_SoftMax(name='norm2', offset="(2.5,0,0)", to="(mlp1-east)", width=1, height=1, depth=1,
               caption='LayerNorm'),
    to_connection('mlp1', 'norm2'),

    to_Conv(name='mlp2', ylabel=4, offset="(2.5,0,0)", to="(norm2-east)", width=1, height=5, depth=1,
                      caption=r'{Linear\\{[B, num\_experts]}}'),
    to_connection('norm2', 'mlp2'),

    # Softmax
    to_SoftMax(name='softmax', offset="(2.5,0,0)", to="(mlp2-east)", width=1, height=1, depth=1,
               caption='Softmax'),
    to_connection('mlp2', 'softmax'),

    # Expert 1
    to_Pool(
        name='expert1', caption='Expert 1\n(SwinT)', offset="(5,10.5,0)", to="(softmax-east)", width=15, height=15, depth=15, opacity=0.5
    ),

    # Expert 2
    to_Pool(
        name='expert2', caption='Expert 2\n(Resnet50)', offset="(5,3.5,0)", to="(softmax-east)", width=15, height=15, depth=15, opacity=0.5
    ),
    to_connection('softmax', 'expert2', dashed=True),

    # Expert 3
    to_Pool(
        name='expert3', caption='Expert 3\n(Resnet101)', offset="(5,-3.5,0)", to="(softmax-east)", width=15, height=15, depth=15, opacity=0.5
    ),

    # Expert 4
    to_Pool(
        name='expert4', caption='Expert 4\n(PVT)', offset="(5,-10.5,0)", to="(softmax-east)", width=15, height=15, depth=15, opacity=0.5
    ),

    to_skip("input", "expert2", pos1=1.39, pos2=1.4, top=True),

    to_Ball(name='mult', offset="(3,0,0)", to="(expert2-east)", radius=2.5, opacity=0.6, logo=r'\times'),
    to_connection('expert2', 'mult', 'blue'),
    to_skip('softmax', 'mult', pos1=4, pos2=4.65, top=False),

    # Merge
    to_Conv(name='stage1', caption='Stage 1', xlabel=192, offset="(3,-3,0)", to="(mult-east)", width=8, height=40, depth=40),
    to_Conv(name='stage2', caption='Stage 2', xlabel=384, offset="(0,0,0)", to="(stage1-east)", width=16, height=40, depth=40),
    to_Conv(name='stage3', caption='Stage 3', xlabel=768, offset="(0,0,0)", to="(stage2-east)", width=32, height=40, depth=40),

    to_connection('mult', 'stage1', color='blue'),

    # Neck
    to_ConvSoftMax(
        name="Neck", caption="Neck\n(GeneralizedLSSFPN)", s_filer=28,
        offset="(5,0,0)", to="(stage3-east)", width=10,
        height=40, depth=40
    ),
    to_connection("stage3", "Neck", color='blue'),

    to_raw(r"""
           \draw[dashed, thick, green] 
  ([xshift=-2.2cm, yshift=8cm] proj-northwest) 
    rectangle 
  ([xshift=0.8cm, yshift=-12cm] softmax-southeast);
           """),

    to_raw(r"""
           \usetikzlibrary{calc}
           \node[above, text=red, font=\bfseries] at ($(proj-northeast) + (-1,7)$) {\huge AttentionRouter};
           """),

    to_raw(r"""
           \usetikzlibrary{calc}
           \node[above, text=red, font=\bfseries] at ($(stage3-northwest) + (1,2)$) {\LARGE Feature Maps};
           """),

    to_raw(r"""
           \usetikzlibrary{calc}
           \node[above, text=red, font=\bfseries] at ($(softmax-north) + (2.2,2)$) {\LARGE top 1};
           """),

    to_raw(r"""
           \usetikzlibrary{calc}
           \node[above, text=red, font=\bfseries] at ($(softmax-south) + (2.5,-1.5)$) {\LARGE weight};
           """),

    to_end()
]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')

if __name__ == '__main__':
    main()
