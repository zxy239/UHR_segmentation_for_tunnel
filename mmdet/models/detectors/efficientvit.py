# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

import torch
import torch.nn as nn
from mmengine.runner.checkpoint import load_state_dict
from mmseg.models.builder import BACKBONES
from .Lightweight_efficientvit.utils import load_state_dict_from_file, build_kwargs_from_config
from .Lightweight_efficientvit.nn import (
    ConvLayer,
    DSConv,
    EfficientViTBlock,
    FusedMBConv,
    IdentityLayer,
    MBConv,
    OpSequential,
    ResBlock,
    ResidualBlock,
)

__all__ = [
    "EfficientViTBackbone",
    "efficientvit_backbone_b0",
    "efficientvit_backbone_b1",
    "efficientvit_backbone_b2",
    "efficientvit_backbone_b3",
    "EfficientViTLargeBackbone",
    "efficientvit_backbone_l0",
    "efficientvit_backbone_l1",
    "efficientvit_backbone_l2",
    "efficientvit_backbone_l3",
]

#@BACKBONES.register_module()
class EfficientViTBackbone(nn.Module):
    def __init__(self,
        # width_list: list[int],
        # depth_list: list[int],
        
        # width_list=[8, 16, 32, 64],
        # depth_list=[1, 2, 2, 2],
        # dim=16, b0
        # width_list=[16, 32, 64, 128],
        # depth_list=[1, 2, 3, 3],
        # dim=16, b1
        # width_list=[24, 48, 96, 192],
        # depth_list=[1, 3, 4, 4],
        # dim=32, b2
        
        width_list=[8, 16, 32, 64],
        depth_list=[1, 2, 2, 2],
        dim=16,
        expand_ratio=4,
        norm="bn2d",
        act_func="hswish",
        inchannels=3,
        pretrained=None,
        ):
        #super(EfficientViTBackbone, self).__init__()
        super().__init__()
        self.pretrained = pretrained
        self.inchannels = inchannels
        # 3x1024x2048
        self.width_list = []
        # input stem
        self.input_stem = [
            ConvLayer(
                in_channels=self.inchannels,
                out_channels=width_list[0],
                stride=2,
                norm=norm,
                act_func=act_func,
            )
        ]
        # 16x512x1024
        for _ in range(depth_list[0]):
            block = self.build_local_block( # DSconv
                in_channels=width_list[0],
                out_channels=width_list[0],
                stride=1,
                expand_ratio=1,
                norm=norm,
                act_func=act_func,
            )
            self.input_stem.append(ResidualBlock(block, IdentityLayer()))
        in_channels = width_list[0]
        self.input_stem = OpSequential(self.input_stem)
        self.width_list.append(in_channels)

        # stages
        self.stages = []
        for w, d in zip(width_list[1:3], depth_list[1:3]): # 32,64
            stage = []
            for i in range(d): # 2 , 3
                stride = 2 if i == 0 else 1
                block = self.build_local_block(
                    in_channels=in_channels,
                    out_channels=w,
                    stride=stride,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=act_func,
                )
                block = ResidualBlock(block, IdentityLayer() if stride == 1 else None)
                stage.append(block)
                in_channels = w

            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)

        for w, d in zip(width_list[3:], depth_list[3:]):
            stage = []
            block = self.build_local_block(
                in_channels=in_channels,
                out_channels=w,
                stride=2,
                expand_ratio=expand_ratio,
                norm=norm,
                act_func=act_func,
                fewer_norm=True,
            )
            stage.append(ResidualBlock(block, None))
            in_channels = w

            #  128x64x128
            for _ in range(d):
                stage.append(
                    EfficientViTBlock(
                        in_channels=in_channels, # 128 ,
                        dim=dim,# 16
                        expand_ratio=expand_ratio,# 4
                        norm=norm,
                        act_func=act_func,
                    )
                )
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)
        self.stages = nn.ModuleList(self.stages)

        # if pretrained is not None:
        #     self.init_weights(pretrained=pretrained)

    @staticmethod
    def build_local_block(
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: float,
        norm: str,
        act_func: str,
        fewer_norm: bool = False,
    ) -> nn.Module:
        if expand_ratio == 1:
            block = DSConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
            )
        else:
            block = MBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio, # 4
                use_bias=(True, True, False) if fewer_norm else False,
                norm=(None, None, norm) if fewer_norm else norm,
                act_func=(act_func, act_func, None),
            )
        return block

    #def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
    # def forward(self, x) :
        # output_dict = {"input": x}
        # output_dict["stage0"] = x = self.input_stem(x)
        # for stage_id, stage in enumerate(self.stages, 1):
            # output_dict["stage%d" % stage_id] = x = stage(x)
        # output_dict["stage_final"] = x

        # return output_dict
        
    # def forward(self, x, stage):
        # if stage == -1:
            # output = self.input_stem(x)
        # else:
            # output = self.stages[stage](x)
        # return output
    def forward(self, x) :
        output = []
        x = self.input_stem(x)
        for stage in self.stages:
            x = stage(x)
            output.append(x)
        return output
        
    def init_weights(self, pretrained=None):
       # weight_url = weight_url or REGISTERED_CLS_MODEL.get(name, None)
        print('init_weights_effvit')
        pretrained = self.pretrained
        #print(pretrained)
        if pretrained is None:
            raise ValueError(f"Do not find the pretrained weight")
        else:
            weight = load_state_dict_from_file(pretrained)
            #model.load_state_dict(weight)

           # backbone.input_stem.op_list.0.conv.weight
            new_weight = {}
            for key in weight:
                # print(key)
                new_key = key
                if key[:8] == 'backbone':
                    new_key = key[9:]
                # print(new_key)
                if new_key == 'input_stem.op_list.0.conv.weight' and self.inchannels != 3:
                    v = weight[key]
                    v = torch.cat([v, v, v], dim=1)
                    weight[key] = v
                    # print(v.shape) # torch.Size([16, 9, 3, 3])
                    # print(weight[key].shape)

                new_weight[new_key] = weight[key]
           # print(new_weight['input_stem.op_list.0.conv.weight'].shape)

          #  raise ValueError(f"Do not find the pretrained weight")
            load_state_dict(self, new_weight, strict=False)

@BACKBONES.register_module()
def efficientvit_backbone_b0(**kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        width_list=[8, 16, 32, 64, 128],
        depth_list=[1, 2, 2, 2, 2],
        dim=16,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    return backbone

@BACKBONES.register_module()
def efficientvit_backbone_b1(**kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        width_list=[16, 32, 64, 128, 256],
        depth_list=[1, 2, 3, 3, 4],
        dim=16,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    return backbone

@BACKBONES.register_module()
def efficientvit_backbone_b2(**kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        width_list=[24, 48, 96, 192, 384],
        depth_list=[1, 3, 4, 4, 6],
        dim=32,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    return backbone

@BACKBONES.register_module()
def efficientvit_backbone_b3(**kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 4, 6, 6, 9],
        dim=32,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    return backbone


class EfficientViTLargeBackbone(nn.Module):
    def __init__(
        self,
        # width_list: list[int],
        # depth_list: list[int],
        width_list,
        depth_list,
        in_channels=3,
        qkv_dim=32,
        norm="bn2d",
        act_func="gelu",
    ) -> None:
        super().__init__()

        self.width_list = []
        self.stages = []
        # stage 0
        stage0 = [
            ConvLayer(
                in_channels=3,
                out_channels=width_list[0],
                stride=2,
                norm=norm,
                act_func=act_func,
            )
        ]
        for _ in range(depth_list[0]):
            block = self.build_local_block(
                stage_id=0,
                in_channels=width_list[0],
                out_channels=width_list[0],
                stride=1,
                expand_ratio=1,
                norm=norm,
                act_func=act_func,
            )
            stage0.append(ResidualBlock(block, IdentityLayer()))
        in_channels = width_list[0]
        self.stages.append(OpSequential(stage0))
        self.width_list.append(in_channels)

        for stage_id, (w, d) in enumerate(zip(width_list[1:4], depth_list[1:4]), start=1):
            stage = []
            for i in range(d + 1):
                stride = 2 if i == 0 else 1
                block = self.build_local_block(
                    stage_id=stage_id,
                    in_channels=in_channels,
                    out_channels=w,
                    stride=stride,
                    expand_ratio=4 if stride == 1 else 16,
                    norm=norm,
                    act_func=act_func,
                    fewer_norm=stage_id > 2,
                )
                block = ResidualBlock(block, IdentityLayer() if stride == 1 else None)
                stage.append(block)
                in_channels = w
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)

        for stage_id, (w, d) in enumerate(zip(width_list[4:], depth_list[4:]), start=4):
            stage = []
            block = self.build_local_block(
                stage_id=stage_id,
                in_channels=in_channels,
                out_channels=w,
                stride=2,
                expand_ratio=24,
                norm=norm,
                act_func=act_func,
                fewer_norm=True,
            )
            stage.append(ResidualBlock(block, None))
            in_channels = w

            for _ in range(d):
                stage.append(
                    EfficientViTBlock(
                        in_channels=in_channels,
                        dim=qkv_dim,
                        expand_ratio=6,
                        norm=norm,
                        act_func=act_func,
                    )
                )
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)
        self.stages = nn.ModuleList(self.stages)

    @staticmethod
    def build_local_block(
        stage_id: int,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: float,
        norm: str,
        act_func: str,
        fewer_norm: bool = False,
    ) -> nn.Module:
        if expand_ratio == 1:
            block = ResBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
            )
        elif stage_id <= 2:
            block = FusedMBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
            )
        else:
            block = MBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, True, False) if fewer_norm else False,
                norm=(None, None, norm) if fewer_norm else norm,
                act_func=(act_func, act_func, None),
            )
        return block

    #def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
    def forward(self, x):
        output_dict = {"input": x}
        for stage_id, stage in enumerate(self.stages):
            output_dict["stage%d" % stage_id] = x = stage(x)
        output_dict["stage_final"] = x
        return output_dict


def efficientvit_backbone_l0(**kwargs) -> EfficientViTLargeBackbone:
    backbone = EfficientViTLargeBackbone(
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 1, 1, 4, 4],
        **build_kwargs_from_config(kwargs, EfficientViTLargeBackbone),
    )
    return backbone


def efficientvit_backbone_l1(**kwargs) -> EfficientViTLargeBackbone:
    backbone = EfficientViTLargeBackbone(
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 1, 1, 6, 6],
        **build_kwargs_from_config(kwargs, EfficientViTLargeBackbone),
    )
    return backbone


def efficientvit_backbone_l2(**kwargs) -> EfficientViTLargeBackbone:
    backbone = EfficientViTLargeBackbone(
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 2, 2, 8, 8],
        **build_kwargs_from_config(kwargs, EfficientViTLargeBackbone),
    )
    return backbone


def efficientvit_backbone_l3(**kwargs) -> EfficientViTLargeBackbone:
    backbone = EfficientViTLargeBackbone(
        width_list=[64, 128, 256, 512, 1024],
        depth_list=[1, 2, 2, 8, 8],
        **build_kwargs_from_config(kwargs, EfficientViTLargeBackbone),
    )
    return backbone
