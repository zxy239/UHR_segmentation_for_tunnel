# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple

from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage import SingleStageDetector

from .efficientvit import EfficientViTBackbone
import torch
import torch.nn as nn
from functools import partial
import torch.utils.checkpoint as cp
from .ops.modules import MSDeformAttn
import torch.nn.functional as F
from mmdet.evaluation.functional import INSTANCE_OFFSET
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData, PixelData
import os
from torchvision.transforms.functional import to_pil_image

def get_reference_points(spatial_shapes, device):
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]
    return reference_points

def deform_inputs(x):
    bs, c, h, w = x.shape
    spatial_shapes = torch.as_tensor([(h // 32, w // 32)],
                                     dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // 32, w // 32)], x.device)
    deform_inputs = [reference_points, spatial_shapes, level_start_index]
    
    return deform_inputs
    # spatial_shapes = torch.as_tensor([(h // 16, w // 16)], dtype=torch.long, device=x.device)
    # level_start_index = torch.cat((spatial_shapes.new_zeros(
        # (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    # reference_points = get_reference_points([(h // 8, w // 8),
                                             # (h // 16, w // 16),
                                             # (h // 32, w // 32)], x.device)
    # deform_inputs2 = [reference_points, spatial_shapes, level_start_index]
    
    # return deform_inputs1, deform_inputs2

class CAv1(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 with_cffn=True, cffn_ratio=0.25, drop=0., drop_path=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), with_cp=False):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
                                 n_points=n_points, ratio=deform_ratio)
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        if with_cffn:
            self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, H, W):
        
        def _inner_forward(query, feat):

            attn = self.attn(self.query_norm(query), reference_points,
                             self.feat_norm(feat), spatial_shapes,
                             level_start_index, None)
            query = query + attn
    
            if self.with_cffn:
                query = query + self.drop_path(self.ffn(self.ffn_norm(query), H, W))
            return query
        
        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)
            
        return query


class CAv2(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0., with_cp=False):
        super().__init__()
        self.with_cp = with_cp
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
                                 n_points=n_points, ratio=deform_ratio)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index):
        
        def _inner_forward(query, feat):

            attn = self.attn(self.query_norm(query), reference_points,
                             self.feat_norm(feat), spatial_shapes,
                             level_start_index, None)
            return query + self.gamma * attn
        
        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)
            
        return query

@MODELS.register_module()
class MaskFormer(SingleStageDetector):
    r"""Implementation of `Per-Pixel Classification is
    NOT All You Need for Semantic Segmentation
    <https://arxiv.org/pdf/2107.06278>`_."""

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 panoptic_head: OptConfigType = None,
                 panoptic_fusion_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 context_cfg: OptConfigType = None,
                 ):
        super(SingleStageDetector, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)

        panoptic_head_ = panoptic_head.deepcopy()
        panoptic_head_.update(train_cfg=train_cfg)
        panoptic_head_.update(test_cfg=test_cfg)
        self.panoptic_head = MODELS.build(panoptic_head_)

        panoptic_fusion_head_ = panoptic_fusion_head.deepcopy()
        panoptic_fusion_head_.update(test_cfg=test_cfg)
        self.panoptic_fusion_head = MODELS.build(panoptic_fusion_head_)

        self.num_things_classes = self.panoptic_head.num_things_classes
        self.num_stuff_classes = self.panoptic_head.num_stuff_classes
        self.num_classes = self.panoptic_head.num_classes

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        self.crop_size = context_cfg['crop_size']
        self.stride_size = context_cfg['stride_size']
        self.context_size  = context_cfg['context_size']
        self.w_context = context_cfg['context']
        self.local_thr = context_cfg['local_thr']
        if self.w_context:
            self.effvit = EfficientViTBackbone(
                width_list=context_cfg['effvit_width_list'], depth_list=context_cfg['effvit_depth_list'], dim=context_cfg['effvit_dim'], pretrained=context_cfg['effvit_pre']
            )
            self.proj = nn.Conv2d(context_cfg['effvit_proj'][0], context_cfg['effvit_proj'][1], kernel_size=1)
            self.ca = CAv2(context_cfg['effvit_proj'][1], context_cfg['effvit_proj'][2])
        
    def add_context(self, x, context, HW):
        context = F.interpolate(context, size=HW, mode='bilinear', align_corners=False) #1024
        my_deform_inputs = deform_inputs(context) #1024
        context = self.effvit(context)
        
        feat_local = x[3]
        feat_context = context[3]
        feat_context = self.proj(feat_context)
        
        B, C, H, W = feat_local.shape
        feat_local = feat_local.reshape(B, C, -1).permute(0, 2, 1)
        feat_context = feat_context.reshape(B, C, -1).permute(0, 2, 1)
        
        result = self.ca(query=feat_local, reference_points=my_deform_inputs[0],
                    feat=feat_context, spatial_shapes=my_deform_inputs[1],
                    level_start_index=my_deform_inputs[2])
        result = result.permute(0, 2, 1).reshape(B, -1, H, W)
        x[3] = result
        return x
    
    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Dict[str, Tensor]:
        """
        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        batch_inputs_context = batch_inputs[1]
        batch_inputs = batch_inputs[0]
        HW = batch_inputs.size()[-2:]
        # print(batch_inputs_context.shape)
        # print(batch_inputs.shape)
        
        x = self.extract_feat(batch_inputs)
        if self.w_context:
            x = self.add_context(x, batch_inputs_context, HW)
        
        losses = self.panoptic_head.loss(x, batch_data_samples)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances' and `pred_panoptic_seg`. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).

            And the ``pred_panoptic_seg`` contains the following key

                - sem_seg (Tensor): panoptic segmentation mask, has a
                    shape (1, h, w).
        """  
        h_stride, w_stride = self.stride_size
        h_crop, w_crop = self.crop_size
        h_context, w_context = self.context_size
        batch_size, _, h_img, w_img = batch_inputs.size()
        # print(batch_inputs.shape)
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        if self.w_context:
            pad_y = (h_context-h_crop) // 2
            pad_x = (w_context-w_crop) // 2
            pad = (pad_y, pad_y, pad_x, pad_x)
            batch_inputs_context = F.pad(batch_inputs, pad, "constant", 0)
        
        device = batch_inputs.device
        global_panoptic = torch.full((h_img, w_img), self.num_classes, dtype=torch.int32, device=device)
        all_ins = []
        global_instance_id = 1
        
        recycled_ids = set()
        # print()
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                # print(h_idx, w_idx)
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = batch_inputs[:, :, y1:y2, x1:x2]
                if self.w_context:
                    crop_context = batch_inputs_context[:, :, y1:y2+(h_context-h_crop), x1:x2+(w_context-w_crop)]
                    
                patch_meta = batch_data_samples[0].metainfo.copy()
                patch_meta['img_shape'] = (h_crop, w_crop)
                patch_meta['ori_shape'] = patch_meta['img_shape']
                patch_meta['batch_input_shape'] = patch_meta['img_shape']
                patch_data_samples = [DetDataSample(metainfo=patch_meta)]
        
                HW = crop_img.size()[-2:]
                feats = self.extract_feat(crop_img)
                if self.w_context:
                    feats = self.add_context(feats, crop_context, HW)
                mask_cls_results, mask_pred_results = self.panoptic_head.predict(
                    feats, patch_data_samples)
                results_list = self.panoptic_fusion_head.predict(
                    mask_cls_results,
                    mask_pred_results,
                    patch_data_samples,
                    rescale=False)
                
                pan_result = results_list[0]['pan_results'].sem_seg[0]  # (H, W)
                
                unique_ids = torch.unique(pan_result)
                for uid in unique_ids:
                    if uid == self.num_classes:
                        continue  
                    class_id = uid % INSTANCE_OFFSET
                    if class_id < self.num_things_classes:
                        mask = (pan_result == uid)
                        global_patch = global_panoptic[y1:y2, x1:x2]
                        overlap = global_patch[mask]
                        candidate = overlap[(overlap != self.num_classes) & (overlap % INSTANCE_OFFSET == class_id)]
                        if candidate.numel() > 0:
                            candidate_ids = torch.unique(candidate)
                            new_uid = int(candidate_ids.min().item())
                            for cid in candidate_ids:
                                global_panoptic[global_panoptic == cid] = new_uid
                            instance_ids = (candidate_ids / INSTANCE_OFFSET).int().tolist()
                            recycled_ids.update(instance_ids)
                            new_instance_id = int(new_uid / INSTANCE_OFFSET)
                            recycled_ids.discard(new_instance_id)
                        else:
                            if recycled_ids:
                                new_uid = class_id + min(recycled_ids) * INSTANCE_OFFSET
                                recycled_ids.remove(min(recycled_ids))
                            else:
                                new_uid = class_id + global_instance_id * INSTANCE_OFFSET
                                global_instance_id += 1
                    else:
                        new_uid = uid
                    pan_result[pan_result == uid] = new_uid
                global_panoptic[y1:y2, x1:x2] = torch.where(
                    (pan_result != self.num_classes) & (global_panoptic[y1:y2, x1:x2] == self.num_classes),# pan_result != self.num_classes,
                    pan_result,
                    global_panoptic[y1:y2, x1:x2]
                )
                
        final_panoptic = PixelData(sem_seg=global_panoptic.unsqueeze(0))
        final_results = [{'pan_results': final_panoptic}]
        results = self.add_pred_to_datasample(batch_data_samples, final_results)

        return results
        # feats = self.extract_feat(batch_inputs)
        # mask_cls_results, mask_pred_results = self.panoptic_head.predict(
            # feats, batch_data_samples)
        # results_list = self.panoptic_fusion_head.predict(
            # mask_cls_results,
            # mask_pred_results,
            # batch_data_samples,
            # rescale=False)
            # rescale=rescale)
        # results = self.add_pred_to_datasample(batch_data_samples, results_list)
        # return results

    def add_pred_to_datasample(self, data_samples: SampleList,
                               results_list: List[dict]) -> SampleList:
        """Add predictions to `DetDataSample`.

        Args:
            data_samples (list[:obj:`DetDataSample`], optional): A batch of
                data samples that contain annotations and predictions.
            results_list (List[dict]): Instance segmentation, segmantic
                segmentation and panoptic segmentation results.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances' and `pred_panoptic_seg`. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).

            And the ``pred_panoptic_seg`` contains the following key

                - sem_seg (Tensor): panoptic segmentation mask, has a
                    shape (1, h, w).
        """
        for data_sample, pred_results in zip(data_samples, results_list):
            if 'pan_results' in pred_results:
                data_sample.pred_panoptic_seg = pred_results['pan_results']

            if 'ins_results' in pred_results:
                data_sample.pred_instances = pred_results['ins_results']

            assert 'sem_results' not in pred_results, 'segmantic ' \
                'segmentation results are not supported yet.'

        return data_samples

    def _forward(self, batch_inputs: Tensor,
                 batch_data_samples: SampleList) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            tuple[List[Tensor]]: A tuple of features from ``panoptic_head``
            forward.
        """
        feats = self.extract_feat(batch_inputs)
        results = self.panoptic_head.forward(feats, batch_data_samples)
        return results
