auto_scale_lr = dict(base_batch_size=16, enable=False)
backbone_embed_multi = dict(decay_mult=0.0, lr_mult=0.1)
backbone_norm_multi = dict(decay_mult=0.0, lr_mult=0.1)
backend_args = None
# batch_augments = [
    # dict(
        # img_pad_value=0,
        # mask_pad_value=0,
        # pad_mask=True,
        # pad_seg=True,
        # seg_pad_value=255,
        # size=(
            # 1024,
            # 1024,
        # ),
        # type='BatchFixedSizePad'),
# ]
# custom_keys = dict({
    # 'absolute_pos_embed':
    # dict(decay_mult=0.0, lr_mult=0.1),
    # 'backbone':
    # dict(decay_mult=1.0, lr_mult=0.1),
    # 'backbone.norm':
    # dict(decay_mult=0.0, lr_mult=0.1),
    # 'backbone.patch_embed.norm':
    # dict(decay_mult=0.0, lr_mult=0.1),
    # 'backbone.stages.0.blocks.0.norm':
    # dict(decay_mult=0.0, lr_mult=0.1),
    # 'backbone.stages.0.blocks.1.norm':
    # dict(decay_mult=0.0, lr_mult=0.1),
    # 'backbone.stages.0.downsample.norm':
    # dict(decay_mult=0.0, lr_mult=0.1),
    # 'backbone.stages.1.blocks.0.norm':
    # dict(decay_mult=0.0, lr_mult=0.1),
    # 'backbone.stages.1.blocks.1.norm':
    # dict(decay_mult=0.0, lr_mult=0.1),
    # 'backbone.stages.1.downsample.norm':
    # dict(decay_mult=0.0, lr_mult=0.1),
    # 'backbone.stages.2.blocks.0.norm':
    # dict(decay_mult=0.0, lr_mult=0.1),
    # 'backbone.stages.2.blocks.1.norm':
    # dict(decay_mult=0.0, lr_mult=0.1),
    # 'backbone.stages.2.blocks.10.norm':
    # dict(decay_mult=0.0, lr_mult=0.1),
    # 'backbone.stages.2.blocks.11.norm':
    # dict(decay_mult=0.0, lr_mult=0.1),
    # 'backbone.stages.2.blocks.12.norm':
    # dict(decay_mult=0.0, lr_mult=0.1),
    # 'backbone.stages.2.blocks.13.norm':
    # dict(decay_mult=0.0, lr_mult=0.1),
    # 'backbone.stages.2.blocks.14.norm':
    # dict(decay_mult=0.0, lr_mult=0.1),
    # 'backbone.stages.2.blocks.15.norm':
    # dict(decay_mult=0.0, lr_mult=0.1),
    # 'backbone.stages.2.blocks.16.norm':
    # dict(decay_mult=0.0, lr_mult=0.1),
    # 'backbone.stages.2.blocks.17.norm':
    # dict(decay_mult=0.0, lr_mult=0.1),
    # 'backbone.stages.2.blocks.2.norm':
    # dict(decay_mult=0.0, lr_mult=0.1),
    # 'backbone.stages.2.blocks.3.norm':
    # dict(decay_mult=0.0, lr_mult=0.1),
    # 'backbone.stages.2.blocks.4.norm':
    # dict(decay_mult=0.0, lr_mult=0.1),
    # 'backbone.stages.2.blocks.5.norm':
    # dict(decay_mult=0.0, lr_mult=0.1),
    # 'backbone.stages.2.blocks.6.norm':
    # dict(decay_mult=0.0, lr_mult=0.1),
    # 'backbone.stages.2.blocks.7.norm':
    # dict(decay_mult=0.0, lr_mult=0.1),
    # 'backbone.stages.2.blocks.8.norm':
    # dict(decay_mult=0.0, lr_mult=0.1),
    # 'backbone.stages.2.blocks.9.norm':
    # dict(decay_mult=0.0, lr_mult=0.1),
    # 'backbone.stages.2.downsample.norm':
    # dict(decay_mult=0.0, lr_mult=0.1),
    # 'backbone.stages.3.blocks.0.norm':
    # dict(decay_mult=0.0, lr_mult=0.1),
    # 'backbone.stages.3.blocks.1.norm':
    # dict(decay_mult=0.0, lr_mult=0.1),
    # 'level_embed':
    # dict(decay_mult=0.0, lr_mult=1.0),
    # 'query_embed':
    # dict(decay_mult=0.0, lr_mult=1.0),
    # 'query_feat':
    # dict(decay_mult=0.0, lr_mult=1.0),
    # 'relative_position_bias_table':
    # dict(decay_mult=0.0, lr_mult=0.1)
# })
# data_preprocessor = dict(
    # batch_augments=[
        # dict(
            # img_pad_value=0,
            # mask_pad_value=0,
            # pad_mask=True,
            # pad_seg=True,
            # seg_pad_value=255,
            # size=(
                # 1024,
                # 1024,
            # ),
            # type='BatchFixedSizePad'),
    # ],
    # bgr_to_rgb=True,
    # mask_pad_value=0,
    # mean=[
        # 123.675,
        # 116.28,
        # 103.53,
    # ],
    # pad_mask=True,
    # pad_seg=True,
    # pad_size_divisor=32,
    # seg_pad_value=255,
    # std=[
        # 58.395,
        # 57.12,
        # 57.375,
    # ],
    # type='DetDataPreprocessor')
data_root = '../Dataset/Italian_panoptic/'
dataset_type = 'TunnelPanopticDataset'
# default_hooks = dict(
    # checkpoint=dict(
        # interval=1, ### 2
        # max_keep_ckpts=2,
        # rule='greater',
        # save_best='coco/segm_mAP',
        # save_last=True,
        # type='CheckpointHook'),
    # logger=dict(interval=10, type='LoggerHook'), ### 50
    # param_scheduler=dict(type='ParamSchedulerHook'),
    # sampler_seed=dict(type='DistSamplerSeedHook'),
    # timer=dict(type='IterTimerHook'),
    # visualization=dict(type='DetVisualizationHook'))
max_iters = 80000 # 80000
interval = 100 # 8000
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        save_last=True,
        max_keep_ckpts=3,
        interval=interval))
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=False)
default_scope = 'mmdet'
# depths = [
    # 2,
    # 2,
    # 18,
    # 2,
# ]

embed_multi = dict(decay_mult=0.0, lr_mult=1.0)
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
# image_size = (
    # 1024,
    # 1024,
# )
launcher = 'none'
load_from = None
log_level = 'INFO'
# log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        attn_drop_rate=0.0,
        convert_weights=True,
        depths=[
            2,
            2,
            18,
            2,
        ],
        drop_path_rate=0.3,
        drop_rate=0.0,
        embed_dims=192,
        frozen_stages=-1,
        init_cfg=dict(
            checkpoint=
            'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth',
            type='Pretrained'),
        mlp_ratio=4,
        num_heads=[
            6,
            12,
            24,
            48,
        ],
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        patch_norm=True,
        pretrain_img_size=384,
        qk_scale=None,
        qkv_bias=True,
        type='SwinTransformer',
        window_size=12,
        with_cp=False),
    data_preprocessor=dict(
        batch_augments=[
            dict(
                img_pad_value=0,
                mask_pad_value=0,
                pad_mask=True,
                pad_seg=True,
                seg_pad_value=255,
                size=(
                    1024,
                    1024,
                ),
                type='BatchFixedSizePad'),
        ],
        context_batch_augments=[
            dict(
                img_pad_value=0,
                mask_pad_value=0,
                pad_mask=True,
                pad_seg=True,
                seg_pad_value=255,
                size=(
                    1536,
                    1536,
                ),
                type='BatchFixedSizePad'),
        ],
        bgr_to_rgb=True,
        mask_pad_value=0,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_mask=True,
        pad_seg=True,
        pad_size_divisor=32,
        seg_pad_value=255,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    init_cfg=None,
    panoptic_fusion_head=dict(
        init_cfg=None,
        loss_panoptic=None,
        num_stuff_classes=4,
        num_things_classes=5,
        type='MaskFormerFusionHead'),
    panoptic_head=dict(
        enforce_decoder_input_project=False,
        feat_channels=256,
        in_channels=[
            192,
            384,
            768,
            1536,
        ],
        loss_cls=dict(
            class_weight=[
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.1,
            ],
            loss_weight=2.0,
            reduction='mean',
            type='CrossEntropyLoss',
            use_sigmoid=False),
        loss_dice=dict(
            activate=True,
            eps=1.0,
            loss_weight=5.0,
            naive_dice=True,
            reduction='mean',
            type='DiceLoss',
            use_sigmoid=True),
        loss_mask=dict(
            loss_weight=5.0,
            reduction='mean',
            type='CrossEntropyLoss',
            use_sigmoid=True),
        num_queries=100,
        num_stuff_classes=4,
        num_things_classes=5,
        num_transformer_feat_level=3,
        out_channels=256,
        pixel_decoder=dict(
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                layer_cfg=dict(
                    ffn_cfg=dict(
                        act_cfg=dict(inplace=True, type='ReLU'),
                        embed_dims=256,
                        feedforward_channels=1024,
                        ffn_drop=0.0,
                        num_fcs=2),
                    self_attn_cfg=dict(
                        batch_first=True,
                        dropout=0.0,
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4)),
                num_layers=6),
            norm_cfg=dict(num_groups=32, type='GN'),
            num_outs=3,
            positional_encoding=dict(normalize=True, num_feats=128),
            type='MSDeformAttnPixelDecoder'),
        positional_encoding=dict(normalize=True, num_feats=128),
        strides=[
            4,
            8,
            16,
            32,
        ],
        transformer_decoder=dict(
            init_cfg=None,
            layer_cfg=dict(
                cross_attn_cfg=dict(
                    batch_first=True, dropout=0.0, embed_dims=256,
                    num_heads=8),
                ffn_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    embed_dims=256,
                    feedforward_channels=2048,
                    ffn_drop=0.0,
                    num_fcs=2),
                self_attn_cfg=dict(
                    batch_first=True, dropout=0.0, embed_dims=256,
                    num_heads=8)),
            num_layers=9,
            return_intermediate=True),
        type='Mask2FormerHead'),
    test_cfg=dict(
        filter_low_score=True,
        instance_on=True,
        iou_thr=0.8,
        max_per_image=100,
        panoptic_on=True,
        semantic_on=False),
    train_cfg=dict(
        assigner=dict(
            match_costs=[
                dict(type='ClassificationCost', weight=2.0),
                dict(
                    type='CrossEntropyLossCost', use_sigmoid=True, weight=5.0),
                dict(eps=1.0, pred_act=True, type='DiceCost', weight=5.0),
            ],
            type='HungarianAssigner'),
        importance_sample_ratio=0.75,
        num_points=12544,
        oversample_ratio=3.0,
        sampler=dict(type='MaskPseudoSampler')),
    context_cfg=dict(
        context=True,
        effvit_width_list=[24, 48, 96, 192, 384],
        effvit_depth_list=[1, 3, 4, 4, 6],
        effvit_dim=32,
        effvit_pre='../mmdet-file/efficientvit_b2_r288.pt',
        effvit_proj=[384, 1536, 48],
        context_size=[1536,1536],
        crop_size=[1024,1024],
        stride_size=[768,768],
        local_thr=0.5,
    ),
    type='Mask2Former')
num_classes = 9
num_stuff_classes = 4
num_things_classes = 5
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.01, norm_type=2),
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ),
        eps=1e-08,
        lr=0.0001,
        type='AdamW',
        weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys=dict({
            'absolute_pos_embed':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone':
            dict(decay_mult=1.0, lr_mult=0.1),
            'backbone.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.patch_embed.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.0.blocks.0.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.0.blocks.1.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.0.downsample.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.1.blocks.0.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.1.blocks.1.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.1.downsample.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.0.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.1.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.10.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.11.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.12.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.13.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.14.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.15.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.16.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.17.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.2.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.3.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.4.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.5.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.6.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.7.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.8.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.9.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.downsample.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.3.blocks.0.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.3.blocks.1.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'level_embed':
            dict(decay_mult=0.0, lr_mult=1.0),
            'query_embed':
            dict(decay_mult=0.0, lr_mult=1.0),
            'query_feat':
            dict(decay_mult=0.0, lr_mult=1.0),
            'relative_position_bias_table':
            dict(decay_mult=0.0, lr_mult=0.1)
        }),
        norm_decay_mult=0.0),
    type='OptimWrapper')
# param_scheduler = [
    # dict(
        # begin=0, by_epoch=False, end=1000, start_factor=0.001, ### param?
        # type='LinearLR'),
    # dict(
        # T_max=100,
        # begin=1,
        # by_epoch=True,
        # end=100,
        # eta_min=1e-07,
        # type='CosineAnnealingLR'),
# ]
param_scheduler = dict(
    type='MultiStepLR',
    begin=0,
    end=max_iters,
    by_epoch=False,
    milestones=[655556, 710184],
    gamma=0.1)
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/panoptic_val.json',
        backend_args=None,
        data_prefix=dict(img='val/', seg='annotations/panoptic_val/'),
        data_root='../Dataset/Italian_panoptic/',
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            # dict(keep_ratio=True, scale=( ### keep raw resoluiton
                # 5000,
                # 5000,
            # ), type='Resize'),
            dict(backend_args=None, type='LoadPanopticAnnotations'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='TunnelPanopticDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = [
    dict(
        ann_file='../Dataset/Italian_panoptic/annotations/panoptic_val.json',
        backend_args=None,
        seg_prefix='../Dataset/Italian_panoptic/annotations/panoptic_val/',
        type='CocoPanopticMetric'),
    dict(
        ann_file=
        '../Dataset/Italian_panoptic/annotations/instances_val.json',
        backend_args=None,
        metric=[
            'bbox',
            'segm',
        ],
        type='CocoMetric'),
]
# test_pipeline = [
    # dict(backend_args=None, type='LoadImageFromFile'),
    # dict(keep_ratio=True, scale=(
        # 1333,
        # 800,
    # ), type='Resize'),
    # dict(backend_args=None, type='LoadPanopticAnnotations'),
    # dict(
        # meta_keys=(
            # 'img_id',
            # 'img_path',
            # 'ori_shape',
            # 'img_shape',
            # 'scale_factor',
        # ),
        # type='PackDetInputs'),
# ]
# train_cfg = dict(max_epochs=100, type='EpochBasedTrainLoop', val_interval=1) ### 2
dynamic_intervals = [(max_iters // interval * interval + 1, max_iters)]
train_cfg = dict(
    max_iters=max_iters,
    val_interval=interval,
    dynamic_intervals=dynamic_intervals)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=2, ### bs 1
    dataset=dict(
        ann_file='annotations/panoptic_train.json',
        backend_args=None,
        data_prefix=dict(img='train/', seg='annotations/panoptic_train/'),
        data_root='../Dataset/Italian_panoptic/',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(backend_args=None, to_float32=True, context_path='/train_context/', type='LoadContextImage'),
            dict(
                backend_args=None,
                type='LoadPanopticAnnotations',
                with_bbox=True,
                with_mask=True,
                with_seg=True),
            dict(prob=0.5, type='RandomFlip'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.5,
                    2.0,
                ),
                scale=( ### keep raw resoluiton
                    1024,
                    1024,
                ),
                scale_context=(1536,1536),
                type='RandomResizeContext'),
            dict(
                allow_negative_crop=True,
                crop_size=(
                    1024,
                    1024,
                ),
                context_size=(1536,1536),
                crop_type='absolute',
                recompute_bbox=True,
                type='RandomCrop'),
            dict(type='PackDetInputs'),
        ],
        type='TunnelPanopticDataset'),
    num_workers=8, ### 1
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
# train_pipeline = [
    # dict(backend_args=None, to_float32=True, type='LoadImageFromFile'),
    # dict(
        # backend_args=None,
        # type='LoadPanopticAnnotations',
        # with_bbox=True,
        # with_mask=True,
        # with_seg=True),
    # dict(prob=0.5, type='RandomFlip'),
    # dict(
        # keep_ratio=True,
        # ratio_range=(
            # 0.1,
            # 2.0,
        # ),
        # scale=(
            # 1024,
            # 1024,
        # ),
        # type='RandomResize'),
    # dict(
        # allow_negative_crop=True,
        # crop_size=(
            # 1024,
            # 1024,
        # ),
        # crop_type='absolute',
        # recompute_bbox=True,
        # type='RandomCrop'),
    # dict(type='PackDetInputs'),
# ]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/panoptic_val.json',
        backend_args=None,
        data_prefix=dict(img='val/', seg='annotations/panoptic_val/'),
        data_root='../Dataset/Italian_panoptic/',
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            # dict(keep_ratio=True, scale=( ### keep raw resoluiton
                # 5000,
                # 5000,
            # ), type='Resize'),
            dict(backend_args=None, type='LoadPanopticAnnotations'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='TunnelPanopticDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = [
    dict(
        ann_file='../Dataset/Italian_panoptic/annotations/panoptic_val.json',
        backend_args=None,
        seg_prefix='../Dataset/Italian_panoptic/annotations/panoptic_val/',
        type='CocoPanopticMetric'),
    dict(
        ann_file=
        '../Dataset/Italian_panoptic/annotations/instances_val.json',
        backend_args=None,
        metric=[
            'bbox',
            'segm',
        ],
        type='CocoMetric'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(
            init_kwargs=dict(
                name='mask2former_0219_cropped', project='Italian_tunnel_uhr'),
            type='WandbVisBackend'),
    ])
work_dir = 'results_uhr/mask2former_0219_cropped'
