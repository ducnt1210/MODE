_base_ = '/home/ducnt155/Project/MultimodalOD/CPA_Enhancer/configs/_base_/default_runtime.py' #TODO: replace with your default_runtime.py path

# TODO: also replace path for dataset as well as coarse mask and text embedding in safetensors file.

custom_imports = dict(
    imports=['projects.CO-DETR.codetr'], allow_failed_imports=False)

data_root = 'data/AdverseWeathers/'
dataset_type = 'CustomCocoDataset'
test_data_root = 'data/DAWN/'
test_data_root_2 = '/home/ducnt155/Project/MultimodalOD/dataset/RTTS'
metainfo = {
    'classes': ('car', 'bus', 'motorcycle', 'bicycle', 'person', 'truck'),
}
image_size = (640, 640)

max_epochs = 36

class_name =  ('car', 'bus', 'motorcycle', 'bicycle', 'person', 'truck')
num_classes = len(class_name)
num_dec_layer = 6
loss_lambda = 2.0

train_batch_size_per_gpu = 2
train_num_workers = 8
# batch_augments = [
#     dict(type='BatchFixedSizePad', size=image_size, pad_mask=False)
# ]

resume = True

model = dict(
    type='MyNewCoDETRV3',
    # If using the lsj augmentation,
    # it is recommended to set it to True.
    use_lsj=False,
    # detr: 52.1
    # one-stage: 49.4
    # two-stage: 47.9
    eval_module='detr',  # in ['detr', 'one-stage', 'two-stage']
    data_preprocessor=dict(
        type='DistDetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=True,
        # batch_augments=batch_augments
    ),
    restormer=dict(
        type='PromptLangRestormerV4',  # use CPA-Enhancer
        c_in=3,
        c_out=3,
        dim=24,
    ),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='ChannelMapper',
        in_channels=[256, 512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=5),
    query_head=dict(
        type='CoDINOHead',
        num_query=900,
        num_classes=num_classes,
        in_channels=2048,
        as_two_stage=True,
        dn_cfg=dict(
            label_noise_scale=0.5,
            box_noise_scale=1.0,
            group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100)),
        transformer=dict(
            type='CoDinoTransformer',
            with_coord_feat=False,
            num_co_heads=2,  # ATSS Aux Head + Faster RCNN Aux Head
            num_feature_levels=5,
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                # number of layers that use checkpoint.
                # The maximum value for the setting is num_layers.
                # FairScale must be installed for it to work.
                with_cp=4,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=256,
                        num_levels=5,
                        dropout=0.0),
                    feedforward_channels=2048,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DinoTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.0),
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=256,
                            num_levels=5,
                            dropout=0.0),
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            temperature=20,
            normalize=True),
        loss_cls=dict(  # Different from the DINO
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0 * num_dec_layer * loss_lambda),
        loss_bbox=dict(
            type='L1Loss', loss_weight=1.0 * num_dec_layer * loss_lambda)),
    roi_head=[
        dict(
            type='CoStandardRoIHead',
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32, 64],
                finest_scale=56),
            bbox_head=dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0 * num_dec_layer * loss_lambda),
                loss_bbox=dict(
                    type='GIoULoss',
                    loss_weight=10.0 * num_dec_layer * loss_lambda)))
    ],
    bbox_head=[
        dict(
            type='CoATSSHead',
            num_classes=num_classes,
            in_channels=256,
            stacked_convs=1,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                ratios=[1.0],
                octave_base_scale=8,
                scales_per_octave=1,
                strides=[4, 8, 16, 32, 64, 128]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[.0, .0, .0, .0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0 * num_dec_layer * loss_lambda),
            loss_bbox=dict(
                type='GIoULoss',
                loss_weight=2.0 * num_dec_layer * loss_lambda),
            loss_centerness=dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                loss_weight=1.0 * num_dec_layer * loss_lambda)),
    ],
    # model training and testing settings
    train_cfg=[
        dict(
            assigner=dict(
                type='HungarianAssigner',
                match_costs=[
                    dict(type='FocalLossCost', weight=2.0),
                    dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                    dict(type='IoUCost', iou_mode='giou', weight=2.0)
                ])),
        dict(
            rpn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    match_low_quality=True,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                allowed_border=-1,
                pos_weight=-1,
                debug=False),
            rpn_proposal=dict(
                nms_pre=4000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)),
        dict(
            assigner=dict(type='ATSSAssigner', topk=9),
            allowed_border=-1,
            pos_weight=-1,
            debug=False)
    ],
    test_cfg=[
        # Deferent from the DINO, we use the NMS.
        dict(
            max_per_img=300,
            # NMS can improve the mAP by 0.2.
            nms=dict(type='soft_nms', iou_threshold=0.8)),
        dict(
            rpn=dict(
                nms_pre=1000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                score_thr=0.0,
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=100)),
        dict(
            # atss bbox head:
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.0,
            nms=dict(type='nms', iou_threshold=0.6),
            max_per_img=100),
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ])


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='CustomResize', scale=image_size, keep_ratio=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='CustomPad', size=image_size, pad_val=dict(img=(114, 114, 114))),
    dict(type='CustomPackDetInputs')
]

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    sampler=dict(
        type='DefaultSampler',
        shuffle=True
    ),
    dataset=dict(
            type=dataset_type,
            data_root=data_root,
            metainfo=metainfo,
            ann_file='train/anno_train_coco.json',
            mask_dir='/AdverseWeathers/train/maskv3',
            tensor_file= '/AdverseWeathers/train/annotations/v1/train.safetensors',
            data_prefix=dict(img='train/img/'),
            pipeline=train_pipeline))

# follow ViTDet
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CustomResize', scale=image_size, keep_ratio=True),  # diff
    dict(type='CustomPad', size=image_size, pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(
        type='CustomPackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

val_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    drop_last=False,
    sampler=dict(
        type='DefaultSampler',
        shuffle=False
    ),
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='val/anno_val_coco.json',
        mask_dir='/AdverseWeathers/val/maskv3',
        tensor_file= '/AdverseWeathers/val/annotations/v1/val.safetensors',
        data_prefix=dict(img='val/img/'),
        pipeline=test_pipeline))

test_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    drop_last=False,
    sampler=dict(
        type='DefaultSampler',
        shuffle=False
    ),
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='val/anno_val_coco.json',
        mask_dir='/AdverseWeathers/val/maskv3',
        tensor_file= '/AdverseWeathers/val/annotations/v1/val.safetensors',
        data_prefix=dict(img='val/img/'),
        pipeline=test_pipeline))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val/anno_val_coco.json',
    metric=['bbox'],
    format_only=False,
    classwise=True,
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val/anno_val_coco.json',
    metric=['bbox'],
    format_only=False,
    classwise=True,
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Checkpoint saving
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=200),
    # param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='coco/bbox_mAP_50', rule='greater', max_keep_ckpts=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook')
)

# custom_hooks = [
#     dict(type='DetWandbVisualizationHook', )
# ]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}))

# Wandb initialization
# wandb_init_kwargs = dict(
#     project='MultimodalOD',
#     config=dict(  # optional, for tracking hyperparameters
#         batch_size=train_batch_size_per_gpu,
#         epochs=max_epochs,
#         model='Co-DETR',
#     )
# )

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=1)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[30],
        gamma=0.1)
]


visualizer = dict(vis_backends = [
    # dict(type='LocalVisBackend'), 
                                #   dict(type='WandbVisBackend'),
                                  ]) # noqa