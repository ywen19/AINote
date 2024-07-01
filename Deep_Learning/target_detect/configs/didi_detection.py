dataset_type = 'CocoDataset'
data_root = 'C:\\Users\\ext.yiwen\\Desktop\\AI\\DL\\target_detect\\data\\dataset_release\\'

classes = ('car','van','bus','truck','person','bicycle','motorcycle','open-tricycle','closed-tricycle','forklift','large-block','small-block')

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# config backup from ver 2.x

# classes = ('car','van','bus','truck','person','bicycle','motorcycle','open-tricycle','closed-tricycle','forklift','large-block','small-block')

# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size_divisor=32),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
# ]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(1333, 800),
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='Pad', size_divisor=32),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ])
# ]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,  # Avoid recreating subprocesses after each iteration
    sampler=dict(type='DefaultSampler', shuffle=True),  # Default sampler, supports both distributed and non-distributed training
    batch_sampler=dict(type='AspectRatioBatchSampler'),  # Default batch_sampler, used to ensure that images in the batch have similar aspect ratios, so as to better utilize graphics memory
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        ann_file=data_root + 'train.json',
        data_prefix=dict(img=data_root + 'train/'),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        ann_file=data_root + 'val.json',
        data_prefix=dict(img=data_root + 'val/'),
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        ann_file=data_root + 'test.json',
        data_prefix=dict(img=data_root + 'test/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val.json',
    metric='bbox',
    format_only=True,
    outfile_prefix='./test_results',
    backend_args=backend_args)


# config backup from ver 2.x

# data = dict(
#     samples_per_gpu=2,
#     workers_per_gpu=2,
#     train=dict(
#         type=dataset_type,
#         classes=classes,
#         ann_file=data_root + 'train.json',
#         img_prefix=data_root + 'train/',
#         pipeline=train_pipeline),
#     val=dict(
#         type=dataset_type,
#         classes=classes,
#         ann_file=data_root + 'val.json',
#         img_prefix=data_root + 'val/',
#         pipeline=test_pipeline),
#     test=dict(
#         type=dataset_type,
#         classes=classes,
#         ann_file=data_root + 'test.json',
#         img_prefix=data_root + 'test/',
#         pipeline=test_pipeline))
# evaluation = dict(interval=1, metric='bbox')
