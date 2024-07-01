_base_ = [
    '../_base_/models/faster-rcnn_r50_anchor84_fpn.py',
    '../_base_/datasets/didi_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
