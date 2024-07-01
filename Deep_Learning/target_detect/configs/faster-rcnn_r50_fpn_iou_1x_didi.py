_base_ = [
    '../_base_/models/faster-rcnn_r50_anchor84_fpn.py',
    '../_base_/datasets/didi_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    roi_head=dict(
        bbox_head=dict(
            reg_decoded_bbox=True,
            loss_bbox=dict(type='IoULoss', loss_weight=10.0))))
