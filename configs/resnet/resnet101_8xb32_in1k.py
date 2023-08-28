_base_ = [
    '../_base_/models/resnet101.py', '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        out_indices=(2,3))
    )