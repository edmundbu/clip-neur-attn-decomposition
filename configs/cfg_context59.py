_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./configs/cls_context59.txt',
    pcs_path=None,  # PC0 for each neuron-head pair
    text_embed_path=None,  # used *during* segmentation
    text_for_topk=None,  # used *to select* pairs 
    k=20000
)

# dataset settings
dataset_type = 'PascalContext59Dataset'
data_root = 'seg_data/VOC2010'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 384), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClassContext'),
        ann_file='ImageSets/SegmentationContext/val.txt',
        pipeline=test_pipeline))
