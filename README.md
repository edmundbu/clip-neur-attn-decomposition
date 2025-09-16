## Dependencies

```setup
pip install einops tqdm ftfy regex scipy scikit-learn matplotlib seaborn
```

To run semantic segmentation, please install PyTorch 1.10.x (otherwise most PyTorch versions work) and the following packages:

```setup
pip install openmim
mim install mmcv==2.0.1 mmengine==0.8.4 mmsegmentation==1.1.1
pip install yapf==0.40.1 numpy<2
```

## Datasets

ImageNet validation and test splits can be downloaded [here](https://image-net.org/). After downloading, run ```python utils/test_dataset.py``` to reduce the test split to 5000 images.  
```ImageNet
root/
├── data/
│   └── val/
└── test_data/
    └── test/
```

PASCAL Context can be downloaded [here](https://cs.stanford.edu/~roozbeh/pascal-context/).
```Context
root/
└── seg_data/
    └── VOC2010/
        ├── ImageSets/
        ├── JPEGImages/
        ├── SegmentationClassContext/
        ├── labels.txt
        └── trainval_merged.json
```

Stanford Cars train split can be downloaded [here](https://www.kaggle.com/datasets/eduardo4jesus/stanford-cars-dataset), and refined colors labels [here](https://github.com/morrisfl/stanford_cars_refined).
```Cars
root/
├── cars/
│   └── stanford_cars/
│       ├── cars_train/
│       └── devkit/
└── cars_refined/
    └──refined_train.csv
```

## Setup

To obtain ImageNet, top 30k English words, and PASCAL Context zero-shot text embeddings:
```embeddings
python text_set_projection.py --embeddings_save_path 'caches/IN_embed.pt'

python text_set_projection.py --classnames '30k' --embeddings_save_path 'caches/30k_embed.pt'

python text_set_projection.py --classnames 'PC' --embeddings_save_path 'caches/context_embed.pt'
```

To collect neuron-head representations over 1000 images:
```collect
python collection.py --save_path 'caches/reprs.dat'  # careful: ~450GB for CLIP-RN50x16
```

To obtain $\hat{r}^{n,h}$ for all pairs:
```svd
python svd_over_dataset.py --shape (1000, 48, 3072, 768) --collect_save_path 'caches/reprs.dat' --mean_save_path 'caches/means.npy' --out_save_path 'caches/pc0.npy' --norm_save_path 'caches/norms.npy'
```

To approximate each $\hat{r}^{n, h}$ with 64 text embeddings:
```
python sparse_decomposition.py --num_components 64 --text_desc_embeddings_path 'caches/30k_embed.pt' --pcs_path 'caches/pc0.npy' --descriptions_save_path 'caches/text_descs.json' --decomposition_save_path 'caches/decomposition.npz'
```

## Analysis

To compute reconstruction from $\hat{r}^{n, h}$:
```reconstruct
python reconstruct.py --imagenet_zeroshot_path 'caches/IN_embed.pt' --pcs_paths 'caches/pc0.npy' --means_path 'caches/means.npy'
```

To compute classification performance with 90\% of neuron-head pairs mean-ablated:
```ablation
python get_masks.py --neuron_head_norms_path 'caches/norms.npy' --percentile 90 --mask_save_path 'caches/mask.npy'

python ablation.py --imagenet_zeroshot_path 'caches/IN_embed.pt' --means_path 'caches/means.npy' --replace_index_path 'caches/mask.npy'
```

To compute reconstruction from sparse text embeddings:
```sparse
python reconstruct.py --imagenet_zeroshot_path 'caches/IN_embed.pt' --means_path 'caches/means.npy' --text_descs_embeddings_path 'caches/30k_embed.pt' --decomposition_save_path 'caches/decomposition.npz'
```

To perform image retrieval for a given neuron (ex. 624) and all of its neuron-head combinations over ImageNet validation:
```retrieval
python subconcept.py --neur_idxs 624 --save_dir 'caches/retrieval'
```

## Applications

To perform semantic segmentation on PASCAL Context using neuron-head pairs, set paths in ```configs/cfg_context59.py``` and run:
```segmentation
python eval.py 
```

To track concepts over Stanford Cars using neuron-head pairs:
```dist
python distribution_shift.py --pcs_path 'caches/pc0.npy'
```

## BibTeX (TODO)