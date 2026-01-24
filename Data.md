# Dataset Setup for PixFoundation

## PixMMVP
* Download [MMVP](https://huggingface.co/datasets/MMVP/MMVP).
* Download our dataset annotations from [hugging face](https://huggingface.co/datasets/IVUlab/pixmmvp)
* Or use the referring expressions and segmentation annotations that are available under data/pixmmvp/
* Setup the dataset directory data/PixMMVP to the following

```
|--- PixMMVP
   |--- MMVP Images
   |--- Questions.csv
   |--- Objects.csv
   |--- Segmentations.json
   |--- visual_patterns.csv
   |--- meta_none.txt
```
* Visualizing the groundtruth annotations
```
python pixmmvp/dataset/test_loader.py --root data/PixMMVP/ --out_dir OUT_DIR
```
## PixCV-Bench
* Download [CV-Bench](https://huggingface.co/datasets/nyu-visionx/CV-Bench)
* Use their tool to recreate the images for the 2D section only, or download directly from ADE20K and COCO2017 datasets.
* Download our dataset annotations from [hugging face](https://huggingface.co/datasets/IVUlab/pixcvbench)
* Or use the referring expressions and segmentation annotations that are available under data/pixcvbench/
* Setup the dataset directory data/PixCVBench to the following

```
|--- PixCVBench
   |--- ADE20K
      |--- validation
         |--- images
         |--- FinalSegmentations.json
   |--- COCO
      |--- coco
         |--- images
      |--- FinalSegmentations.json
   |--- CV-Bench
      |--- test.parquet
      |--- Objects.csv
```

* Visualizing the groundtruth annotations
```
python pixcvbench/dataset/test_loader.py --ade_root data/PixCVBench/ADE20K/ --coco_root data/PixCVBench/COCO/ --out_dir OUT_DIR
```
