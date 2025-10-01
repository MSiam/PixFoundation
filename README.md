# PixFoundation: Are We Heading in the Right Direction with Pixel-level Vision Foundation Models?
[Project Webpage](https://msiam.github.io/PixFoundationSeries/), [Paper](https://arxiv.org/abs/2502.04192), [Datasets](https://huggingface.co/IVUlab)

Official implementation of my work on PixFoundation direction.

<div align="center">
<img src="https://github.com/MSiam/PixFoundation/blob/1f9a5711ed7a3f87338bcb7d8b8381bd79d38431/imgs/ICML25PixFoundation.drawio.png" width="70%" height="70%"><br><br>
</div>


## Benchmarking Pixel-level MLLMs on PixMMVP & PixCV-Bench
PixMMVP and PixCV-Bench augment recent benchmarks with referring expression annotations and their corresponding segmentation. These are paired with the object of interest in the question within their original visual question answering task. The goal is to evaluate the pixel-level visual grounding and visual question answering capabilities of recent pixel-level MLLMs, e.g.,OMG-Llava, Llava-G, GLAMM and LISA.

<div align="center">
<img src="https://github.com/MSiam/PixFoundation/blob/693bfd82d5c1f6f95b6adf9eac8e3725637a6bb4/imgs/dataset.png" width="70%" height="70%"><br><br>
</div>

### Dataset Setup
[Data](https://github.com/MSiam/PixFoundation/blob/master/Data.md)

### PixMMVP Evaluation

#### Installation
* Clone the repository recursively to include the submodules
```
git clone --recursive https://github.com/MSiam/PixFoundation
```
* Follow installation setup for each model within conda environment/s
* Setup detectron2 for the IoU evaluation

#### Evaluation
* Run evaluation script after modifying it to the models needed, it includes two examples:
```
bash pixmmvp/scripts/run_all.sh
```
* Each of the pixel-level MLLMs inference code is based on their respective gradio demo codes not customized for a certain task.
* Instructions for the automatic baseline [AutoBaseline](https://github.com/MSiam/PixFoundation/blob/master/autobaseline.md)
* PixCV-Bench evaluation code coming soon

## When does grounding emerge in MLLMs?
Our finding is that grounding can emerge coinciding with output text that describes the object of interest in terms of color, location or state and not necessarily the exact output text of this object. We persistently find the most frequent emergence occurs in the last 40-60% of the output text in MLLMs not trained with pixel-level grounding supervision (e.g., Llava 1.5 & Cambrian-1). We also show a histogram of the concept categories of the output text that coincides with the best segmentation emerging in such MLLMs.
* Minor fix where the examples with ground-truth mask allbackground are discarded in the "When" analysis.

<div align="center">
<img src="https://github.com/MSiam/PixFoundation/blob/1f9a5711ed7a3f87338bcb7d8b8381bd79d38431/imgs/histograms_new.png" width="70%" height="70%"><br><br>
</div>

<div align="center">
<img src="https://github.com/MSiam/PixFoundation/blob/55686df651a7ceaf43f649eaa4e8a47c14aae91b/imgs/emerging_text.png" width="70%" height="70%">
</div>

<div align="center">
<img src="https://github.com/MSiam/PixFoundation/blob/3703c3c23144294f7be4e6e5d013b935a14f14f7/imgs/example1.png" width="40%" height="70%">
<img src="https://github.com/MSiam/PixFoundation/blob/3703c3c23144294f7be4e6e5d013b935a14f14f7/imgs/example2.png" width="40%" height="70%"><br><br>
</div>

<div align="center">
<img src="https://github.com/MSiam/PixFoundation/blob/3703c3c23144294f7be4e6e5d013b935a14f14f7/imgs/example3.png" width="40%" height="70%">
<img src="https://github.com/MSiam/PixFoundation/blob/3703c3c23144294f7be4e6e5d013b935a14f14f7/imgs/example4.png" width="40%" height="70%"><br><br>
</div>

# Acknowledgements
These repositories were used as part of our work:
* [OMG-Llava](https://github.com/lxtGH/OMG-Seg)
* [GLAMM](https://github.com/mbzuai-oryx/groundingLMM)
* [LISA](https://github.com/dvlab-research/LISA)
* [Llava-G](https://github.com/UX-Decoder/LLaVA-Grounding)
* [Llava](https://github.com/haotian-liu/LLaVA)
* [Emerging Grounding](https://groundlmm.github.io/)
* [Cambrian](https://github.com/cambrian-mllm/cambrian)
* [Eyes Wide Shut](https://github.com/tsb0601/MMVP)

# References
Please cite my paper if you find it useful in your research

```
@article{siam2025pixfoundation,
  title={PixFoundation: Are We Heading in the Right Direction with Pixel-level Vision Foundation Models?},
  author={Siam, Mennatullah},
  journal={arXiv preprint arXiv:2502.04192},
  year={2025}
}
```
