# PixFoundation: Are We Heading in the Right Direction with Pixel-level Vision Foundation Models?
[Project Webpage](https://msiam.github.io/PixFoundationSeries/), [Paper](https://arxiv.org/abs/2502.04192), [Datasets](https://huggingface.co/IVUlab)

Official implementation of my work on PixFoundation direction.

<div align="center">
<img src="https://github.com/MSiam/PixFoundation/blob/075f7b4da0f45fe6e391c3aaa63a3d0aeeddc58b/imgs/PixFoundationOverview.drawio.png" width="70%" height="70%"><br><br>
</div>


## Benchmarking Pixel-level MLLMs on PixMMVP & PixCV-Bench
PixMMVP and PixCV-Bench augment recent benchmarks with referring expression annotations and their corresponding segmentation. These are paired with the object of interest in the question within their original visual question answering task. The goal is to evaluate the pixel-level visual grounding and visual question answering capabilities of recent pixel-level MLLMs, e.g.,OMG-Llava, Llava-G, GLAMM and LISA. We also provide an interpretability mechanism for MLLMs that understands when
visual grounding occurs w.r.t the output tokens, through using MLLM as a Judge on the output segmentation.

<div align="center">
<img src="https://github.com/MSiam/PixFoundation/blob/693bfd82d5c1f6f95b6adf9eac8e3725637a6bb4/imgs/dataset.png" width="70%" height="70%"><br><br>
</div>

### Dataset Setup
[Data](https://github.com/MSiam/PixFoundation/blob/master/Data.md)

### Installation
[Installation](https://github.com/MSiam/PixFoundation/blob/master/INSTALL.md)

### PixMMVP Evaluation

* Run evaluation script after modifying it to the models needed, it includes two examples:
```
bash pixmmvp/scripts/run_all.sh
```
* Each of the pixel-level MLLMs inference code is based on their respective gradio demo codes not customized for a certain task.
* This a repository that includes example visualizations used in the automatic selection for LLaVA variants
```
git clone https://github.com/MSiam/AutoGPTImages
```

### Demo Interpretability Mechanism
* Run the following standalone script:
```
python interpretability_demo/demo.py --image_path image2.jpg --ref_expr "the hands holding the hat" --openai_api_key API_KEY
python interpretability_demo/demo.py --image_path image1.jpeg --ref_expr "the closed kitten's eyes" --openai_api_key API_KEY
```
* First example output:

|Noun Phrase | Image #1 | Image #2  | Image #3  |
|---|---|---|---|
|The hands|   |   |   |
|the hat|   |   |   |
|the scene|   |   |   |
|a pair|   |   |   |
|human hands|   |   |   |

* Second example output:

|Noun Phrase | Image #1 | Image #2  | Image #3  |
|---|---|---|---|
|the image| <img src="https://github.com/MSiam/PixFoundation/blob/3f38704b022adf353cfccf46d6967387e7381e0a/interpretability_demo/demo_out/demo_0.png" width="40%" height="40%">  |   |   |
|three kittens|   |   |   |
|them|   |   |   |
|its eyes|   |   |   |

### Changes to Evaluation Protocols
[Changes](https://github.com/MSiam/PixFoundation/blob/master/Changes.md)

## When does grounding emerge in MLLMs?
Our finding is that grounding can emerge coinciding with output text that describes the object of interest in terms of color, location or state and not necessarily the exact output text of this object. We persistently find the most frequent emergence occurs in the last 40-60% of the output text in MLLMs not trained with pixel-level grounding supervision (e.g., Llava 1.5 & Cambrian-1). We also show a histogram of the concept categories of the output text that coincides with the best segmentation emerging in such MLLMs.

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
