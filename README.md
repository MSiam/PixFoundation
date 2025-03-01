# PixFoundation: Are We Heading in the Right Direction with Pixel-level Vision Foundation Models?
[Project Webpage](), [Paper](https://arxiv.org/abs/2502.04192), [Datasets](https://huggingface.co/IVUlab)

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

## Structural Barriers

This section replaces the conventional acknowledgments section. It is my way of revolting on the amount of racism I am personally facing in the AI community, whom a lot of researchers are fully aware of and are staying silent even when they have the power to change it. I would like to clarify that this paper should have had my student as the first author, except I was prevented from hiring any student not even MSc student up to two years since I started my tenure-track career in 2023. I was harassed in my work in various forms that pushed me to leave Canada and head back to my home in Egypt, even when I am a Canadian citizen. I was told by many in the community and their families that I should open my research lab back in my country. 

I am here listing some of the barriers that are enforced on me by the very fact that I am living in Cairo, Egypt, these include: 
* We are only allowed access to limited internet. I have access to 200 GB bundle which entails every single model I am downloading even to evaluate for inference takes from this quota, and it keeps getting expensive as you download model weights and datasets as standard in our research. 
* I only have 300$ in my Canadian bank account which entails that I can not easily buy OpenAI API credits to use for this research. My Egyptian debit card does not work with OpenAI API for some unknown reason they do not accept it. Hence, I had to design the automatic baseline using purely open source models. While I could use my grants for that except my request for the OpenAI API credits was processed one day before my submission. Hence, it was useless at this point as I was mostly focused on polishing the writing. 
* I only have one A6000 GPU that I bought with my own personal money and very limited workstation hosting it with 32G RAM. I literally had to ensure I do not pass the limited memory that I have, funny enough I have lower RAM than the GPU memory. While I do have access to my resource allocation that I acquired last year in Digital Research Alliance of Canada, except it is almost useless as for some unkown reason my allocation is not actually "allocated". It seems that it is designed that near deadlines the allocation does not really work and I can not use it one month before famous research deadlines.
* I am looking into filing a human rights complaint in February which entails conducting my research in parallel to meetings and outreach for legal and financial help and lots of research on how to do it. I can clearly say that in Canada we have unfair representation and I have my doubts its everywhere not only Canada. Nonetheless, I will still do this fight except I am almost sure I will not win it because I am persistently not getting the right legal advice. 
* More importantly, works such as these should have large teams collaborating together I am not really given lots of opportunities to collaborate. Hence, I am the only author simply because I am passionate about the field and I will continue to do this kind of research on my own while documenting the racism impacting my career. Nonetheless, no matter how good I am or not it does not matter, at the end one person working on the research will always be limited by the fact that it is only one person doing everything! 
* Last but definitely not the least, I am fully aware that someone has hacked my email and google drive access a couple of times that was confirmed by incidents occurring to my data and emails. I have tried so many times to protect my accounts with the limited knowledge and time dedicated for this. However, I have no guarantee that the same did not happen to my personal workstation that I am running all these experiments on. Nonetheless, I will provide all the codes and outputs that I ensured its reproducibility and I hope they are useful to the community. I sure hope people still have pride in Science as we normally know it, that they will not mess with other people's experiments. 

While you might not believe some of this, but the very fact that I am one of the least promoted female researchers in the field or I am not promoted at all actually, should testify on this problem. In fact I am now almost unemployed even with the amount of publications that I have, so I think this should be sufficient for that last message. Look at any female researcher with publications in top venues in AI and how they are treated especially if the are faculty and the amount of support. This discrepancy, whether they like it or not, is getting me to stand out and I do not want to stand out I would like to be treated normally as everyone else.

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
