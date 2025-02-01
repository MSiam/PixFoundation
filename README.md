# PixFoundation: Are we Heading in the Right Direction with Pixel-level MLLMs?
[Project Webpage](), [Paper]()

Official implementation of our work on PixFoundation direction, Code coming soon...

## Benchmarking Pixel-level MLLMs on PixMMVP & PixCV-Bench
PixMMVP and PixCV-Bench augment recent benchmarks with referring expression annotations and their corresponding segmentation. These are paired with the object of interest in the question within their original visual question answering task. The goal is to evaluate the pixel-level visual grounding and visual question answering capabilities of recent pixel-level MLLMs, e.g.,OMG-Llava, Llava-G, GLAMM and LISA.

<div align="center">
<img src="https://github.com/MSiam/PixFoundation/" width="70%" height="70%"><br><br>
</div>

## When does grounding emerge in MLLMs?
Our finding is that grounding can emerge coinciding with output text that describes the object of interest in terms of color, location or state and not necessarily the exact output text of this object. We persistently find it mostly emerges int he last 40% of the output text in MLLMs not trained with pixel-level grounding supervision (e.g., Llava 1.5 & Cambrian-1). We also show a histogram of the concept categories of the output text that coincides with the best segmentation emerging
in such MLLMs.

<div align="center">
<img src="https://github.com/MSiam/PixFoundation/" width="70%" height="70%"><br><br>
</div>

<div align="center">
<img src="https://github.com/MSiam/PixFoundation/" width="70%" height="70%"><br><br>
</div>

<div align="center">
<img src="https://github.com/MSiam/PixFoundation/" width="70%" height="70%"><br><br>
</div>
<div align="center">
<img src="https://github.com/MSiam/PixFoundation/" width="70%" height="70%"><br><br>
</div>
<div align="center">
<img src="https://github.com/MSiam/PixFoundation/" width="70%" height="70%"><br><br>
</div>

<div align="center">
<img src="https://github.com/MSiam/PixFoundation/" width="70%" height="70%"><br><br>
</div>

<div align="center">
<img src="https://github.com/MSiam/PixFoundation/" width="70%" height="70%"><br><br>
</div>

