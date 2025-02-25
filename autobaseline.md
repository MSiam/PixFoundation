# Automatic Baseline (PixFoundation)

The automatic baseline runs in two stages, refer to "pixmmvp/scripts/infer_gptauto.sh":

* Generate the output images highlighting the objects using the predicted masks (--stage 1)
* Upload the images to a github repo and use the raw github link of the images
* Modify "base_github_url" accordingly in "pixmmvp/scripts/infer_pixfoundation_auto_gpt.py"
* Generate the automatically selected mask using GPT segmentation grading (--stage 2)

## Example Visualizations
This a repository that include example visualizations used to generate the automatic selection in PixMMVP:
```
git clone https://github.com/MSiam/AutoGPTImages
```
