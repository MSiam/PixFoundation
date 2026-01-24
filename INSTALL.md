# Installation Setup for PixFoundation Benchmarking

* Clone the repository recursively to include the submodules
```
git clone --recursive https://github.com/MSiam/PixFoundation
```
* Install the base requirements for evaluations
```
pip install -r requirements.txt
```
* Setup detectron2 for some utilities used (**Optional**)
```
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
```
* Follow installation setup for each model you are evaluating, refer to their README, and build its conda environment


