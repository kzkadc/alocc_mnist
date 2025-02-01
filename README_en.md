# Implementation of ALOCC, an anomary detection method with deep learning
Implemented with Chainer and PyTorch.

Sabokrou, et al. "Adversarially Learned One-Class Classifier for Novelty Detection", The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 3379-3388

https://arxiv.org/abs/1802.09088


## Requirements (PyTorch)
Requires PyTorch, PyTorch-Ignite, OpenCV, Matplotlib, and scikit-learn.

Installation:  
PyTorch: see [the official document](https://pytorch.org/get-started/locally/).

```bash
sudo pip install pytorch-ignite opencv-python matplotlib sklearn
```

## Run
Hyperparameters can be set by editting `setting.json`.

```bash
$ python train.py setting_file output_directory [-g GPUID]
```

Example:
```bash
$ python train.py setting.json result
```