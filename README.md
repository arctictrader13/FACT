# Visualizing Neural Network responses

##### Authors
P. Chandrikasingh - 11059842 - puja.chandrikasingh@student.uva.nl

R. Leushuis - 10988270 - radmir.leushuis@student.uva.nl

P. Lintl - 12152498 - philipp.lintl@student.uva.nl

A. Vicol - 12408913 - anca.vicol@student.uva.nl

Teacher Assistant: Leon Lang

## Summary

In this Repo we aim to reproduce an extend the ideas presented in the paper
[Full-Gradient Representation for Neural Network Visualization](https://arxiv.org/abs/1905.00780).

All details, experiments, results and discussions can be found in the notebooks `PP_bias.ipynb` and `remove_and_retrain.ipynb`.


# Full-Gradient Saliency Maps 

For FullGrad we use the code from the publication of Srinivas and Fleuret ["Full-Gradient Representation for Neural Network Visualization"](https://arxiv.org/abs/1905.00780).

Their repository implements two methods: the reference `FullGrad` algorithm, and a variant called `Simple FullGrad`, which omits computation of bias parameters for bias-gradients. The related `full-gradient decomposition` is implemented within `FullGrad`. Note that while `full-gradient decomposition` applies to any ReLU neural network, `FullGrad` saliency is <b>specific to CNNs</b>.

The codebase currently supports VGG, ResNet. Extending support for any other architecture of choice should be straightforward, and contributions are welcome! Among non-linearities, only ReLU-like functions are supported. For more information, please read the description of "implicit  biases" in the paper on how to include support for non-ReLU functions.

## Dependencies
See also the envoirnment file.
``` 
torch torchvision cv2 numpy 
```
