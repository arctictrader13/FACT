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

All details, experiments, results and discussions can be found in the notebooks `PP_bias.ipynb`, `remove_and_retrain.ipynb` the ['final report'](https://github.com/rmleushuis/FACT/blob/master/11059842_10988270_12152498_12408913-FACT-AI-report.pdf) and our ['presentation'](https://github.com/rmleushuis/FACT/blob/master/presentation/Presentation-converted.pdf)


# Full-Gradient Saliency Maps 

For FullGrad we use the code from the publication of Srinivas and Fleuret ["Full-Gradient Representation for Neural Network Visualization"](https://arxiv.org/abs/1905.00780).

Their repository implements two methods: the reference `FullGrad` algorithm, and a variant called `Simple FullGrad`, which omits computation of bias parameters for bias-gradients. The related `full-gradient decomposition` is implemented within `FullGrad`. Note that while `full-gradient decomposition` applies to any ReLU neural network, `FullGrad` saliency is <b>specific to CNNs</b>.

The codebase currently supports VGG, ResNet. Extending support for any other architecture of choice should be straightforward, and contributions are welcome! Among non-linearities, only ReLU-like functions are supported. For more information, please read the description of "implicit  biases" in the paper on how to include support for non-ReLU functions.

# Bias Experiments 
In order to demonstrate that Saliency Maps can help to identify biases within trained models, we scraped images of doctors and nurses and trained them disproportionally according to gender. Our hypothesis was that training a model on almost exclusively male doctors and female nurses would yield classification based on gender rather than characteristics such as a stethoscop. Saliency Maps based on the biased model revealed that in fact not neccessarily characteristics of the profession but the gender were crucial for classification. For doctors the presence of a tie turned out to be highly important. Some examples demonstrate the observations: 

Background classification:\
![](/results/bias_experiment/resnet/images_pred/val_73_0_saliency_.jpg )

Classification based on Tie:\
![](/results/bias_experiment/resnet/images_pred/val_97_0_saliency_.jpg "Title")
![](/results/bias_experiment/resnet/images_pred/val_96_0_saliency_.jpg "Title")

However saliency maps also help qualitative assessment of the model. As for instance some classifications show that not only the tie but also other features seem to be important. Thus, the stethoscop in fact seems to be detected:

For the nurses on the other hand mainly the color of their clothing, the presence of a clipboard and the clothing itself seems relevant as the entire region tends to be highly activated. 

![](/results/bias_experiment/resnet/images_pred/val_69_0_saliency_.jpg "Title")
![](/results/bias_experiment/resnet/images_pred/val_69_0_prediction.jpg "Title")

![](/results/bias_experiment/resnet/images_pred/val_82_0_saliency_.jpg "Title")
![](/results/bias_experiment/resnet/images_pred/val_82_0_prediction.jpg "Title")

![](/results/bias_experiment/resnet/images_pred/val_70_0_saliency_.jpg "Title")
![](/results/bias_experiment/resnet/images_pred/val_70_0_prediction.jpg "Title")
\

Misclassfications can also be explained: \
![](/results/bias_experiment/resnet/images_pred/train_7_0_saliency_.jpg "Title")
![](/results/bias_experiment/resnet/images_pred/train_7_0_prediction.jpg "Title")

![](/results/bias_experiment/resnet/images_pred/train_43_0_saliency_.jpg "Title")
![](/results/bias_experiment/resnet/images_pred/train_43_0_prediction.jpg "Title")



Complete results are found in [Folder](https://github.com/rmleushuis/FACT/tree/master/results/bias_experiment)

## Dependencies
See also the environment file.
``` 
torch torchvision cv2 numpy 
```


