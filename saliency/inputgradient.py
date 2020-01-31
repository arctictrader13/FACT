import torch
import torch.nn as nn
import torch.nn.functional as F
from math import isclose


class Inputgrad():
    """
    Compute Inputgrad saliency map
    """

    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_inputgradients(self, image):
        """
        Compute Input-gradient for an image
        """

        image = image.requires_grad_()
        # forward pass to get outputs of model
        output = self.model.forward(image)
        # target class as most confident class of output
        target_class = output.data.max(1, keepdim=True)[1]

        # aggregate losses for autograd module
        agg = 0
        for i in range(image.size(0)):
            agg += output[i, target_class[i]]

        # set gradients to zero
        self.model.zero_grad()
        # Gradients w.r.t. input
        gradients = torch.autograd.grad(outputs=agg, inputs=image, only_inputs=True)

        # First element in the feature list is the image
        input_gradient = gradients[0]

        return input_gradient

    def rescale_gradients(self, input):
        # Absolute value
        input = abs(input)

        # Rescale operations to ensure gradients lie between 0 and 1
        input = input - input.min()
        input = input / (input.max())
        return input

    def saliency(self, image, target_class=None):

        # Inputgrad  saliency
        self.model.eval()
        input_grad = self.generate_inputgradients(image)

        # Input-gradient * image
        grd = input_grad * image
        saliency = self.rescale_gradients(input=grd).sum(1, keepdim=True)

        return saliency

