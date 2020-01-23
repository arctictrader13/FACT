import unittest
import torch
from misc_functions import remove_salient_pixels

class MiscFunctionsTestCase(unittest.TestCase):

    def test_remove_salient_pixels_simple(self):
        image_batch = torch.ones((1, 1, 2, 3))
        saliency_maps = torch.tensor([[[[0, 1, 0], [1, 0, 1]]]])
        output_batch = remove_salient_pixels(image_batch, saliency_maps, 3, \
                                             most_salient=True, replacement=[0.0])
        assert torch.all(torch.eq(output_batch[0][0], \
                         torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])))

    def test_remove_salient_pixels_multiple_images(self):
        image_batch = torch.zeros((3, 1, 2, 3))
        saliency_maps = torch.FloatTensor([[[[1, 0, 0], [0, 1, 0]]],
                                           [[[0, 1, 0], [0, 0, 1]]],
                                           [[[0, 0, 1], [1, 0, 0]]]])
        output_batch = remove_salient_pixels(image_batch, saliency_maps, 2, \
                                             most_salient=True, replacement=[1.0])
        assert torch.all(torch.eq(output_batch, saliency_maps))

    def test_remove_salient_pixels_multiple_channels(self):
        image_batch = torch.ones((1, 3, 2, 3))
        for i in range(3):
            image_batch[:, i, :, :] = torch.ones((2, 3)) * (i + 1)
        saliency_maps = torch.tensor([[[[0, 1, 0], [1, 0, 1]]]])
        output_batch = remove_salient_pixels(image_batch, saliency_maps, 3, \
                                             most_salient=True, replacement=[0.0])
        assert torch.all(torch.eq(output_batch,
                                  torch.tensor([[[[1., 0., 1.], [0., 1., 0.]],
                                                 [[2., 0., 2.], [0., 2., 0.]],
                                                 [[3., 0., 3.], [0., 3., 0.]]]])))
        

if __name__ == '__main__':
    unittest.main()