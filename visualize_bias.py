import numpy as np
from torchvision import datasets
import os
from models.resnet import *
from models.vgg import *
import cv2
from saliency.fullgrad import *
from bias_experiment import imshow
from pixel_perturbation import compute_saliency_per_grad
from pixel_perturbation import unnormalize
import argparse
from misc_functions import transform
import matplotlib as mpl
mpl.use('Agg')
# get PATH of file
PATH = os.path.dirname(os.path.abspath(__file__)) + '/'

# console testing
# PATH = os.path.abspath(os.getcwd()) + "/"

data_dir = 'biased_dataset/'
dataset = PATH + data_dir

# get transformator
transform_standard = transform()
# load Imagefolder
image_datasets = {x: datasets.ImageFolder(os.path.join(dataset, x),
                                          transform_standard)
                  for x in ['train', 'val']}
# get dataloaders
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                                              shuffle=False)
               for x in ['train', 'val']}
# retrieve dataset stats
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
data_stats = [dataset_sizes, class_names]


def fill_config_dict():
    config_dict = {}
    config_dict["bias_result_images_path"] =  'results/bias_experiment/' + ARGS.model_type + "/" + 'images_pred/'
    config_dict["model_file_name"] = "model_" + ARGS.model_type + ".pt"
    config_dict["model_type"] = ARGS.model_type
    config_dict["device"] = ARGS.device
    config_dict["target_layer"] =  ARGS.target_layer
    config_dict["model_path"] = PATH + 'results/bias_experiment/model'

    return config_dict

def load_finetuned_model(PATH, model_type):
    """
    :param PATH: Path of model.pt file
    :param model_type: "resnet" or "vgg"
    :return: loaded model
    """

    model = create_empty_finetune_model(model_type)
    # loads both cpu and gpu trained model_state_dicts
    model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
    model.eval()

    return model

def create_empty_finetune_model(model_type):
    """
    Creates untrained model of the type of the pretrained one to fill with the trained params
    :param model_type: "vgg" (vgg16_bn) or "resnet" (resnet18)
    :return:
    """

    if model_type == "vgg":
        model_ft = vgg16_bn(pretrained=True)

        model_ft.classifier = nn.Sequential(
            nn.Linear(25088, 4096, bias=True),
            nn.ReLU(True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.5, inplace=False),
            # last layer changed to binary
            nn.Linear(512, 2))

    else:
        model_ft = resnet18(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 2)

    return model_ft

def prediction_with_image(images, targets, predictions, saliency, config_dict, counter, data_type):
    """
    Saves image with prediction and target in title and saliency map together in same folder
    :param images: image batches of dim [batch_size, 224, 224, 3]
    :param targets: actual target tensors of dim [batch_size]
    :param predictions: most confident class of model output with tensors of dim [batch_size]
    :param saliency: saliency maps of dim [batch_size, 1, 224, 224]
    :param data_stats: [dataset_sizes, class_names]
    :param counter: counter of batches to create right file names
    :return: None
    """
    bias_result_images_path = config_dict["bias_result_images_path"]
    dataset_sizes, class_names = data_stats
    # plot saliency map correctly
    for i in range(len(images)):
        im = unnormalize(images[i, :, :, :].cpu())
        im = im.view(1, 3, 224, 224)[-1, :, :, :]
        reg = saliency[i, :, :, :]

        save_saliency_image(im, reg,
                            bias_result_images_path + data_type + "_" + str(counter) + "_" + str(i) + "_saliency" + ".jpg")

        title = "Target: " + class_names[targets[i].item()] + "/ Prediction: " + class_names[predictions[i].item()]
        # Make a grid from batch
        imshow(images[i].permute(1, 2, 0).numpy(), title=title,
               file_name=bias_result_images_path + data_type + "_" + str(counter) + "_" + str(i) + "_prediction" + ".jpg")

def save_saliency_image(image, saliency_map, filename):
    """
    Save saliency map on image.

    Args:
        image: Tensor of size (3,H,W)
        saliency_map: Tensor of size (1,H,W)
        filename: string with complete path and file extension

    """
    image = image.data.cpu().numpy()
    saliency_map = saliency_map.data.cpu().numpy()

    saliency_map = saliency_map - saliency_map.min()
    saliency_map = saliency_map / saliency_map.max()
    saliency_map = saliency_map.clip(0, 1)

    saliency_map = np.uint8(saliency_map * 255).transpose(1, 2, 0)
    saliency_map = cv2.resize(saliency_map, (224, 224))

    image = np.uint8(image * 255).transpose(1, 2, 0)
    image = cv2.resize(image, (224, 224))

    # Apply JET colormap
    color_heatmap = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)

    # Combine image with heatmap
    img_with_heatmap = np.float32(color_heatmap) + np.float32(image)
    img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap)

    cv2.imwrite(filename, np.uint8(255 * img_with_heatmap))

def compute_and_save_saliency_images(counter, data, grad, grad_type, initial_out, target, target_layer, config_dict, data_type):
    """
    Wrapper function to compute saliency maps and save images with their targets and predictions.
    :param data:
    :param grad:
    :param grad_type:
    :param target_layer:
    :param initial_out:
    :return:
    """
    cam = compute_saliency_per_grad(grad_type, grad, data, target_layer=target_layer, target_class=initial_out)

    # remove from gpu
    data = data.detach()
    target = target.detach()
    initial_out = initial_out.detach()
    cam = cam.detach()

    prediction_with_image(config_dict=config_dict, images=data, targets=target, predictions=initial_out,
                          saliency=cam, counter=counter, data_type=data_type)

def saliency_map_from_pretrained_model(dataloaders, config_dict, n_val=None, n_correct=None, data_type="val", grad_type="fullgrad"):
    """
    Creates the prediction/saliency images for n_correct correct classifications, all misclassifications and if data_type
    is "val" it also creates all of them.
    :param dataloaders:
    :param model_path: path to model.pt file
    :param model_type:
    :param device: device to compute gradients on. GPU much faster.
    :param grad_type: "fullgrad" is only supportet atm. For future experiments add gradcam.
    :param target_layer: For resnet = "layer4", for vgg = "features"
    :param file_name: name of .pt file
    :param data_type: "val" or "train"
    :return: None
    """
    # load finetuned model
    model_path = config_dict['model_path']
    model_file_name = config_dict['model_file_name']
    model_type = config_dict['model_type']
    device = config_dict['device']
    target_layer = config_dict['target_layer']

    model = load_finetuned_model(model_path + "/" + model_file_name, model_type)

    if grad_type == "fullgrad":
        # Initialize Gradient objects
        grad = FullGrad(model)
    # possibly added gradcam or Inputgrad to visualize how these methods show bias

    # to ensure not all correct classifications are plotted (~250 images)
    correct_counter = 1
    # For correct file names
    counter = 1

    for batch_idx, (data, target) in enumerate(dataloaders[data_type]):
        # console testing: for debugging purposes
        # data, target = next(iter(dataloaders[data_type]))

        data = data.to(device).requires_grad_()
        target = target.to(device)
        # evaluate without gradients
        with torch.no_grad():
            initial_output = model.forward(data)
            initial_out = initial_output.to(torch.device("cpu")).max(1)[1]

        if data_type == "train":
            # for train save all misclassifications
            if target != initial_out:
                compute_and_save_saliency_images(counter=counter, data=data, grad=grad, grad_type=grad_type,
                                                 initial_out=initial_out, target=target, target_layer=target_layer,
                                                 config_dict=config_dict, data_type=data_type)
                counter += 1
            else:
                # Save n_correct correctly classified images
                if correct_counter <= n_correct:
                    compute_and_save_saliency_images(counter=counter, data=data, grad=grad, grad_type=grad_type,
                                                     initial_out=initial_out, target=target, target_layer=target_layer,
                                                     config_dict=config_dict, data_type=data_type)
                    counter += 1
                    correct_counter += 1
        else:
            if counter <= n_val:
                # For val save all images
                compute_and_save_saliency_images(counter=counter, data=data, grad=grad, grad_type=grad_type,
                                                 initial_out=initial_out, target=target, target_layer=target_layer,
                                                 config_dict=config_dict, data_type=data_type)
                counter += 1

def main():

    config_dict = fill_config_dict()
    # save all validation images with targets, predictions and corresponding saliency map
    saliency_map_from_pretrained_model(dataloaders, config_dict, n_val=ARGS.n_val, data_type="val")
    # save all misclassifications on the train set and n_correct correctly classified images
    saliency_map_from_pretrained_model(dataloaders, config_dict, n_correct = ARGS.n_correct, data_type="train")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default="cuda:0", type=str,
                        help='cpu or gpu')
    parser.add_argument('--model_type', default="resnet", type=str,
                        help='which model type: resnet / vgg')
    parser.add_argument('--n_correct', default=10, type=int,
                        help='How many images/saliency maps saved for correctly classified train images')
    parser.add_argument('--n_val', default=10, type=int,
                        help='If only a few of the validation data is to be plotted to save time.')
    parser.add_argument('--target_layer', default="layer4", type=str,
                        help='For resnet: layer4, for vgg: features ')
    ARGS = parser.parse_args()

    main()

