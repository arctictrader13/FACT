""" Compute saliency maps of images from dataset folder
    and dump them in a results folder """

from pixel_perturbation import *
from misc_functions import *
from saliency import fullgrad
from models.resnet import *
from models.vgg import *
import argparse

# PATH variables
# PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
PATH = os.path.abspath(os.getcwd()) + "/"
dataset = PATH + 'dataset/'
result_path = PATH + 'results/imagenet/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_PATH = PATH + 'dataset/'
dataset = data_PATH
unnormalize = unnormalize()
save_path = PATH + 'results/'

def get_filename(result_path, grad_type, index, name_addition=None):
    filename = result_path + "/" + grad_type + "_" + ARGS.model_name + "_" + str(index) + name_addition + ".png"
    return filename

def compute_saliency_and_save(sample_loader):
    for grad_type in ARGS.grads:
        inputgrad_bool = False
        model, grad = initialize_grad_and_model(grad_type, ARGS.model_name, device)
        if grad_type == "inputgrad":
            inputgrad_bool = True

        grad_counter = 0
        print("grad_type:{}".format(grad_type))
        grad_counter += 1
        counter = 1

        for batch_idx, (data, target) in enumerate(sample_loader):
            if counter >= ARGS.n_images_save:
                break
            data, target = data.to(device).requires_grad_(), target.to(device)
            # data, _ = next(iter(sample_loader))

            # Compute saliency maps for the input data
            if grad_type == "gradcam":
                probs, ids = grad.forward(data)
                # Grad-CAM
                grad.backward(ids=ids[:, [0]])
                saliency = grad.generate(target_layer=ARGS.target_layer)

            else:
                saliency = grad.saliency(data)

            for i in range(len(data)):
                im = unnormalize(data[i, :, :, :].cpu())
                im = im.view(1, 3, 224, 224)[-1, :, :, :]
                reg = saliency[i, :, :, :]
                filename = get_filename(result_path, grad_type, counter, name_addition=ARGS.name_addition)
                counter += 1
                print(filename)

                # print("filename:{}".format(filename))
                save_saliency_map_inputgrad(im, reg, filename, inputgrad=inputgrad_bool)

def save_saliency_map_inputgrad(image, saliency_map, filename, inputgrad=False, save_image=False):
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
    if inputgrad is True:
        img_with_heatmap = np.float32(color_heatmap)
    else:
        img_with_heatmap = np.float32(color_heatmap) + np.float32(image)

    img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap)

    cv2.imwrite(filename, np.uint8(255 * img_with_heatmap))

def get_sample_loader():
    dataset = data_PATH
    sample_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(dataset, transform=transform_standard),
        batch_size=ARGS.batch_size, shuffle=False)

    return sample_loader

def main():
    sample_loader = get_sample_loader()
    compute_saliency_and_save(sample_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=10, type=int,
                        help='How large are processed image batches')
    parser.add_argument('--grads', default=["inputgrad", "fullgrad", "gradcam"], type=str,nargs='+',
                        help='fullgrad, gradcam, inputgrad')
    parser.add_argument('--model_name', default="resnet18", type=str,
                        help='resnet18 or vgg16_bn')
    parser.add_argument('--target_layer', default="layer4", type=str,
                        help='resnet18: layer4 or vgg16_bn:features')
    parser.add_argument('--n_images_save', default=10, type=int,
                        help='How many images to compute saliency maps from')
    parser.add_argument('--name_addition', default="test", type=str,
                        help='Additional string for filename')
    ARGS = parser.parse_args()
    # Create folder to saliency maps
    main()
    print('Saliency maps saved.')