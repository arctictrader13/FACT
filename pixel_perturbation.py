from torchvision import datasets, transforms, utils, models
from misc_functions import *
from gradcam import grad_cam
from functools import reduce
from saliency.inputgradient import Inputgrad
import gc
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from models.resnet import *
from models.vgg import *

# PATH variables
PATH = os.path.dirname(os.path.abspath(__file__)) + '/'


data_PATH = PATH + 'dataset/'
result_path = PATH + 'results/'

# get unnormalize object for plotting saliency maps
unnormalize = unnormalize()
# same transformations for each dataset
transform_standard = transform()

# prevents F.interpolate from random behaviour which caused Cuda memory errors
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# HELPER FUNCTIONS
def print_dict(dictionary, div=False):
    """
    For Debugging purposes
    :param dictionary:
    :param div:
    :return:
    """
    for k, v in dictionary.items():
        print("Gradient method: {}".format(k))
        if div is True:
            print("KL Div: {}".format(v[0]))
        else:
            print("Mean: {}, Std: {}".format(v[0], v[1]))

def kl_div(P, Q):
    """
    returns actual KL divergence. torch.nn.functional.kl_div returns different values
    :param P: discrete distribution
    :param Q: discrete distribution
    :return:
    """
    kl = (P * (P / Q).log()).sum(1)
    return kl

def compute_difference_metrics(initial_output, final_output, tmp_results):
    """
    Given NN output before and after removal, all absolute difference metrics are stored.
    :param initial_output: model.forward(x)
    :param final_output:  model.forward(x_salient_pixels_removed)
    :param tmp_results: object to store all temporary results
    :return:
    """

    # get probabilities instead of output scores
    initial_probabilities = F.softmax(initial_output, dim=1)
    final_probabilities = F.softmax(final_output, dim=1)
    # compute kl divergence on probability vectors
    kldiv = kl_div(final_probabilities, initial_probabilities)

    # score differences
    tmp_score_diffs = get_temp_result(initial_output, final_output)
    # prob differences
    tmp_prob_diffs = get_temp_result(initial_probabilities, final_probabilities)

    # changes of mean class score other classes based on topk classes, otherwise division by zero
    tmp_other_diffs = get_other_class_change(initial_output, final_output, ARGS.topk)
    tmp_other_probs_diffs = get_other_class_change(initial_probabilities, final_probabilities, ARGS.topk)

    # max change in other classes than most confident one
    tmp_other_max_diffs = tmp_other_diffs.max(1)[0]
    tmp_other_max_prob_diffs = tmp_other_probs_diffs.max(1)[0]

    # save per image
    tmp_results[0].append(np.round(tmp_score_diffs.tolist(), 8))
    tmp_results[1].append(np.round(tmp_prob_diffs.tolist(), 8))
    tmp_results[2].append(np.round(kldiv.tolist(), 8))
    tmp_results[3].append(np.round(tmp_other_diffs.tolist(), 8))
    tmp_results[4].append(np.round(tmp_other_probs_diffs.tolist(), 8))
    tmp_results[5].append(np.round(tmp_other_max_diffs.tolist(), 8))
    tmp_results[6].append(np.round(tmp_other_max_prob_diffs.tolist(), 8))

    return tmp_results

def initialize_grad_and_model(grad_type, model_name, device):
    """
    Given a saliency method (grad_type), the exact modelname (vgg16, vgg16_bn,...) and the device, returns model and
    grad object
    :param grad_type:
    :param model_name:
    :param device:
    :return:
    """
    model, grad = None, None
    if grad_type == 'gradcam':
        # Gradcam
        model, grad = initialize_grad_cam(model_name, device)
    else:
        model = initialize_fullgrad(model_name, device)
        # same model for Inputgrad and Fullgrad, different grad object
        if grad_type == "fullgrad":
            # Initialize Gradient objects
            grad = FullGrad(model)
        elif grad_type == "inputgrad":
            grad = Inputgrad(model)

    return model, grad

def get_temp_result(initial_out, final_out):
    """
    Takes NN outputs before and after removal to compute absolute fractional differences
    :param initial_out:
    :param final_out:
    :return:
    """
    # initially most confident class
    initial_class_scores, predicted_class = initial_out.max(1)
    # same value after modification
    final_class_scores = final_out.index_select(1, predicted_class).max(0)[0]
    # absolute fractional difference of raw results
    tmp_result = abs(final_class_scores - initial_class_scores) / initial_class_scores

    return tmp_result

def get_other_class_change(initial_out, final_out, topk):
    """
    Takes NN output before and after removal and the number of most confident classes. Then gets topk classes other than
    the most confident class before and computes bs fractional differences to after removal.
    and
    :param initial_out:
    :param final_out:
    :param topk:
    :return:
    """
    other_initial_scores = get_topk_other_scores(initial_out, initial_out, topk)
    other_final_scores = get_topk_other_scores(initial_out, final_out, topk)

    # absolute fractional difference of raw results
    tmp_result = abs(other_initial_scores - other_final_scores) / other_initial_scores

    return tmp_result

def get_other_classes_scores(initial_out, final_out):
    """
    Takes NN output before and after removal, removes most confident class before removal and considers all other classes
    to return respective tensors without most confident class.
    :param initial_out:
    :param final_out:
    :return:
    """
    # most confident class before removal
    _, predicted_class = initial_out.max(1)
    # did not find an easier method to remove the most confident class from tensor
    ind_tensor = torch.LongTensor(initial_out.size()[0] * [list(range(1000))]).to("cpu")
    new_tensor = torch.LongTensor(initial_out.size()[0] * [[1] * 999]).to("cpu")
    final_class_scores = torch.zeros(initial_out.size()[0], 999)
    # remove predicted class
    for i in range(len(predicted_class)):
        new_tensor[i] = np.delete(ind_tensor[i], predicted_class[i], None)
        final_class_scores[i] = final_out[i, new_tensor[i]]

    return final_class_scores

def get_topk_other_scores(initial_out, final_out, topk):
    """
    Takes NN output before and after removal, gets topk other class scores and returns topk tensors without topk
    most confident classes.
    :param initial_out:
    :param final_out:
    :param topk:
    :return:
    """

    other_initial_scores = get_other_classes_scores(initial_out, initial_out)
    other_final_scores = get_other_classes_scores(initial_out, final_out)

    topk_scores, topk_ind = torch.topk(other_initial_scores, k=topk)

    final_class_scores = torch.zeros(other_initial_scores.size()[0], topk)

    # remove predicted class
    for i in range(len(topk_ind)):
        final_class_scores[i] = other_final_scores[i, topk_ind[i]]

    return final_class_scores

def get_max_other_class_change(initial_output, final_output):
    """
    Takes NN output before and after removal, and computes max ab fract change of the most confident of the topk classes
    other than the initially most confident class.
    :param initial_output:
    :param final_output:
    :return:
    """
    other_initial_scores = get_other_classes_scores(initial_output, initial_output)
    other_final_scores = get_other_classes_scores(initial_output, final_output)

    changes = abs(other_initial_scores - other_final_scores) / other_initial_scores
    tmp_result = changes.max(1)

    return tmp_result

def append_mean_std(tmp_results, means, stds):
    """
    Append batch means to mean list that keeps track the mean and stds for each k value.
    :param tmp_results:
    :param means:
    :param stds:
    :return:
    """
    means.append(np.round(np.mean(tmp_results), 8))
    stds.append(np.round(np.std(tmp_results), 8))

def plot_all_grads(results_dict, filename=None):
    """
    Plots all metrics for one specification (model and which kind of removal)
    :param results_dict:
    :param filename:
    :param div:
    :return:
    """
    plt.figure()
    axes = plt.gca()
    # axes.set_xlim([0, ARGS.k[-1]*100])
    axes.set_xlabel('% pixels removed')
    axes.set_ylabel('Absolute fractional output change')

    x_labels = [i * 100 for i in ARGS.k]

    for key, v in results_dict.items():
        # Plot the mean and variance of the predictive distribution on the 100000 data points.
        plt.plot(ARGS.k, np.array(v[0]), linewidth=1.2, label=str(key))
        plt.fill_between(ARGS.k, np.array(v[0]) - np.array(v[1]), np.array(v[0]) + np.array(v[1]), alpha=1 / 3)
    plt.xticks(ARGS.k, x_labels, rotation=45)
    plt.tight_layout()
    plt.legend()
    #plt.savefig(filename + ".png")
    plt.show()

def initialize_grad_cam(model_name, device, pretrained=True):
    """
    Gradcam needs original torch model object to train on. Therefore different initialization than fullgrad.
    :param model_name:
    :param device:
    :param pretrained:
    :return:
    """
    model = models.__dict__[model_name](pretrained=pretrained)
    model.to(device)
    model.eval()
    gcam = grad_cam.GradCAM(model=model)

    return model, gcam

def initialize_fullgrad(model_name, device):
    """
    Initializes Fullgrad Object given modelname and device.
    :param model_name:
    :param device:
    :return:
    """
    model = eval(model_name)(pretrained=True)
    model = model.to(device)
    model.eval()

    return model

def compute_saliency_per_grad(grad_type, grad, data, target_layer, target_class=None):
    """
    Given the grad_type, computes saliency maps for a batch of images: data.
    :param grad_type:
    :param grad:
    :param data:
    :param target_layer: vgg: features, resnet: layer4
    :param target_class: To visualize wrong predictions take original target
    :return:
    """
    saliency = None

    if grad_type == "fullgrad" or grad_type == "inputgrad":
        # print("calculating saliency")
        saliency = grad.saliency(data, target_class=target_class)

    elif grad_type == "gradcam":
        probs, ids = grad.forward(data)
        # Grad-CAM
        grad.backward(ids=ids[:, [0]])
        saliency = grad.generate(target_layer=target_layer)

    return saliency

def get_filename(result_path, grad_type, index):
    model_name = ARGS.model + ARGS.model_type

    filename = result_path + "/" + grad_type + "_" + model_name + "_" + str(index) + ".png"
    return filename

def save_saliency_map_batch(saliency, data, result_path, grad_type, index):
    """
    Save all  saliency maps of a batch of images.
    :param saliency:
    :param data:
    :param result_path:
    :param grad_type:
    :param index:
    :return:
    """
    for i in range(len(data)):
        im = unnormalize(data[i, :, :, :].cpu())
        im = im.view(1, 3, 224, 224)[-1, :, :, :]
        reg = saliency[i, :, :, :]
        filename = get_filename(result_path, grad_type, index)
        # print("filename:{}".format(filename))
        save_saliency_map(im, reg, filename)

def print_memory():
    """
    print all cuda tensors for memory leakage issues.
    :return:
    """
    total = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())

                if len(obj.size()) > 0:
                    if obj.type() == 'torch.cuda.FloatTensor':
                        total += reduce(lambda x, y: x * y, obj.size()) * 32
                    elif obj.type() == 'torch.cuda.LongTensor':
                        total += reduce(lambda x, y: x * y, obj.size()) * 64
                    elif obj.type() == 'torch.cuda.IntTensor':
                        total += reduce(lambda x, y: x * y, obj.size()) * 32
                    # else:
                    # Few non-cuda tensors in my case from dataloader
        except Exception as e:
            pass
    print("{} GB".format(total / ((1024 ** 3) * 8)))

def get_sample_loader():
    dataset = data_PATH
    sample_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(dataset, transform=transform_standard),
        batch_size=ARGS.batch_size, shuffle=False)

    return sample_loader

def get_salient_type():
    if ARGS.most_salient == "True":
        salient_type = "most"
    else:
        salient_type = "least"

    return salient_type

def print_results(all_results):
    print("For K values: {}".format(ARGS.k))
    print("############ Score absolute fractional differences ############")
    print_dict(all_results[0])
    print("############ Probs absolute fractional differences ############")
    print_dict(all_results[1])
    print("KL divergences per k")
    print_dict(all_results[2])
    print("############ Top: {} Other Score absolute fractional differences ############".format(ARGS.topk))
    print_dict(all_results[3])
    print("############ Top: {} Other Probs absolute fractional differences ############".format(ARGS.topk))
    print_dict(all_results[4])
    print("############ Max Other Score absolute fractional differences ############")
    print_dict(all_results[5])
    print("############ Max other Probs absolute fractional differences ############")
    print_dict(all_results[6])

def initialize_means_std_dict():
    means_std_dict= {}
    means_std_dict["score_means"] = []
    means_std_dict["score_stds"] = []
    means_std_dict["prob_means"] = []
    means_std_dict["prob_stds"] = []
    means_std_dict["kl_div_means"] = []
    means_std_dict["kl_div_stds"] = []
    means_std_dict["other_score_means"] = []
    means_std_dict["other_score_stds"] = []
    means_std_dict["other_prob_means"] = []
    means_std_dict["other_prob_stds"] = []
    means_std_dict["other_max_score_means"] = []
    means_std_dict["other_max_score_stds"] = []
    means_std_dict["other_max_prob_means"] = []
    means_std_dict["other_max_prob_stds"] = []

    return means_std_dict

def append_tmp_means_stds(tmp_results, means_std_dict):
    # save mean and std of
    append_mean_std(tmp_results[0], means_std_dict["score_means"], means_std_dict["score_stds"])
    append_mean_std(tmp_results[1], means_std_dict["prob_means"], means_std_dict["prob_stds"])
    append_mean_std(tmp_results[2], means_std_dict["kl_div_means"], means_std_dict["kl_div_stds"])
    append_mean_std(tmp_results[3], means_std_dict["other_score_means"], means_std_dict["other_score_stds"])
    append_mean_std(tmp_results[4], means_std_dict["other_prob_means"], means_std_dict["other_prob_stds"])
    append_mean_std(tmp_results[5], means_std_dict["other_max_score_means"], means_std_dict["other_max_score_stds"])
    append_mean_std(tmp_results[6], means_std_dict["other_max_prob_means"], means_std_dict["other_max_prob_stds"])

def append_batch_means_stds(all_results, grad_type, means_std_dict):
    all_results[0][grad_type] = [means_std_dict["score_means"], means_std_dict["score_stds"]]
    all_results[1][grad_type] = [means_std_dict["prob_means"], means_std_dict["prob_stds"]]
    all_results[2][grad_type] = [means_std_dict["kl_div_means"], means_std_dict["kl_div_stds"]]
    all_results[3][grad_type] = [means_std_dict["other_score_means"], means_std_dict["other_score_stds"]]
    all_results[4][grad_type] = [means_std_dict["other_prob_means"], means_std_dict["other_prob_stds"]]
    all_results[5][grad_type] = [means_std_dict["other_max_score_means"], means_std_dict["other_max_score_stds"]]
    all_results[6][grad_type] = [means_std_dict["other_max_prob_means"], means_std_dict["other_max_prob_stds"]]

def get_pickle_file_name(model_name, salient_type):
    # plot for all gradient methods stds and means for all k% values
    file_name = ARGS.dataset + "_" + model_name + "_" + salient_type + "_" + ARGS.replacement + str(ARGS.n_images)
    return file_name

def main():
    device = ARGS.device
    sample_loader = get_sample_loader()
    salient_type = get_salient_type()
    model_name = ARGS.model + ARGS.model_type
    save_path = PATH + 'results/' + ARGS.dataset

    # initialize results dictionary: key: gradient method (random, fullgrad,...), values: [[mean, std],..] per k%
    all_results = [{}, {}, {}, {}, {}, {}, {}]
    total_features = 224 * 224

    for grad_type in ARGS.grads:
        model, grad = initialize_grad_and_model(grad_type, model_name, device)
        # makes sure each saliency method only saves image once
        grad_counter = 0
        means_std_dict = initialize_means_std_dict()
        print("grad_type:{}".format(grad_type))

        for i in ARGS.k:
            # print("grad:{}".format(grad))
            grad_counter += 1
            k_most_salient = int(i * total_features)
            # print("k_most_salient:{}".format(k_most_salient))
            counter = 0
            tmp_results = [[], [], [], [], [], [], []]  # score diffs, prob diffs, kl divs, score other diffs, prob other diffs, max other diffs

            for batch_idx, (data, target) in enumerate(sample_loader):
                counter += 1
                # console testing
                # data, _ = next(iter(sample_loader))
                # for debugging purposes
                if counter % 100 == 0:
                    print("{} image batches processed".format(counter))
                if counter == ARGS.n_images:
                    break

                data = data.to(device).requires_grad_()

                # Run Input through network (two different networks if gradcam or fullgrad)
                with torch.no_grad():
                    initial_output = model.forward(data)
                    initial_out = initial_output.to(torch.device("cpu"))

                    # compute saliency maps for grad methods not random
                if grad_type != "random":
                    # print("data size:{}".format(data.size()))
                    cam = compute_saliency_per_grad(grad_type, grad, data, target_layer="features")

                    if ARGS.save_grad is True and grad_counter == 1 and counter <= ARGS.n_save:
                        save_saliency_map_batch(cam, data, result_path, grad_type, salient_type, counter)

                    new_data = remove_salient_pixels(data, cam, num_pixels=k_most_salient,
                                                     most_salient=ARGS.most_salient,
                                                     replacement=ARGS.replacement)
                    new_data.to("cpu")
                    # output after pixel perturbation
                    with torch.no_grad():
                        final_output = model.forward(new_data)
                        final_out = final_output.to("cpu")

                    tmp_results = compute_difference_metrics(initial_out, final_out, tmp_results)

                # change pixels based on random removal
                elif grad_type == "random":
                    # run n_random_runs for random pixel removal
                    sample_seeds = np.random.randint(0, 10000, ARGS.n_random_runs)
                    for seed in sample_seeds:
                        tmp_data = remove_random_salient_pixels(data, seed, k_percentage=i,
                                                                replacement=ARGS.replacement)

                        with torch.no_grad():
                            final_output = model.forward(tmp_data)
                            final_out = final_output.to("cpu")

                        tmp_results = compute_difference_metrics(initial_out, final_out, tmp_results)

            append_tmp_means_stds(tmp_results, means_std_dict)

        append_batch_means_stds(all_results, grad_type, means_std_dict)


    # plot for all gradient methods stds and means for all k% values
    file_name = get_pickle_file_name(model_name, salient_type)
    save_experiment_file = result_path + file_name
    print_results(all_results)
    # save dictionary
    save_obj(all_results, save_experiment_file)
    # print_memory()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=5, type=int,
                        help='')
    parser.add_argument('--dataset', default="imagenet", type=str,
                        help='which dataset')
    parser.add_argument('--device', default="cuda:0", type=str,
                        help='cpu or gpu')
    parser.add_argument('--grads', default=["gradcam"], type=str, nargs='+',
                        help='which grad methods to be applied')
    parser.add_argument('--k', default=[0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1], type=float, nargs="+",
                        help='Percentage of k% most salient pixels')
    parser.add_argument('--most_salient', default="True", type=str,
                        help='most salient = True or False depending on retrain or pixel perturbation')
    parser.add_argument('--model', default="vgg", type=str,
                        help='which model to use')
    parser.add_argument('--model_type', default="16_bn", type=str,
                        help='which model type: resnet_18, ...')
    parser.add_argument('--n_images', default=50, type=int,
                        help='Test for n_images images ')
    parser.add_argument('--n_random_runs', default=5, type=int,
                        help='Number of runs for random pixels to be removed to decrease std of random run')
    parser.add_argument('--n_save', default=50, type=int,
                        help='Save saliency maps for first n_save images')
    parser.add_argument('--target_layer', default="features", type=str,
                        help='Which layer to be visualized in GRADCAM')
    parser.add_argument('--replacement', default="black", type=str,
                        help='black = 1.0 or mean = [0.485, 0.456, 0.406]')
    parser.add_argument('--save_grad', default=False, type=bool,
                        help='saliency map to be saved?')
    parser.add_argument('--topk', default=100, type=int,
                        help='Number of other classes considered for metric of other class changes')
    ARGS = parser.parse_args()
    main()
