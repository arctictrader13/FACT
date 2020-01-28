from misc_functions import *
from matplotlib import pyplot as plt

import re

# all run on k values
k = [0.001, 0.005, 0.01, 0.05, 0.1]

# PATH variables
PATH = os.path.dirname(os.path.abspath(__file__)) + '/'

# console testing
# PATH = os.path.abspath(os.getcwd()) + "/"

# path to folder of pickle files
pp_experiments_path = PATH + "/pp_experiments"
experiment_pickles = os.listdir(pp_experiments_path)
# remove .pkl ending
for i in range(len(experiment_pickles)):
    #print(path)
    path = experiment_pickles[i]
    experiment_pickles[i] = path.replace(".pkl", "")

def plot_all_grads(results_dict, filename=None, div=False):
    plt.figure()
    axes = plt.gca()
    #axes.set_xlim([0, ARGS.k[-1]*100])
    axes.set_xlabel('% pixels removed')
    axes.set_ylabel('Absolute fractional output change')

    x_labels = [i*100 for i in k]

    for key, v in results_dict.items():
        # Plot the mean and variance of the predictive distribution on the 100000 data points.
        plt.plot(k, np.array(v[0]), linewidth=1.2, label=str(key))
        if div is False:
            plt.fill_between(k, np.array(v[0]) - np.array(v[1]), np.array(v[0]) + np.array(v[1]), alpha=1/3)

    plt.xticks(k, x_labels, rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.savefig(filename + ".png")
    # plt.show()

for j in range(len(experiment_pickles)):
    # load pkl file as dictionary
    all_results = load_obj(pp_experiments_path + "/" + experiment_pickles[j])

    # plot all gradients
    folder_path = pp_experiments_path + "/" + experiment_pickles[j] + "/"

    plot_all_grads(all_results[0], filename= folder_path + "scores")
    plot_all_grads(all_results[1], filename=folder_path + "probs")
    plot_all_grads(all_results[2], filename=folder_path + "kl")
    plot_all_grads(all_results[3], filename=folder_path + "topk_other_scores")
    plot_all_grads(all_results[4], filename=folder_path + "topk_other_probs")
    plot_all_grads(all_results[5], filename=folder_path + "max_other_scores")
    plot_all_grads(all_results[6], filename=folder_path + "max_other_probs")

