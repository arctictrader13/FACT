from misc_functions import *
from matplotlib import pyplot as plt
import re

# console testing
PATH = os.path.abspath(os.getcwd())

# path to folder of pickle files
pp_experiments_path = PATH + "/results/pp_experiments/"
experiment_pickles = os.listdir(pp_experiments_path)

os.chdir(pp_experiments_path)

# remove .pkl ending
for i in range(len(experiment_pickles)):
    #print(path)
    path = experiment_pickles[i]
    experiment_pickles[i] = path.replace(".pkl", "")

def merge_inputgrad(all_dict, inputgrad_dict):
    for i in range(len(all_dict)):
        all_dict[i]["inputgrad"] = inputgrad_dict[i]["inputgrad"]
    return all_dict

def rearrange_k_order(all_dict_list):
    myorder = [0, 1, 2, 5, 3, 6, 4]
    all_dict_list = [all_dict_list[i] for i in myorder]

    return all_dict_list

def merge_additional_k(all_dict, additional_dict):
    for i in range(len(all_dict)):
        for key in all_dict[i]:
            for j in range(len(all_dict[i][key])):
                all_dict[i][key][j]  = all_dict[i][key][j] + additional_dict[i][key][j]
                all_dict[i][key][j] = rearrange_k_order(all_dict[i][key][j])

    return all_dict


# merge them all
steps = range(0, 24, 3)

for i in steps:

    dict_first_run = load_obj(pp_experiments_path+experiment_pickles[i])
    dict_inputgrad = load_obj(pp_experiments_path+experiment_pickles[i+1])

    full_dict = merge_inputgrad(dict_first_run, dict_inputgrad)
    dict_additional = load_obj(pp_experiments_path+experiment_pickles[i+2])

    full = merge_additional_k(full_dict, dict_additional)

    save_obj(full, experiment_pickles[i] + "_full")

# console testing
PATH = os.path.abspath(os.getcwd()) + "/"

# path to folder of pickle files
pp_experiments_path = PATH + "results/pp_experiments/all_final_dicts"
experiment_pickles = os.listdir(pp_experiments_path)

os.chdir(pp_experiments_path)

for i in range(len(experiment_pickles)):
    #print(path)
    path = experiment_pickles[i]
    experiment_pickles[i] = path.replace(".pkl", "")

a = load_obj(pp_experiments_path + "/" + experiment_pickles[4])

k = [0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1]

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
    #plt.savefig(filename + ".png")
    plt.show()

