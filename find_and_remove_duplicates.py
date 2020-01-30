import hashlib
from scipy.misc import imread, imresize, imshow
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import numpy as np
import hashlib, os
import imageio

path = os.getcwd()

def file_hash(filepath):
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def remove_duplicates(image_path):
    duplicates = []
    hash_keys = dict()

    os.chdir(image_path)
    file_list = os.listdir()
    print("Number of files: {}".format(len(file_list)))

    for index, filename in  enumerate(os.listdir('.')):  #listdir('.') = current directory
        if os.path.isfile(filename):
            with open(filename, 'rb') as f:
                filehash = hashlib.md5(f.read()).hexdigest()
            if filehash not in hash_keys:
                hash_keys[filehash] = index
            else:
                duplicates.append((index,hash_keys[filehash]))

    print("Number of duplicates: {}".format(len(duplicates)))
    # remove duplicates
    for index in duplicates:
        os.remove(file_list[index[0]])
    os.chdir(path)



nurses_male = path + "/biased_data/nurses/nurse_male"
nurses_female = path + "/biased_data/nurses/nurse_female"
doctors_male = path + "/biased_data/doctors/doctor_male"
doctors_female= path + "/biased_data/doctors/doctor_female"

remove_duplicates(nurses_male)
remove_duplicates(nurses_female)
remove_duplicates(doctors_male)
remove_duplicates(doctors_female)

# visualize duplicates
# for file_indexes in duplicates[:10]:
#     try:
#
#         plt.subplot(121), plt.imshow(imageio.imread(file_list[file_indexes[1]]))
#         plt.title(file_indexes[1]), plt.xticks([]), plt.yticks([])
#
#         plt.subplot(122), plt.imshow(imageio.imread(file_list[file_indexes[0]]))
#         plt.title(str(file_indexes[0]) + ' duplicate'), plt.xticks([]), plt.yticks([])
#         plt.show()
#
#     except OSError as e:
#         continue




