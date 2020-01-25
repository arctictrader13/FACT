import hashlib
from scipy.misc import imread, imresize, imshow
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import numpy as np
import hashlib, os
import imageio

path = os.getcwd()

nurses_male = path + "/nurses/nurse_male"
nurses_female = path + "/nurses/nurse_female"
doctors_male = path + "/doctors/doctor_male"
doctors_female= path + "/doctors/doctor_female"

os.chdir(doctors_female)

file_list = os.listdir()
print(len(file_list))

def file_hash(filepath):
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

duplicates = []
hash_keys = dict()
for index, filename in  enumerate(os.listdir('.')):  #listdir('.') = current directory
    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            filehash = hashlib.md5(f.read()).hexdigest()
        if filehash not in hash_keys:
            hash_keys[filehash] = index
        else:
            duplicates.append((index,hash_keys[filehash]))

print(len(duplicates))

# visualize duplicates
for file_indexes in duplicates[:30]:
    try:

        plt.subplot(121), plt.imshow(imageio.imread(file_list[file_indexes[1]]))
        plt.title(file_indexes[1]), plt.xticks([]), plt.yticks([])

        plt.subplot(122), plt.imshow(imageio.imread(file_list[file_indexes[0]]))
        plt.title(str(file_indexes[0]) + ' duplicate'), plt.xticks([]), plt.yticks([])
        plt.show()

    except OSError as e:
        continue
# remove duplicates
for index in duplicates:
    os.remove(file_list[index[0]])


