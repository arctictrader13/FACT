from os import rename, listdir
import os
path = os.getcwd()
from os.path import isfile, join
import re


def change_names_dir(dir_name, path):
    os.chdir(dir_name)
    filenames = [f for f in listdir(dir_name) if isfile(join(dir_name, f))]
    change_names(filenames)
    os.chdir(path)

def create_label_names(dir_name, path, label):
    os.chdir(dir_name)
    filenames = [f for f in listdir(dir_name) if isfile(join(dir_name, f))]

    for name in filenames:
        rename(name, re.sub('.jpg', "", name) + "_" + label + ".jpg")

    os.chdir(path)


def change_names(filenames):
    counter = 0
    for name in filenames:
        counter += 1
        rename(name, re.sub(r'.*.jpg', str(counter) + '.jpg', name))

nurses = path + "/biased_data/nurses/nurse_female"
change_names_dir(nurses, path)
create_label_names(nurses, path, "nurse_female")

doctors = path + "/biased_data/doctors/doctor_female"
change_names_dir(doctors, path)
create_label_names(doctors, path, "doctor_female")

doctors = path + "/biased_data/doctors/doctor_male"
change_names_dir(doctors, path)
create_label_names(doctors, path, "doctor_male")

nurses = path + "/biased_data/nurses/nurse_male"
change_names_dir(nurses, path)
create_label_names(nurses, path, "nurse_male")


