"""
ADAPTED FROM
Transfer Learning for Computer Vision Tutorial
==============================================
**Author**: `Sasank Chilamkurthy <https://chsasank.github.io>`_
https://github.com/pytorch/tutorials/blob/master/beginner_source/transfer_learning_tutorial.py
"""

import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import argparse
from misc_functions import transform
from models.resnet import *
from models.vgg import *
from os import listdir
from PIL import Image

PATH = os.path.dirname(os.path.abspath(__file__)) + '/'

## console testing
# PATH = os.path.abspath(os.getcwd()) + "/"

# get transformator
transform_standard = transform()

# set PATH variables
data_dir = 'biased_dataset/'
model_dir = 'results/bias_experiment/model'
dataset = PATH + data_dir
model_path = PATH + "/" + model_dir

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def imshow(inp, file_name, title=None):
    """Imshow for Tensor."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    if title is not None:
        plt.title(title)
    plt.imshow(inp)
    plt.savefig(file_name)
    plt.pause(0.001)  # pause a bit so that plots are updated

def train_model(model, dataloaders,  criterion, optimizer, scheduler, data_stats, num_epochs=25):
    dataset_sizes, class_names = data_stats
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        #print(labels)
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def test_bad_files():
    """
    Apparently scraping yielded corrputed files that were causing issues on GPU of cluster. Therefore, detect corrupted
    images and remove them.
    :return:
    """
    for filename in listdir('./'):
        if filename.endswith('.jpg'):
            try:
                img = Image.open('./' + filename)  # open the image file
                img.verify()  # verify that it is, in fact an image
            except (IOError, SyntaxError) as e:
                os.remove("./" + filename)
                print('Bad file:', filename)

def detect_and_remove_corrupt_ims():
    """
    For all datasplits detects and removes corrupt images.
    :return:
    """
    os.chdir(PATH + "biased_dataset/train/doctor")
    print("Train Doctors:")
    test_bad_files()
    os.chdir(PATH + "biased_dataset/train/nurse")
    print("Train Nurse:")
    test_bad_files()
    os.chdir(PATH + "biased_dataset/val/doctor")
    print("Val Doctors:")
    test_bad_files()
    os.chdir(PATH + "biased_dataset/val/nurse")
    print("Val Nurse:")
    test_bad_files()

def initialize_model(model_type, frozen_bool="False"):
    """
    Initializes pretrained model. Frozen conv layers or full finetuning.
    :param model_type:
    :param frozen_bool:
    :return:
    """

    if model_type== "vgg":
        model = vgg16_bn(pretrained=True)

        if frozen_bool == "True":
            frozen = "frozen"
            for param in model.parameters():
                param.requires_grad = False
        else:
            frozen = "unfrozen"
        model.classifier = nn.Sequential(
            nn.Linear(25088, 4096, bias=True),
            nn.ReLU(True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(512, 2))

    else:
        model = resnet18(pretrained=True)
        num_ftrs = model.fc.in_features

        if frozen_bool == "True":
            # frozen names the obtained model depending on whether conv layers were frozen or not
            frozen = "frozen"
            for param in model.parameters():
                param.requires_grad = False
        else:
            frozen = "unfrozen"

        # Here the size of each output sample is set to 2.
        model.fc = nn.Linear(num_ftrs, 2)

    return model, frozen

def initialize_optimizer(model_type, model, lr):
    # Observe that all parameters are being optimized
    if model_type == "resnet":
        optimizer = optim.SGD(model.fc.parameters(), lr=lr, momentum=0.9)
    elif model_type == "vgg":
        optimizer = optim.SGD(model.classifier.parameters(), lr=lr, momentum=0.9)

    return optimizer

def main():

    image_datasets = {x: datasets.ImageFolder(os.path.join(dataset, x),
                                              transform_standard)
                      for x in ['train', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=ARGS.batch_size,
                                                  shuffle=True)
                   for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    data_stats = [dataset_sizes, class_names]

    model, frozen = initialize_model(ARGS.model, frozen_bool=ARGS.frozen)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = initialize_optimizer(ARGS.model, model, ARGS.lr)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model = train_model(model, dataloaders,  criterion, optimizer, exp_lr_scheduler,
                           data_stats, num_epochs=ARGS.n_epochs)

    # visualize_model(model_ft)
    torch.save(model.state_dict(), model_path + "/model_" + ARGS.name + "_" + ARGS.model + "bias_" + frozen + ".pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=2, type=int,
                        help='Freeze conv layers')
    parser.add_argument('--n_epochs', default=5, type=int,
                        help='Number of epochs to train')
    parser.add_argument('--lr', default=0.001, type=int,
                        help='Learning Rate')
    parser.add_argument('--model', default="resnet", type=str,
                        help='vgg or resnet')
    parser.add_argument('--frozen', default="False", type=str,
                        help='Freeze conv layers?')
    parser.add_argument('--name', default="", type=str,
                        help='Extra to name of model?')
    ARGS = parser.parse_args()

    main()
