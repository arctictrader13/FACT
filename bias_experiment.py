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
from models.resnet import *
from models.vgg import *
from os import listdir
from PIL import Image

## console testing
# PATH = os.path.abspath(os.getcwd()) + "/"

# DATASET Description
# 136 female doctors
# 199 male doctors
# 258 female nurses
# 57 male nurses

# Data augmentation and normalization for training
# Just normalization for validation
transform_standard = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]), ])

data_dir = 'more_biased_dataset/'
model_dir = 'results/bias_experiment/model'
dataset = PATH + data_dir
model_path = PATH + "/" + model_dir

# TODO female and male doctors
# TODO male correct female misclassified

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"

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

def show_ex_images(dataloaders, data_stats):
    dataset_sizes, class_names = data_stats

    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['val']))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[class_names[x] for x in classes])

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


def visualize_model(model, dataloaders, data_stats, num_images=6):
    dataset_sizes, class_names = data_stats

    was_training = model.training
    model.eval()
    images_so_far = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

# show_ex_images(dataloaders)
#show_ex_images(dataloaders, data_stats)

def test_bad_files():
    for filename in listdir('./'):
        if filename.endswith('.jpg'):
            try:
                img = Image.open('./' + filename)  # open the image file
                img.verify()  # verify that it is, in fact an image
            except (IOError, SyntaxError) as e:
                os.remove("./" + filename)
                print('Bad file:', filename)


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

    if ARGS.model == "vgg":
        model_ft = vgg16_bn(pretrained=True)

        if ARGS.frozen == "True":
            frozen = "frozen"
            for param in model_ft.parameters():
                param.requires_grad = False
        else:
            frozen = "unfrozen"
        model_ft.classifier = nn.Sequential(
            nn.Linear(25088, 4096, bias=True),
            nn.ReLU(True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(512, 2))

    else:
        model_ft = resnet18(pretrained=True)

        num_ftrs = model_ft.fc.in_features

        if ARGS.frozen == "True":
            frozen = "frozen"

            for param in model_ft.parameters():
                param.requires_grad = False
        else:
            frozen = "unfrozen"

        # Here the size of each output sample is set to 2.
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        model_ft.fc = nn.Linear(num_ftrs, 2)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    if ARGS.model == "resnet":
        optimizer_ft = optim.SGD(model_ft.fc.parameters(), lr=ARGS.lr, momentum=0.9)
    else:
        optimizer_ft = optim.SGD(model_ft.classifier.parameters(), lr=ARGS.lr, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, dataloaders,  criterion, optimizer_ft, exp_lr_scheduler,
                           data_stats, num_epochs=ARGS.n_epochs)

    # visualize_model(model_ft)
    torch.save(model_ft.state_dict(), model_path + "/model_" + ARGS.model + "more_bias" +".pt")

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
    parser.add_argument('--frozen', default="True", type=str,
                        help='Freeze conv layers')
    parser.add_argument('--grads', default=["fullgrad"], type=str, nargs='+',
                        help='which grad methods to be applied')
    ARGS = parser.parse_args()

    os.chdir(PATH + "more_biased_dataset/train/doctor")
    print("Train Doctors:")
    test_bad_files()
    os.chdir(PATH + "more_biased_dataset/train/nurse")
    print("Train Nurse:")
    test_bad_files()
    os.chdir(PATH + "more_biased_dataset/val/doctor")
    print("Val Doctors:")
    test_bad_files()
    os.chdir(PATH + "more_biased_dataset/val/nurse")
    print("Val Nurse:")
    test_bad_files()

    main()
