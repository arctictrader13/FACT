
import os

import torch
from torchvision import datasets, transforms
from datetime import datetime

# Import saliency methods and models
from saliency.fullgrad import FullGrad
from saliency.simple_fullgrad import SimpleFullGrad
from models.vgg import vgg16_bn
from models.resnet import resnet18
from misc_functions import create_folder, compute_and_store_saliency_maps, remove_salient_pixels


# PATH variables
PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
dataset = PATH + 'dataset/'

batch_size = 1
k_most_salient = 100
max_train_steps = 3
learning_rate = 0.01

device = "cpu"

def train(data_loader, model, max_train_steps, learning_rate, use_saliency=False):
    criterion = torch.nn.CrossEntropyLoss() 
    optimizer = torch.optim.RMSprop(model.parameters(), \
                                    lr=learning_rate)
    accuracies = []
    losses = []
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        if use_saliency:
            saliency_map = torch.load(os.path.join(dataset, "saliency_maps", "imagenet_custom_resnet18", "full", "saliency_map_" + str(step)))
            batch_inputs = remove_salient_pixels(batch_inputs, saliency_map)

        output = model.forward(batch_inputs)
        
        loss = criterion(output, batch_targets)
        accuracy = float(sum(batch_targets == torch.argmax(output, 1))) / float(batch_size)

        # conpute gradients
        loss.backward()

        # update weights
        optimizer.step()

        if step % 1 == 0:
            accuracies += [accuracy]
            losses += [loss]
            if (len(losses) > 2 and abs(losses[-1] - losses[-2]) < 0.0001) or \
               (loss == 0.0 and accuracy == 1.0):
                break
            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    max_train_steps, batch_size, accuracy, loss
            ))

        if step == max_train_steps:
            break


def remove_and_retrain(data_loader, model):
    train(data_loader, model, max_train_steps, learning_rate)
    saliency_path = os.path.join(dataset, "saliency_maps", "imagenet_custom_resnet18")
    create_folder(saliency_path)
    compute_and_store_saliency_maps(data_loader, model, device, saliency_path)
    # save model
    train(data_loader, model, max_train_steps, learning_rate, use_saliency=True)


def main():
    # Dataset loader for sample images
    data_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(dataset, transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])),
        batch_size=batch_size, shuffle=False)

    # model = vgg16_bn(pretrained=True)
    model = resnet18(pretrained=True)
        
    remove_and_retrain(data_loader, model)


if __name__ == "__main__":
    main()
