'''
File handels the evaluation of the approximated matrices.
There are several functions to evaluate the approximation quality.

There are four pretrained models we want to compare
to archive a more stable results.

- mobilenet     (classifier  no fc)
  last layer Linear(in_features=1280, out_features=1000, bias=True)

- densenet161   (classifier  no fc)
  Linear(in_features=2208, out_features=1000, bias=True)

- resnet        (fc no classifier)
  Linear(in_features=512, out_features=1000, bias=True)

- googlenet     (fc no classifier)
  Linear(in_features=1024, out_features=1000, bias=True)

This file implements the following functions.
- get_model : Load pretrained pytorch model.
- change_last_layer : Changes the last layer of the model.
- load_imagenet : Load imagenet data.
- get_accuracy : Evaluates the accuracy of the model.
'''

import os
from typing import List, Union

import torch
from torchvision import datasets, transforms
import torchvision.models as models


def get_model(identifier: str = 'mobilenet_v2'):
    '''
    Method loads pretrained pytorch model.
    Allowed id's are :
    - "mobilenet_v2" (default)
    - "resnet18"
    - "densenet161"
    - "googlenet"

    Args :
    identifier : ID for model selection.

    Info :
    Downloaded models are stored in
    https://download.pytorch.org/models/xxx.pth
    '''

    allowed_networks = ['mobilenet_v2',
                        'resnet18',
                        'densenet161',
                        'googlenet']

    if identifier not in allowed_networks:
        raise Exception('Identifier is not in allowed')

    # We load the model by "model = models.name()"
    class_method = getattr(models, identifier)
    model = class_method(pretrained=True)

    # Set evaluation mode
    model.eval()

    return model


def get_last_layer(model_name: str):
    '''
    Function returns the last layer of a given neural network.
    '''

    model = get_model(model_name)
    fc_layers = ('resnet18', 'googlenet')
    classifier_layers = ('mobilenet_v2', 'densenet161')

    if model_name in fc_layers:
        return model.fc.weight.data

    elif model_name in classifier_layers:
        return model.classifier.weight.data


def change_last_layer(model,
                      new_last_layer=None):
    '''
    Function changes the last layer of the model.
    The last layer is removed and exchanged with
    "new_last_layer". Since different models have different attributes
    we need to adapt the changing of the last layer.

    Args :
    model : Model to change the last layer.
    new_last_layer : Layer to set if None we return the original
    model.
    copy_model : Boolean indicating if we make a copy of the model.

    Returns a model with a new last layer.
    '''

    # Change last layer of the model
    fc_layers = ('resnet18', 'googlenet')
    classifier_layers = ('mobilenet_v2', 'densenet161')

    # Return model without changes
    if new_last_layer is None:
        pass

    elif model.__class__.__name__.lower() in fc_layers:

        old_shape = model.fc.weight.data.shape
        new_shape = new_last_layer.shape

        if new_shape != old_shape:
            raise Exception('The new layers shape does not match')

        model.fc.weight.data = new_last_layer

    elif model.__class__.__name__.lower() in classifier_layers:
        old_shape = model.classifier.weight.data.shape
        new_shape = new_last_layer.shape

        if new_shape != old_shape:
            raise Exception('The new layers shape does not match')

        model.classifier.weight.data = new_last_layer

    else:
        raise Exception('The models last layer type is not known')

    return model


def get_random_weights(model_name: str) -> torch.tensor:
    '''
    Function returns a random weight matrix (torch tensor).
    This method is just for debugging purpose.
    Returns a torch tensor with shape defined by model.
    - mobilenet     (classifier  no fc)
    last layer Linear(in_features=1280, out_features=1000, bias=True)

    - densenet161   (classifier  no fc)
    Linear(in_features=2208, out_features=1000, bias=True)

    - resnet        (fc no classifier)
    Linear(in_features=512, out_features=1000, bias=True)

    - googlenet     (fc no classifier)
    Linear(in_features=1024, out_features=1000, bias=True)
    '''

    rand_approximation = None

    if model_name == 'mobilenet':
        rand_approximation = torch.rand((1000, 1280))

    elif model_name == 'densenet161':
        rand_approximation = torch.rand((1000, 2208))

    elif model_name == 'resnet18':
        rand_approximation = torch.rand((1000, 512))

    elif model_name == 'googlenet':
        rand_approximation = torch.rand((1000, 1024))

    else:
        return None

    return rand_approximation


def load_imagenet(data_set_type: str,
                  batch_size: int = 4) -> torch.utils.data.DataLoader:
    '''
    Method loads and transforms imagenet data.
    To use this method one first have to download and
    extract the data from
    (here: https://download.pytorch.org/tutorial/hymenoptera_data.zip)
    in data/imgnet.
    https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

    Args :
    data_set : The data set to choose from ["val", "train"]
    '''

    allowed_sets = ('val', 'train')

    if data_set_type not in allowed_sets:
        raise Exception(f'data_set_type not in {allowed_sets}')

    # Use default directory
    data_directory = os.path.join('data', 'imagenet')

    # Preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    # Load images from folder
    dataset = \
        datasets.ImageNet(root=data_directory,
                          split=data_set_type,
                          transform=preprocess)

    # Define dataloader
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=4)

    return dataloader


def get_accuracy(matches_list: List[bool]) -> float:
    '''
    Function evaluates the prediction accuracy.
    Given a matches list containing boolean indicating
    if the prediction matches the label.

    Args :
    matches_list : List of boolean.

    Returns the ratio of True/len(list).
    '''

    return sum(matches_list)/len(matches_list)


def evaluate(model_name: str,
             approximation: Union[None, torch.tensor] = None,
             data_set_type: str = 'val',
             batch_size: int = 4):
    '''
    Evaluate the performance of a given model & approximation of the
    last layer.

    Args :
    model_name : Model to evaluate.
    approximation : Approximation of the last layer.
    data_set_type : Dataset type possible values are "devkit", "val".
    batch_size : Batchsize use values 2^N
    '''

    # Load model by name
    model = get_model(model_name)

    # Approximate the last layer
    model = change_last_layer(model, approximation)

    # Load data
    loader = load_imagenet(data_set_type=data_set_type,
                           batch_size=batch_size)

    # define device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Do predictions
    match_list = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)

            matches = (labels == predictions).tolist()
            match_list.extend(matches)

            # TODO remove just for debugging
            if len(match_list) > 200:
                break
            print(f'Prediction accuracy : {get_accuracy(match_list):.3f}  ', end='\r')

    print(f'Prediction accuracy : {get_accuracy(match_list):.3f}')


if __name__ == '__main__':
    # networks : mobilenet_v2, densenet161, googlenet, resnet18
    # datasubsets : val, (train,) devkit
    model_name = 'mobilenet_v2'
    data_subset = 'val'
    do_random_approximation = False

    # Get a random approximation for debugging purpose
    approximation = None
    if do_random_approximation:
        approximation = get_random_weights(model_name)

    # Do evaluation
    evaluate(model_name=model_name,
             approximation=approximation,
             data_set_type=data_subset)
