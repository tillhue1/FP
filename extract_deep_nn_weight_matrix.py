import torch
import torchvision.models as models
import torch

def get_mobilenet_target_mats():
    target_mats = []
    # Load the model
    model = models.googlenet(pretrained=True)#mobilenet_v2(pretrained=True)
    #
    # Put moel into eval mode
    model.eval()
    for layer in model.classifier:#model.fc_layers:#model.classifier:
        if isinstance(layer, torch.nn.Linear):
            # Obtain the weights of this layer
            weights = layer.weight.detach().numpy()
            target_mats.append(weights)
    return target_mats

if __name__ == "__main__":
    target_mats = get_mobilenet_target_mats()
    print(target_mats[0].shape)
