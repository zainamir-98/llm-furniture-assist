"""
Example for a simple Feature Extractor based on ResNet50
"""
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from typing import List


def resnet50_extractor(img=None, img_path=None, model=None,
                      reduce_dim=False, reduction_dim=100):
    if model is None:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.eval()

    transform = transforms.ToTensor()

    # load cropped image
    # img_path = os.path.join("activity recognition", "mock data", "cropped", "rabbit_cropped.jpg")
    if img is not None:
        if not isinstance(img, torch.Tensor):
            img = transform(img)
        # img /= 255.0
    elif img_path:
        image = Image.open(img_path).convert('RGB')
        img = transform(image).unsqueeze(0)
    else:
        raise Exception("Missing input: provide either an image or image path")

    with torch.no_grad():
        # print(img.shape)
        features = model(img)

    if reduce_dim and features.shape[-1] != reduction_dim:
        features = features.view(features.size(0), -1)
        reduction_layer = nn.Linear(features.size(-1), reduction_dim)
        features = reduction_layer(features)

    # we can use the extracted features for the activity recognition
    #print(features.shape)
    return features


def pad_and_stack_vectors(vectors: List[torch.Tensor],
                            max_num_objects=8, pad_only=False, dim=1, squeeze=True):
    max_num_objects = max_num_objects
    num_objects = len(vectors)
    num_padding = max_num_objects - num_objects
    
    padded_tensor = torch.zeros_like(vectors[0])
    padded_vectors = vectors + [padded_tensor] * num_padding
    if not pad_only:
        if squeeze:
            stacked = torch.stack(padded_vectors, dim=dim).squeeze(0)
        else:
            stacked = torch.stack(padded_vectors, dim=dim)
        return stacked
    else:
        return padded_vectors