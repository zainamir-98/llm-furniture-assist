import os
import numpy as np
import torch
from torch.utils import data
from torchvision.transforms import transforms
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import cv2 as cv
from PIL import Image
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from copy import copy

from dataset import ActionDataset, read_action_annotations


ACTIVITIES = {
    "take leg": 0,
    "assemble leg": 1,
    "grab drill": 2,
    "use drill": 3,
    "drop drill": 4,
    "take screw driver": 5,
    "use screw driver": 6,
    "drop screw driver": 7
}

# "Assemble leg" can be skipped
classes_to_skip = [1]

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomVerticalFlip(p=1.0),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.GaussianBlur(kernel_size=3),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

rotation_angle = 30


def parse_coco(ann_dir, file_name="annotations.json", name_only=True):
    N = 5 # For the length of an image name
    data = [] # List of dicts

    with open(os.path.join(ann_dir, file_name)) as f:
        annotation_file = json.load(f)
    image_files = [img_file.get("file_name") for img_file in annotation_file.get("images")]
    annotations = [{
        "bbox": annotation.get("bbox"),
        "bbox_mode": "XYWH",
        "category_id": annotation.get("category_id"),
        "image_id": annotation.get("image_id")
    } for annotation in annotation_file.get("annotations")]

    prefix = f"{ann_dir}"

    for img in image_files:
        if name_only:
            color_file = img
        else:
            color_file = os.path.join(prefix, img.split("/")[0], img.split("/")[1])
        d = {
            "file_name": color_file,
            "bb_info": [],
            "object_classes": []
        }
        for annotation in annotations:
            iid = str(annotation.get("image_id"))
            end = iid.split("_")[-1]
            length = len(end)
            img_id = iid[:-length] + (N - length) * "0" +  end

            if "/" in img:
                img_name = img.split("/")[1].split(".")[0] # Example: "images/00001.jpg" -> "00000"
            else:
                img_name = img.split(".")[0]
            if img_id == img_name:
                d["bb_info"].append(annotation.get("bbox"))
                d["object_classes"].append(annotation.get("category_id"))
        data.append(d)

    return data


ann_dir = os.path.join("activity recognition", "mock data")
root_dir = os.path.join("activity recognition", "mock data", "all_in_one", "all_images")
data_ = parse_coco(ann_dir, name_only=True, file_name="annotations_new.json")

actions = read_action_annotations(os.path.join(ann_dir, "action_annotations_new.json"))

aug_actions = {} 
aug_data_ = [] 

rounds = 1

for i, round in enumerate(range(rounds)):
    for image_info in data_:
        file_name = image_info["file_name"]
        bb_info = image_info["bb_info"]
        object_classes = image_info["object_classes"]
        try:
            action = actions[file_name]
        except KeyError:
            continue

        # Load frame
        path = os.path.join(root_dir, file_name)
        frame = cv.imread(path)

        if action not in classes_to_skip:
            augmented_image = transform(frame)

        else:
            continue

        size_y, size_x = frame.shape[:2]

        for i, _bb in enumerate(bb_info):
            bb = copy(_bb)
            bbox_tensor = torch.tensor(bb).float()
            bbox_tensor[1] = size_y - bbox_tensor[1] - bbox_tensor[3]

            ratio_x = 128 / size_x
            ratio_y = 128 / size_y
            bbox_tensor[0] *= ratio_x
            bbox_tensor[2] *= ratio_x
            bbox_tensor[1] *= ratio_y
            bbox_tensor[3] *= ratio_y

            # Update the bounding box coordinates in the list
            bb_info[i] = bbox_tensor.tolist()

        plt.imsave(os.path.join(root_dir, "..", "augmented", "aug3_" + file_name), augmented_image.permute(1, 2, 0).numpy())

        aug_actions["aug3_" + file_name] = action
        aug_data_.append({
            "file_name": "aug3_" + file_name,
            "bb_info": bb_info,
            "object_classes": object_classes
        })
    
    with open(os.path.join(root_dir, "..", "augmented", "aug3_action_annotations.json"), "w") as action_ann:
            json.dump(aug_actions, action_ann)

    with open(os.path.join(root_dir, "..", "augmented", "aug3_annotations.json"), "w") as annotations:
            json.dump(aug_data_, annotations)
