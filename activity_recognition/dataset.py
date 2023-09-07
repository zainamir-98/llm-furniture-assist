"""
Dataset and -loader class that takes frames and bounding box information, crops the
image and applies some optional preprocessing for later (postprocessing here
if you want to be exact ... lol).
Class takes a root directory of frames and bounding box information and
returns the cropped and (optional) transformed region of interest.
"""
import os
import numpy as np
import torch
from torch.utils import data
from torchvision.transforms import transforms
import cv2 as cv
from typing import List
import json

from feature_extractor import feature_extractor, pad_and_stack_vectors


################ NOT USED ################
class FrameDataSet(data.Dataset):
    
    def __init__(self, root_dir: str,
                 data: List[dict],
                 transform: transforms.Compose = None,
                 batch_size: int = 1):
        """Constructor

        Args:
            root_dir (str): directory of the frames
            data (List[dict]): bounding box information, format: [{"file_name": str, "bb_info": List[int]}, ...]
            transform (transforms.Compose, optional): Compose Transformer Pipline. Defaults to None
            batch_size (int): batch size to return, note: not really used yet, Defaults to 1
        """
        self.root_dir = root_dir
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((256, 256))
        ])
        else:
            self.transform = transform
        self.batch_size = batch_size

        # TODO: take bb info in coco format or preprocess?
        # bb_info: [x, y, width, height] or if multiple objects in scene: [[x, y, w, h], [x, y, w, h]]
        # self.data = [
        #     (os.path.join(root_dir, frame_id), bbox_list) for frame_id, bbox_list in bb_info.items()
        # ]
        self.data = data



    def _crop(self, img: np.ndarray, bbox: List[int]) -> torch.Tensor:
        roi = img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
        return roi

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> dict:
        # path, bbox_list = self.data[idx]
        path = self.data[idx].get("file_name")
        # 00001.jpg for example
        frame_name = path.split("\\")[-1]
        bbox_list = self.data[idx].get("bb_info")
        img = cv.imread(path)
        # multiple objects in scene if bbox info is 2D array/list
        if np.array(bbox_list).ndim == 2:
            roi = []
            for bb in bbox_list:
                roi.append(self._crop(img, bb))
        # single object in scene
        else:
            roi = self._crop(img, bbox_list)

        if self.transform is not None:
            if isinstance(roi, list):
                tensor_imgs = []
                for region in roi:
                    tensor_img = self.transform(region)
                    tensor_imgs.append(tensor_img)
                tensor_imgs = torch.stack(tensor_imgs)
                return {
                    "frame_name": frame_name,
                    "imgs": tensor_imgs
                }
            else:
                tensor_img = self.transform(roi).unsqueeze(dim=0)
                print(tensor_img.shape)
                return {
                    "frame_name": frame_name,
                    "imgs": tensor_img
                }
        return {
                    "frame_name": frame_name,
                    "imgs": roi
                }
    

class ActionDataset(data.Dataset):

    def __init__(self,
                 data: FrameDataSet,
                 annotations: dict):
        self.data = data
        self.frame_to_class = {}
        for frame in annotations["frames"]:
            frame_name = frame["file_name"]
            activity_class = frame["activity_class"]
            self.frame_to_class[frame_name] = activity_class

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        cropped_frame_dict = self.data[idx]
        frame_name = cropped_frame_dict.get("frame_name")
        imgs = cropped_frame_dict.get("imgs")
        activity = self.frame_to_class[frame_name]

        feature_vectors = []
        for img in imgs:
            img = img / 255
            features = feature_extractor(img.unsqueeze(0))
            feature_vectors.append(features)
        feature_vectors = pad_and_stack_vectors(feature_vectors)

        return feature_vectors.flatten(), activity
    

class ActionDatasetOld(data.Dataset):

    def __init__(self,
                 data: FrameDataSet,
                 annotations: dict):
        self.data = data
        self.frame_to_class = {}
        for frame in annotations["frames"]:
            frame_name = frame["file_name"]
            activity_class = frame["activity_class"]
            self.frame_to_class[frame_name] = activity_class

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        cropped_frame_dict = self.data[idx]
        frame_name = cropped_frame_dict.get("frame_name")
        imgs = cropped_frame_dict.get("imgs")
        activity = self.frame_to_class[frame_name]
        return imgs, activity
    
################ NOT USED UNTIL HERE ################


# if name_only parameter is True, the path to the image will be ignored
# only the file name, e.g. "00000.jpg" will be saved
def parse_coco(ann_dir, file_name="annotations.json", name_only=True):
    N = 5 # for the length of an image name
    data = [] # list of dicts

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
            #color_file = img.split("/")[1]
            color_file = img
        else:
            color_file = os.path.join(prefix, img.split("/")[0], img.split("/")[1])
        d = {
            "file_name": color_file,
            "bb_info": [],
            "object_classes": []
        }
        for annotation in annotations:
            # image id as a string
            iid = str(annotation.get("image_id"))
            end = iid.split("_")[-1]
            #length = len(iid)
            length = len(end)
            #img_id = (N - length) * "0" + iid
            img_id = iid[:-length] + (N - length) * "0" +  end

            if "/" in img:
                img_name = img.split("/")[1].split(".")[0] # example: "images/00001.jpg" -> "00000"
            else:
                img_name = img.split(".")[0]
            if img_id == img_name:
                d["bb_info"].append(annotation.get("bbox"))
                d["object_classes"].append(annotation.get("category_id"))
        data.append(d)

    return data


if __name__ == "__main__":
    ann_dir = os.path.join("activity recognition", "mock data")
    # data from coco file
    data_ = parse_coco(ann_dir)

    # test the logic
    root_dir = os.path.join("activity recognition", "mock data", "images")

    # get 4 examples
    for i in range(0, 50, 10):
        # TODO: this is ugly
        file_name = data_[i].get("file_name").split("/")[-1]
        bb_info = data_[i].get("bb_info")
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        #ds = FrameDataSet(root_dir,
        #                  bb_info={"rabbit.jpg": [[120, 90, 300, 400], [200, 90, 300, 400]]},
        #                  transform=transform)
        ds = FrameDataSet(root_dir,
                        bb_info={file_name: bb_info},
                        transform=transform)
        
        import matplotlib.pyplot as plt
        img = torch.transpose(ds[0][2], 0, 2).transpose(0, 1)
        print(img.shape)
        plt.imshow(img)
        #plt.imsave("rabbit_cropped.jpg", img.numpy())
        plt.show()
