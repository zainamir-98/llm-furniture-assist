import torch
import os
from torch.utils import data
from torchvision.transforms import transforms
import torchvision.models as models
import cv2 as cv
from typing import List
from PIL import Image


"""
Class to load pre-defined number of frames in a sequence
"""
class Sequence:

    def __init__(self,
                 root_dir: str,
                 activity: str,
                 frames: List[str] = None,
                 walk: bool = False):
        self.frames = []
        self.frame_names = []

        if frames is not None:
            for frame in frames:
                path = os.path.join(root_dir, frame)
                img = cv.imread(path)
                self.frames.append(img)
                self.frame_names.append(frame)

        elif walk:
            files = [file for file in os.listdir(root_dir)\
                     if os.path.isfile(os.path.join(root_dir, file))]

            for file in files:
                path = os.path.join(root_dir, file)
                img = cv.imread(path)
                self.frames.append(img)
                self.frame_names.append(file)

        else:
            raise Exception("Provide either 'walk' or 'frames' attribute")

        self.activity = activity
        self._index = 0

    
    def __str__(self):
        return f"Frame Sequence containing {len(self.frames)} frames"
    

    def __iter__(self):
        self._index = 0
        return self
    

    def __next__(self):
        if self._index < len(self.frames):
            fname, f = self.frame_names[self._index], self.frames[self._index]
            self._index += 1
            return fname, f
        else:
            raise StopIteration
        
    def __len__(self):
        return len(self.frames)
    


class SequenceLoader(data.Dataset):
    def __init__(self,
                 data: List[Sequence],
                 feature_extractor: callable,
                 transform: transforms.Compose = None,
                 seq_length: int = 3,
                 test=False):
        self.data = data
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((128, 128))
        ])
        else:
            self.transform = transform
        self.feature_extractor = feature_extractor
        self.sequence_length = seq_length
        self.test = test


    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):
        # Get frames from Sequence and crop + transform
        seq = []
        activity = self.data[idx].activity
        for frame_name, frame in self.data[idx]:
            tensor_img = self.transform(frame)
            seq.append(tensor_img)

        # Padding for sequences with less frames than definded sequence length
        while len(seq) < self.sequence_length:
            seq.append(torch.zeros_like(tensor_img))

        tensor_seq = torch.stack(seq)

        # Extract features
        feature_vectors = []
        for frame in tensor_seq:
            frame = frame / 255
            features = self.feature_extractor(frame.unsqueeze(0))
            feature_vectors.append(features.squeeze(0))
        feature_vectors = torch.stack(feature_vectors)

        if self.test:
            return feature_vectors
        return feature_vectors, activity