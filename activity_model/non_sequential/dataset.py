import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from torchvision.transforms import transforms
import cv2 as cv
from typing import List
import json
import cv2
import chumpy as ch
import mediapipe as mp


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

# 2 = left hand, 3 = right hand
HAND_CATEGORY = [2, 3]

# Table plate not relevant
IRRELEVANT_CLASSES = [1]

class BBoxLandmarks:
    def __init__(self, x, y, width, height, image):
        image_height, image_width, _ = image.shape
        if x - width/2 > 0:
            self.x1 = int(x - width/2)
        else:
            self.x1 = int(0)
        if y - height/2 > 0:
            self.y1 = int(y - height/2)
        else:
            self.y1 = int(0)
        if x + width/2 < image_width:
            self.x2 = int(x + width/2)
        else:
            self.x2 = image_width
        if y + height/2 < image_height:
            self.y2 = int(y + height/2)
        else:
            self.y2 = image_height
        self.image_width = image_width
        self.image_height = image_height

    def applyExtension(self, val):
        if self.x1 - val >= 0:
            self.x1 -= val
        else:
            self.x1 = 0

        if self.y1 - val >= 0:
            self.y1 -= val
        else:
            self.y1 = 0

        if self.x2 + val <= self.image_width:
            self.x2 += val
        else:
            self.x2 = self.image_width
            
        if self.y2 + val <= self.image_height:
            self.y2 += val
        else:
            self.y2 = self.image_height

    def getRegion(self, image):
        return image[self.y1:self.y2, self.x1:self.x2]


class Landmark:
    def __init__(self, image_width, image_height, x, y, z):
        self.x = x
        self.y = y #round(y * image_height)
        self.z = z

    def output_dict(self):
        return {"x" : self.x, "y" : self.y}
    
    def output_vector(self):
        return [self.x, self.y]
    

class LandmarkDetector:
    def __init__(self, static_image_mode=True, 
                max_num_hands=1,
                min_detection_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.out = None

        self.mp_hands = mp.solutions.hands

    def detect(self, bbox : BBoxLandmarks, image):
        output = []

        bbox.applyExtension(20)

        for t in range(3):
            with self.mp_hands.Hands(static_image_mode=self.static_image_mode,
                                max_num_hands=self.max_num_hands,
                                min_detection_confidence=self.min_detection_confidence) as hands:
                image_height, image_width, _ = image.shape
                image_cropped = bbox.getRegion(image)
                self.annotated_image = image_cropped.copy()

                # Convert the BGR image to RGB before processing.
                # print(image_cropped.shape)
                results = None
                if image_cropped.shape[0] != 0 and image_cropped.shape[1] != 0:
                    results = hands.process(cv2.cvtColor(image_cropped, cv2.COLOR_BGR2RGB))

                if results is None or results.multi_hand_landmarks is None:
                    # Increase bounding box size until detection is successful
                    bbox.applyExtension(20)
                    continue
                else:
                    # Append landmarks to output vector
                    for hand_landmarks in results.multi_hand_landmarks:
                        for l in hand_landmarks.landmark:
                            l_obj = Landmark(image_width, image_height, l.x, l.y, l.z)
                            output.append(l_obj.output_vector())

                    break
            
        if output == []:
            for i in range(21):
                output.append([0, 0])
            # print("From detector")
            # print(output)

        assert len(output) == 21
        self.out = output        
        assert self.out is not None


class ActionDataset(data.Dataset):

    def __init__(self,
                 root_dir: str,
                 data: List[dict],
                 actions: dict,
                 feature_extractor: callable,
                 pad: callable,
                 transform: transforms.Compose = None,
                 distance_weight: int = 5,
                 max_length_distance_vector: int = 16,
                 test=False):
        """
        data must be in format: [
            {
                "file_name": str,    # name of the frame file
                "bb_info": List[list],
                "object_classes": List[int]
            }
        ]
        actions must be in format: {
            "file_name": action_class (str) # should be read from folder directly
        }
        distance weight is the weighting factor that the normalized distance between the
        key object (hand) and the other objects are multiplied with in order to increase
        the impact of the distance vector during the learning process
        """
        self.root_dir = root_dir
        self.data = [d for d in data
                     # make sure that the image exists, reason:
                     # sometimes we manually delete bad images, but they are still annotated
                     # and therefore present in the annotation file
                     if os.path.exists(os.path.join(root_dir, d.get("file_name")))]
        if not test:
            self.actions = {file_name: ACTIVITIES[action] for file_name, action in actions.items()
                            if os.path.exists(os.path.join(root_dir, file_name))}
        else:
            self.actions = None

        self.feature_extractor = feature_extractor
        self.padding = pad

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((128, 128))
        ])
        else:
            self.transform = transform

        self.distance_weight = distance_weight
        self.max_length_distance_vector = max_length_distance_vector
        self.test = test


    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):
        # get frame information from index
        frame_info = self.data[idx]
        file_name = frame_info.get("file_name")
        bb_infos = frame_info.get("bb_info")
        object_classes = frame_info.get("object_classes")

        # drop irrelevant classes
        self._drop_irrelevant(bb_infos, object_classes)

        # load frame
        path = os.path.join(self.root_dir, file_name)
        frame = cv.imread(path)

        # calculate distances between hand and objects
        # then normalize and weight the distances
        distance_vector = self._calc_distances(bb_infos, object_classes)
        distance_vector = self._normalize_and_weight(distance_vector)
        if distance_vector.shape[0] < self.max_length_distance_vector:
            num_padding = self.max_length_distance_vector - len(distance_vector)
            distance_vector = torch.cat((distance_vector, torch.zeros(num_padding)))
        else:
            raise Exception("distance vector is longer then max defined distance vector")

        # crop and transform
        regions = []
        for region in bb_infos:
            if region[2] > 10 and region[3] > 10:
                cropped = self._crop(frame, region)
                cropped_tensor = self.transform(cropped)
                regions.append(cropped_tensor)
        tensor_frame = torch.stack(regions)

        # extract features
        feature_vectors = []
        for region in tensor_frame:
            region = region / 255
            features = self.feature_extractor(region.unsqueeze(0),
                                              reduce_dim=True)
            feature_vectors.append(features)

        # padding if necessary
        feature_vectors = self.padding(feature_vectors)
        # add distance vector to feature vector
        feature_vectors = feature_vectors.flatten()
        feature_vectors = torch.cat((feature_vectors, distance_vector))

        # --- Landmarks can be added to feature vector here --- #

        landmarks_vector = self._get_hand_landmarks(bb_infos, object_classes, frame)
        assert len(landmarks_vector) == 42
        landmarks_vector = torch.tensor(landmarks_vector)
        feature_vectors = torch.cat((feature_vectors, landmarks_vector))
        # print(len(feature_vectors))

        # ----------------------------------------------------- #

        if self.test:
            return feature_vectors
        action = self.actions.get(file_name)
        return feature_vectors, action#, (frame, bb_infos)


    def _crop(self, img: np.ndarray, bbox: List[int]) -> torch.Tensor:
        if isinstance(bbox[0], int):
            roi = img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
        else:
            roi = img[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
        return roi
    

    def _get_hand_landmarks(self, bb_infos: List[list], object_classes: List[list], frame):
        right_hand_idx = None
        left_hand_idx = None
        r_landmarks = []
        l_landmarks = []
        image = frame
        bbox_r = None
        bbox_l = None
        detector = LandmarkDetector()

        # Get indices of right hand and left hand classes
        for index, object_class in enumerate(object_classes):
            if object_class is HAND_CATEGORY[0]:
                left_hand_idx = index
            elif object_class is HAND_CATEGORY[1]:
                right_hand_idx = index

        if right_hand_idx is not None or left_hand_idx is not None:
            for index, bbox in enumerate(bb_infos):
                x, y, width, height = bbox

                if index is left_hand_idx:
                    bbox_l = BBoxLandmarks(x, y, width, height, image)
                    detector.detect(bbox_l, image)
                    l_landmarks = detector.out

                elif index is right_hand_idx:
                    bbox_r = BBoxLandmarks(x, y, width, height, image)
                    detector.detect(bbox_r, image)
                    r_landmarks = detector.out
                else:
                    continue  

        if r_landmarks == [] or len(r_landmarks) < 21:
            r_landmarks = []
            for i in range(21):
                r_landmarks.append([0, 0])
            # print(r_landmarks)
        if l_landmarks == [] or len(l_landmarks) < 21:
            l_landmarks = []
            for i in range(21):
                l_landmarks.append([0, 0])

        assert len(r_landmarks) == 21
        assert len(l_landmarks) == 21
        landmarks = r_landmarks + l_landmarks

        # Convert to 1D list

        temp = []
        for i in range(21):
            for j in range(2):
                temp.append(landmarks[i][j])
        landmarks = temp
        assert len(landmarks) == 42
        return landmarks

    def _calc_distances(self, bb_infos: List[list], object_classes: List[list]):
        hand_indices = []
        for index, object_class in enumerate(object_classes):
            if object_class in HAND_CATEGORY:
                hand_indices.append(index)
        
        centre_points = {
            "hand": [],
            "not_hand": []
        }
        for index, bbox in enumerate(bb_infos):
            # calculate centre point of bounding box
            x, y, width, height = bbox
            center_x = x + (width / 2)
            center_y = y + (height / 2)

            if index in hand_indices:
                centre_points["hand"].append((center_x, center_y))
            else:
                centre_points["not_hand"].append((center_x, center_y))
        
        distances = []
        for centre in centre_points.get("not_hand"):
            x, y = centre
            for centre_hand in centre_points.get("hand"):
                x_hand, y_hand = centre_hand
                # calculate euclidean distance
                distance = np.sqrt(
                    (x - x_hand)**2 + (y - y_hand)**2
                )
                distances.append(distance)

        return distances
    

    def _normalize_and_weight(self, distance_vector: List) -> torch.Tensor:
        distance_vector = torch.tensor(distance_vector)
        normalized = distance_vector / torch.max(distance_vector)
        # invert the distance before weighting, because shorter distance are relevant
        # for the learning process
        inverted = 1 - normalized
        return self.distance_weight * inverted

    
    def _drop_irrelevant(self, bb_infos, object_classes):
        for index, obj_class in enumerate(object_classes):
            if obj_class in IRRELEVANT_CLASSES:
                bb_infos.pop(index)
                object_classes.pop(index)
    

def read_action_annotations(file):
    with open(file, "r") as jsonf:
        data = json.load(jsonf)
    return data


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
