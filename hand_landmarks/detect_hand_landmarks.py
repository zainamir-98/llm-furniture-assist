import cv2
import chumpy as ch
import mediapipe as mp
import matplotlib.pyplot as plt
from typing import List
import numpy as np
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 2 = left hand, 3 = right hand
HAND_CATEGORY = [2, 3]


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
        print("Before: ")
        print(x, y, width, height)
        print("After correction: ")
        print(self.x1, self.x2, self.y1, self.y2)
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
        self.annotated_image = None

        self.mp_hands = mp.solutions.hands
        # self.mp_drawing = mp.solutions.drawing_utils
        # self.mp_drawing_styles = mp.solutions.drawing_styles

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

                        mp_drawing.draw_landmarks(
                            self.annotated_image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())

                    break
            
        if output == []:
            for i in range(21):
                output.append([0, 0])
            # print("From detector")
            # print(output)

        assert len(output) == 21
        self.out = output        
        assert self.out is not None

def get_hand_landmarks(bb_infos: List[list], object_classes: List[list], frame):
        right_hand_idx = None
        left_hand_idx = None
        r_landmarks = []
        l_landmarks = []
        image = frame
        bbox_r = None
        bbox_l = None
        annotated_l = None
        annotated_r = None
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
                    annotated_l = cv2.cvtColor(detector.annotated_image, cv2.COLOR_BGR2RGB)

                elif index is right_hand_idx:
                    bbox_r = BBoxLandmarks(x, y, width, height, image)
                    detector.detect(bbox_r, image)
                    r_landmarks = detector.out
                    annotated_r = cv2.cvtColor(detector.annotated_image, cv2.COLOR_BGR2RGB)
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
        
        # print("RH")
        # print(r_landmarks)
        # print("LH")
        # print(l_landmarks)
        # print("H")
        landmarks = r_landmarks + l_landmarks
        # print(len(landmarks))
        # print(landmarks)
        # print(len(landmarks))
        
        # Convert to 1D list

        temp = []
        for i in range(21):
            for j in range(2):
                temp.append(landmarks[i][j])
        landmarks = temp
        # print(len(landmarks))
        # print("H2")
        # print(landmarks)
        # print(len(landmarks))
        assert len(landmarks) == 42
        return landmarks, annotated_r, annotated_l

def detect(img, data : List[dict]):
    bb_infos = data.get("bb_info")
    object_classes = data.get("object_classes")
    landmarks_vector, annotated_r, annotated_l = get_hand_landmarks(bb_infos, object_classes, img)
    print("Length of landmarks vector:", len(landmarks_vector))
    plt.subplot(1, 2, 1)
    if not annotated_l is None:
        plt.imshow(annotated_l)
    plt.subplot(1, 2, 2)
    if not annotated_r is None:
        plt.imshow(annotated_r) 