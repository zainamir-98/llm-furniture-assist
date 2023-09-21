import os
from glob import glob
from PIL import Image
from ultralytics import YOLO
from IPython import display
display.clear_output()
import ultralytics
ultralytics.checks()


def object_detection(img_dir, model_dir, conf=0.4, save=False):

    directory = img_dir
    
    # Get all image paths in the directory
    image_paths = glob(os.path.join(directory, "*.jpg"))
    
    # Extract file names from image paths
    file_names = [os.path.basename(path) for path in image_paths]
    
    # Open the images
    images = [Image.open(path) for path in image_paths]
    
    # Make predictions
    model = YOLO(model_dir)
    
    # Perform object detection on the images
    test_results = model(images, conf=conf, save=save, project="yolo_predictions")
    
    results_list = []

    for i, result in enumerate(test_results):
        boxes = result.boxes
        box_list = boxes.xywh.tolist()
        bbox_list = [[int(coord) for coord in box] for box in box_list]
    
        object_classes = [int(cl) for cl in boxes.cls.tolist()]
    
        result_dict = {
            "file_name": file_names[i],
            "bb_info": bbox_list,
            "object_classes": object_classes
        }
    
        results_list.append(result_dict)

    return results_list
    