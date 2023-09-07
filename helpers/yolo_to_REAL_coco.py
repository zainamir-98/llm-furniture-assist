import os
import json

# Source folder containing the annotations
cwd = os.getcwd()
annotations_folder = os.path.join(cwd, 'helpers', 'annotations')
print(annotations_folder)

# Destination folder to save the modified images and JSON files
output_folder = 'REAL_annotations'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

output_file = os.path.join(output_folder, "annotations_new.json")
new_annotations = {"images": [], "annotations": []}
# Iterate over the annotations folders
for subfolder in os.listdir(annotations_folder):
    annotations_folder_path = os.path.join(annotations_folder, subfolder)

    # Get the images folder path
    images_folder_path = os.path.join(annotations_folder_path, 'images')

    # Get the path of the JSON file
    json_file_path = os.path.join(annotations_folder_path, 'annotations.json')

    # Read the JSON file
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    # Access the 'annotations' field
    annotations = data['annotations']

    # Create the output subfolder within the YOLO_annotations folder
    #output_subfolder = os.path.join(output_folder, subfolder)
    #os.makedirs(output_subfolder, exist_ok=True)

    # Create the path for the output text file within the subfolder
    #output_file = os.path.join(output_subfolder, "annotations_new.json")

    #renamed_images = {"images": []}
    # Iterate over the images
    for image in data['images']:
        # Get the image ID and file name
        image_id = image['id']
        file_name = image['file_name']
        image_width = image['width']
        image_height = image['height']

        # Create the text file name
        new_image_name = f"{subfolder}_{os.path.splitext(os.path.basename(file_name))[0]}.jpg"

        new_annotations['images'].append({"file_name": new_image_name, "id": f"{subfolder}_{image_id}"})

    ann = []
    for dictionary in annotations:
        dictionary["image_id"] = str(subfolder) + "_" + str(dictionary["image_id"])
        ann.append(dictionary)

    new_annotations["annotations"].extend(ann)

with open(output_file, "w") as f:
    f.write(json.dumps(new_annotations))
                

print('COCO annotations in YOLO format created and saved in the YOLO_annotations folder.')