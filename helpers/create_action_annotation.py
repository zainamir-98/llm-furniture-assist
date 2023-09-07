import os
import json


def create_image_label_mapping(root_dir):
    mapping = {}

    for subdir, dirs, files in os.walk(root_dir):
        print(subdir)
        if "all_images" in subdir:
            continue
        for file in files:
            file_path = os.path.join(subdir, file)
            action_class = os.path.basename(subdir)
            mapping[file] = action_class

    return mapping


root_dir = "./activity recognition/mock data/all_in_one"

image_label_mapping = create_image_label_mapping(root_dir)

# Write the mapping to a JSON file
json_file_path = "./helpers/action_annotations.json"
with open(json_file_path, "a") as json_file:
    json_file.write(json.dumps(image_label_mapping))
