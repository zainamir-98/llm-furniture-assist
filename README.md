# LLM Furniture Assist: A digital assistant for furniture assembly powered by ChatGPT 

Details:
- An object detection model, based on YOLOv8, identifies objects in the scene. The model outputs the bounding box coordinates of each object.
- Google's Mediapipe algorithm is used to identify landmarks of the hands. Bounding box coordinates from YOLOv8 are used to extract the hands only before application of the Mediapipe algorithm.
- Activity recognition is carried out by an ensemble model by computing the weighted average of two models: A sequential LSTM classifier, and a non-sequential artificial neural network. The sequential model is fed with 3 consecutive image frames, while the non-sequential model is provided with the bounding box coordinates from the object detection model and the hand landmarks from Mediapipe. The model also calculates an additional feature, called the distance vector, representing the Euclidean distance of each hand from the centre of the hands to different objects in the scene.
- The recognised activity is provided to ChatGPT (which has been fine-tuned on a text explaining the assembly instructions in detail) through an API call. The LLM sends back the next instruction to assemble the furniture. The user can ask for further details or clarifications through a separate text terminal, allowing for highly personalised assembly instructions.
