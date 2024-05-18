import cv2
import numpy as np
from tkinter import Tk, filedialog

def get_the_images():
    # For allowing user to select single or batch of images.
    root = Tk()
    root.withdraw()
    
    # Use filedialog to open the file selection dialog
    path_of_file = filedialog.askopenfilenames(
        title="Select Images",
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png")]
    )
    return list(path_of_file)

def yolo(img_path):
    # Load YOLO model
    yoloo = cv2.dnn.readNet("yolov3_training_last.weights", "yolov3_training.cfg")
    if yoloo.empty():
        print("Error: Unable to load model.")
        return None
    
    # Load class names
    classes = []
    with open("classes.txt", "r") as f:
        classes = [l.strip() for l in f.readlines()]
    
    # Define the plant names based on class IDs
    plant_names = {
        0: "Apple Scab",
        1: "Apple Black Rot",
        2: "Apple Cedar Rust",
        3: "Healthy Apple Leaf"
    }
    
    # Get the output layer names
    default_layers = yoloo.getLayerNames()
    output_layers = [default_layers[i - 1] for i in yoloo.getUnconnectedOutLayers()]
    
    # Load the image
    img = cv2.imread(img_path)
    if img is None:
        print("Error: Unable to load image", img_path)
        return None
    
    # Resize image and prepare it for YOLO
    img_resized = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, _ = img_resized.shape
    blob = cv2.dnn.blobFromImage(img_resized, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    
    # Set the input blob for the network
    yoloo.setInput(blob)
    
    # Forward pass to get the predictions
    out = yoloo.forward(output_layers)
    
    # Initialize variables for detection
    highest_confidence = 0
    detected_id = None
    
    # Process detections
    for i in out:
        for detection in i:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # Filter detections by confidence
            if confidence > 0.3:
                if confidence > highest_confidence:
                    highest_confidence = confidence
                    detected_id = class_id

    
    # Return the name of the plant based on the detected class ID
    if detected_id is not None:
        plant_name = plant_names.get(detected_id, "Unknown")
        return plant_name
    else:
        return None

def calculate_accuracy(predicted_labels, actual_labels):
    # Check the len of both labels
    if len(predicted_labels) != len(actual_labels):
        print("The lengths of labels lists do not match.")
        return None
    
    # (convert to lowercase and strip any whitespace)
    predicted_normalized = [l.lower().strip() for l in predicted_labels]
    actual_normalized = [l.lower().strip() for l in actual_labels]
    
    # Calculate the number of correct predictions
    fianl_predictions = sum(1 for p, a in zip(predicted_normalized, actual_normalized) if p == a)
    
    # Calculate accuracy
    total = len(predicted_labels)
    accuracy = (fianl_predictions / total) * 100
    
    return accuracy


def main():
    # Allow the user to select multiple image files
    img_paths = get_the_images()
    
    # List to store predicted labels
    predicted_labels = []
    
    
    # Process each image
    for img_path in img_paths:
        # Perform object detection
        plant_name = yolo(img_path)
        
        # Store the predicted label
        if plant_name is not None:
            predicted_labels.append(plant_name)
        else:
            predicted_labels.append("Unknown")

    actual_labels = []
    labels={'0':'Apple Scab','1':'Apple Black Rot','2':'Apple Cedar Rust','3':'Healthy Apple Leaf'}

    acc=input('Do you want the accuracy score: \n 1:Yes \n 2: No \n')
    if acc=='1':
        for class_name in range(len(img_paths)):
            actual=input(f"class number 0: Apple Scab,\n class number 1: Apple Black Rot,\n class number 2: Apple Cedar Rust,\n class number 3: Healthy Apple Leaf \n Enter class number for image {class_name+1}: ")
            actual_value=labels[actual]
            actual_labels.append(actual_value)
         # Calculate accuracy
        accuracy = calculate_accuracy(predicted_labels, actual_labels)
        
    # Output the predicted labels and accuracy
    for x, plant_name in enumerate(predicted_labels):
        print(f"Image {x + 1}: Predicted Plant Name: {plant_name}")
    if acc=='1':
        if accuracy is not None:
            print(f"Accuracy: {accuracy:.2f}%")

# Run the main function
if __name__ == "__main__":
    main()
