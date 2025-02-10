Dataset:
The dataset was chosen from Kaggle. This particular dataset includes 4 different diseases of apple leaves. The dataset size for each classification is around 1800 to 2000.

Models and Tools used:
Several machine learning algorithms are implemented on this dataset for identifying leaf diseases. But we are using the YOLO model including RNN algorithms which run in the backend. 
The LabelImg tool facilitated the manual labeling process during model training, enabling the annotation of images with class labels. This annotation process ensured that the machine learning model could learn to recognize and classify objects accurately, enhancing its performance in tasks such as object detection or disease identification. Further Open CV tool for Testing.

Labelling

Using LabelImg an open-source graphical image annotation tool used for labeling objects in images.
It supports various annotation formats such as PascalVOC,YOLO, and COCO. We will be using YOLO for labeling.


Training         
                                          
We split the data into 70:30 ratio, where 70% of the data is for training and 30% of the data is for testing.
Input: Preprocessed images of individual leaf specimens.
Output: Classification label corresponding to the type of leaf.

