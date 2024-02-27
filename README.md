# Meniscus-Tear-Diagnosis-using-YoloV8-and-Mask-RCNN

Abstract
This study’s main goal was to compare two Deep Learning models—YOLOv8 and Mask RCNN—for the purpose of diagnosing damaged and healthy meniscus from knee MRI images. The purpose of the study was to eval-
uate their performance in relation to training time, regression scores, and mean average precision (MAP).
0.0.2 Purpose
Comparing YOLOv8 with Mask RCNN’s efficacy in accurately diagnos- ing meniscus problems with MRI was the main objective of this study. One of the specific goals was to assess their performance measures, which included Regression and MAP scores, while taking training time into ac- count.
0.0.3 Methodology
Dataset for this research included 685 total MRI images, out of which 456 for train , 130 for valid and 99 for test dataset. Multiple Pre-Processing steps were performed on this dataset like image resizing , data augmenta- tion , data saturation and Converting it from dicom to jpeg format. Dataset was Labelled and annotated with the help of Radiologist and data scientist team. Labelled data was than trained on Mask RCNN and YOLOV8 model for 15 Epochs each. Trained model was evaluated using metrics such as MAP, Regression score and recall.
 0.0.4 Results
When it came to performance, YOLOv8 outperformed Mask RCNN, at- taining a MAP value of 0.89 as opposed to Mask RCNN’s 0.360. With an average regression score of 0.977 on the test dataset, Mask RCNN showed improved meniscus tear identification despite the lower MAP value. This dissimilarity could be attributed to the architectural differences between Mask RCNN and YOLOv8, namely the latter’s creation of exact masks for each class while the former uses bounding boxes. The increased MAP value of YOLOv8 may also be explained by the Non-Maximum Suppres- sion (NMS) post-processing phase.
0.0.5 Conclusion
This study offers valuable insights into the strengths and weaknesses of YOLOv8 and Mask RCNN in the context of diagnosing meniscus tears. The results of this research provide healthcare professionals with valuable information when considering the implementation of these deep learning models in diagnostic applications.
