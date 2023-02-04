import os
import cv2
from models.YOLOv8 import YOLOv8

if __name__ == '__main__':
    model_path = "./EduBinYolov8_3_2_2023.onnx"
    class_names = ["bottle", "milo", "redbull"]
    # Initialize YOLOv8 object detector
    yolov8_detector = YOLOv8(model_path,  # path to onnx model 
                            class_names= class_names, # class names
                            conf_thres=0.5, 
                            iou_thres=0.5
                            )

    path = r"C:\Users\PC\Documents\HCMUT K21\EduBin\DEMO_DAY\CODE\Yolov8onnx\test\images\\"
    for file in os.listdir(path=path):

        img = cv2.imread(path + file)
        # Detect Objects
        yolov8_detector(img)

        # Draw detections
        combined_img = yolov8_detector.draw_detections(img)

        # Resize image
        out_img = cv2.resize(combined_img, ( 720,720))
        cv2.imshow("Output", out_img)
        cv2.waitKey(300)