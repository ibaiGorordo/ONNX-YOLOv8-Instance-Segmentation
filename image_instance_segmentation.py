import cv2
from imread_from_url import imread_from_url

from yoloseg import YOLOSeg

# Initialize YOLOv5 Instance Segmentator
model_path = "models/yolov8m-seg.onnx"
yoloseg = YOLOSeg(model_path, conf_thres=0.5, iou_thres=0.3)

# Read image
img_url = "https://upload.wikimedia.org/wikipedia/commons/e/e6/Giraffes_at_west_midlands_safari_park.jpg"
img = imread_from_url(img_url)

# Detect Objects
boxes, scores, class_ids, masks = yoloseg(img)

# Draw detections
combined_img = yoloseg.draw_masks(img)
cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
cv2.imshow("Detected Objects", combined_img)
cv2.imwrite("doc/img/detected_objects.jpg", combined_img)
cv2.waitKey(0)
