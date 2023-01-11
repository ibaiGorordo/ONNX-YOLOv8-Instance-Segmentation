import cv2
from cap_from_youtube import cap_from_youtube

from yoloseg import YOLOSeg

# # Initialize video
# cap = cv2.VideoCapture("input.mp4")

videoUrl = 'https://youtu.be/-bhSSispEcg'
cap = cap_from_youtube(videoUrl, resolution='1080p')
start_time = 22  # skip first {start_time} seconds
cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * cap.get(cv2.CAP_PROP_FPS))

# Initialize YOLOv5 Instance Segmentator
model_path = "models/yolov8m-seg.onnx"
yoloseg = YOLOSeg(model_path, conf_thres=0.5, iou_thres=0.3)

# fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# out = cv2.VideoWriter("output.mp4", fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
frame_countdown = 3
while cap.isOpened():

    # Press key q to stop
    if cv2.waitKey(1) == ord('q'):
        break

    # Read frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Update object localizer
    boxes, scores, class_ids, masks = yoloseg(frame)

    combined_img = yoloseg.draw_masks(frame, mask_alpha=0.4)
    # out.write(combined_img)
    cv2.imshow("Detected Objects", combined_img)

cap.release()
# out.release()