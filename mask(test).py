# For testing
import torch
from torchvision import models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
import numpy as np
import mediapipe as mp
import time, sys
from win32com.client import Dispatch
speak = Dispatch("SAPI.SpVoice").Speak
device = "cuda" if torch.cuda.is_available() else "cpu"
num_classes = 6
net = models.detection.fasterrcnn_resnet50_fpn()
in_features = net.roi_heads.box_predictor.cls_score.in_features
net.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
net.load_state_dict(torch.load("pmodel.pth"))
net.eval()
net = net.to(device)
last_detected = False
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
camera = cv2.VideoCapture(1)
camera2 = cv2.VideoCapture(2)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while camera.isOpened() and camera2.isOpened():
        success, frame = camera.read()
        if not success:
            break
        success2, frame2 = camera2.read()
        if not success2:
            break
        g=1
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = np.array(img, dtype=np.float32) / 255
        img = torch.from_numpy(img)
        img = torch.permute(img, (2, 0, 1))
        img = torch.unsqueeze(img, 0)
        img = img.to(device)
        # print(img.shape)
        h = net(img)
        # print(h)
        o_boxes = h[0]["boxes"]
        o_labels = h[0]["labels"]
        o_scores = h[0]["scores"]
        cv2.rectangle(frame, (0, 75), (166, 270), (218, 218, 218), 2)
        cv2.rectangle(frame, (66, 8), (195, 256), (218, 218, 218), 2)
        cv2.rectangle(frame, (136, 54), (305, 255), (218, 218, 218), 2)
        cv2.rectangle(frame, (274, 61), (564, 254), (218, 218, 218), 2)
        cv2.rectangle(frame, (533, 81), (639, 229), (218, 218, 218), 2)
        cnt = 0
        for box, label, score in zip(o_boxes, o_labels, o_scores):
            if score < 0.7:
                g=0
                continue
            x1, y1, x2, y2 = map(int, box)
            if label==1:
                if y1<70 or x2>170 or y2>280:
                    color = (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label_names = ["obj1", "obj2", "obj3", "obj4", "obj5"]
                    cv2.putText(frame, label_names[label - 1], (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    g=0
                else:
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label_names = ["obj1", "obj2", "obj3", "obj4", "obj5"]
                    cv2.putText(frame, label_names[label - 1], (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            if label==2:
                if x1<50 or y1<0 or x2>205 or y2>270:
                    color = (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label_names = ["obj1", "obj2", "obj3", "obj4", "obj5"]
                    cv2.putText(frame, label_names[label - 1], (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    g=0
                else:
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label_names = ["obj1", "obj2", "obj3", "obj4", "obj5"]
                    cv2.putText(frame, label_names[label - 1], (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            if label==3:
                if x1<126 or y1<44 or x2>305 or y2>255:
                    color = (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label_names = ["obj1", "obj2", "obj3", "obj4", "obj5"]
                    cv2.putText(frame, label_names[label - 1], (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    g=0
                else:
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label_names = ["obj1", "obj2", "obj3", "obj4", "obj5"]
                    cv2.putText(frame, label_names[label - 1], (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            if label==4:
                if x1<264 or y1<51 or x2>584 or y2>264:
                    color = (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label_names = ["obj1", "obj2", "obj3", "obj4", "obj5"]
                    cv2.putText(frame, label_names[label - 1], (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    g=0
                else:
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label_names = ["obj1", "obj2", "obj3", "obj4", "obj5"]
                    cv2.putText(frame, label_names[label - 1], (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            if label==5:
                if x1<523 or y1<71 or x2>649 or y2>239:
                    color = (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label_names = ["obj1", "obj2", "obj3", "obj4", "obj5"]
                    cv2.putText(frame, label_names[label - 1], (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    g=0
                else:
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label_names = ["obj1", "obj2", "obj3", "obj4", "obj5"]
                    cv2.putText(frame, label_names[label - 1], (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cnt += 1
        if cnt!=5:  g=0
        frame2.flags.writeable = False
        imageRGB = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        results = pose.process(imageRGB)
        frame2.flags.writeable = True
        if results.pose_landmarks:
            cv2.putText(frame2, 'human detected', (0, 25), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 5)
            if last_detected==False:
                print("\x1b[33m[INFO] detected human\x1b[0m")
                last_detected = True
        else:
            cv2.putText(frame2, 'human not detected', (0, 25), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 5)
            if last_detected==True:
                print("\x1b[33m[INFO] human leave\x1b[0m")
                if g==0:
                    print("\x1b[31m[WARN] item lost or not placed well\x1b[0m")
                    speak("detected item lost or not placed well")
                else:
                    print("\x1b[32m[OK] item placed well\x1b[0m")
                last_detected = False
        cv2.imshow("frame", frame)
        cv2.imshow("frame2", frame2)
        key_code = cv2.waitKey(3)
        if key_code in [27, ord('q')]:
            break
camera.release()
camera2.release()
cv2.destroyAllWindows()