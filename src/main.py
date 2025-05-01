import cv2
import time
from window_utils import set_window_icon
from stickman_drawer import draw_stickman
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh

pose = mp_pose.Pose(static_image_mode=False)
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Window settings
WINDOW_WIDTH = 960
WINDOW_HEIGHT = 720
WINDOW_NAME = "I-am-Stickman"

show_cam = False
flip_cam = True

# Path to the icon
icon_path = os.path.join(os.getcwd(), 'assets', 'icon.ico')

cap = cv2.VideoCapture(0)
start_time = time.time()

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)
set_window_icon(WINDOW_NAME, icon_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if flip_cam:
        frame = cv2.flip(frame, 1)

    t = time.time() - start_time
    height, width = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results_pose = pose.process(frame_rgb)
    results_face = face_mesh.process(frame_rgb)

    background = frame if show_cam else None

    if results_pose.pose_landmarks:
        face_landmarks = results_face.multi_face_landmarks[0].landmark if results_face.multi_face_landmarks else None
        stickman = draw_stickman(frame, results_pose.pose_landmarks.landmark, width, height, t, face_landmarks=face_landmarks, background=background)
    else:
        stickman = frame if show_cam else np.ones_like(frame) * 255

    win_width = cv2.getWindowImageRect(WINDOW_NAME)[2]
    win_height = cv2.getWindowImageRect(WINDOW_NAME)[3]
    resized = cv2.resize(stickman, (win_width, win_height), interpolation=cv2.INTER_LINEAR)

    cv2.imshow(WINDOW_NAME, resized)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break
    elif key == ord('q'):
        show_cam = not show_cam
    elif key == ord('f'):
        flip_cam = not flip_cam

cap.release()
cv2.destroyAllWindows()
