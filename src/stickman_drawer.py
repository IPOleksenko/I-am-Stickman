import cv2
import numpy as np
from utils import get_rainbow_color
import mediapipe as mp

# Initialize MediaPipe Pose and FaceMesh
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh

def draw_stickman(frame, landmarks, width, height, t, face_landmarks=None, background=None):
    canvas = background.copy() if background is not None else np.ones_like(frame) * 255
    center = np.array([width // 2, height // 2])
    color = get_rainbow_color(t)

    def get_point(index):
        try:
            lm = landmarks[index]
            return np.array([lm.x * width, lm.y * height], dtype=int)
        except:
            return None

    def safe_point(index, fallback_indices=(), default=center + np.array([0, 40])):
        pt = get_point(index)
        if pt is not None:
            return pt
        fallback_pts = [get_point(i) for i in fallback_indices]
        fallback_pts = [p for p in fallback_pts if p is not None]
        if len(fallback_pts) == 2:
            return ((fallback_pts[0] + fallback_pts[1]) // 2).astype(int)
        elif len(fallback_pts) == 1:
            return fallback_pts[0]
        return default

    def draw_line(a, b):
        cv2.line(canvas, tuple(a), tuple(b), color, 4)

    # Head
    nose = safe_point(mp_pose.PoseLandmark.NOSE)
    left_eye = safe_point(mp_pose.PoseLandmark.LEFT_EYE)
    right_eye = safe_point(mp_pose.PoseLandmark.RIGHT_EYE)

    head_radius = int(abs(left_eye[0] - right_eye[0]) * 2) if left_eye is not None and right_eye is not None else 20
    cv2.circle(canvas, tuple(nose), head_radius, color, 4)

    # Eyes (FaceMesh)
    if face_landmarks:
        for idx in [468, 473]:
            try:
                pt = face_landmarks[idx]
                cx = int(pt.x * width)
                cy = int(pt.y * height)
                cv2.circle(canvas, (cx, cy), 5, color, -1)
            except:
                pass

        def draw_mouth(indices):
            points = []
            for idx in indices:
                try:
                    pt = face_landmarks[idx]
                    x = int(pt.x * width)
                    y = int(pt.y * height)
                    points.append((x, y))
                except:
                    continue
            if len(points) >= 2:
                cv2.polylines(canvas, [np.array(points, np.int32).reshape((-1, 1, 2))], isClosed=True, color=color, thickness=2)

        draw_mouth([61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308])
        draw_mouth([78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308])

    # Shoulders, hips, torso
    l_shoulder = safe_point(mp_pose.PoseLandmark.LEFT_SHOULDER)
    r_shoulder = safe_point(mp_pose.PoseLandmark.RIGHT_SHOULDER)
    l_hip = safe_point(mp_pose.PoseLandmark.LEFT_HIP)
    r_hip = safe_point(mp_pose.PoseLandmark.RIGHT_HIP)

    draw_line(l_shoulder, r_shoulder)
    draw_line(l_hip, r_hip)
    draw_line((l_shoulder + r_shoulder) // 2, (l_hip + r_hip) // 2)

    # Arms
    for side in ['LEFT', 'RIGHT']:
        shoulder_idx = getattr(mp_pose.PoseLandmark, f'{side}_SHOULDER')
        elbow_idx = getattr(mp_pose.PoseLandmark, f'{side}_ELBOW')
        wrist_idx = getattr(mp_pose.PoseLandmark, f'{side}_WRIST')

        shoulder = safe_point(shoulder_idx)
        elbow = safe_point(elbow_idx, fallback_indices=[shoulder_idx, wrist_idx])
        wrist = safe_point(wrist_idx, fallback_indices=[elbow_idx])

        draw_line(shoulder, elbow)
        draw_line(elbow, wrist)

    # Legs
    for side in ['LEFT', 'RIGHT']:
        hip_idx = getattr(mp_pose.PoseLandmark, f'{side}_HIP')
        knee_idx = getattr(mp_pose.PoseLandmark, f'{side}_KNEE')
        ankle_idx = getattr(mp_pose.PoseLandmark, f'{side}_ANKLE')

        hip = safe_point(hip_idx)
        knee = safe_point(knee_idx, fallback_indices=[hip_idx, ankle_idx])
        ankle = safe_point(ankle_idx, fallback_indices=[knee_idx])

        draw_line(hip, knee)
        draw_line(knee, ankle)

    return canvas
