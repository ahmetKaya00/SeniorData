import cv2
import mediapipe as mp
import math
import time

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_stylews = mp.solutions.drawing_styles

def euclidean_distance(p1,p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def eye_open_ratio(landmarks,eye_indices, image_w,image_h):
    outer = landmarks[eye_indices[0]]
    inner = landmarks[eye_indices[1]]
    upper = landmarks[eye_indices[2]]
    lower = landmarks[eye_indices[3]]

    outer_pt = (int(outer.x * image_w), int(outer.y * image_h))
    inner_pt = (int(inner.x * image_w), int(inner.y * image_h))
    upper_pt = (int(upper.x * image_w), int(upper.y * image_h))
    lower_pt = (int(lower.x * image_w), int(lower.y * image_h))

    horiz = euclidean_distance(outer_pt,inner_pt)
    vert = euclidean_distance(upper_pt,lower_pt)

    ratio = (vert / horiz) if horiz > 0 else 0.0
    return ratio, (outer_pt, inner_pt, upper_pt,lower_pt)

cap = cv2.VideoCapture(0)

BLINK_RATIO_THRESEHOLD = 0.24
MIN_CONSEC_FRAMES = 2

blink_count = 0
frames_below_thresh = 0
p_time = 0.0

with mp_face_mesh.FaceMesh(
    max_num_faces =1,
    refine_landmarks=True,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
) as face_mesh:
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h,w = frame.shape[:2]
        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_idx = [33,133,159,145]
                right_idx = [362,263,386,374]

                left_ratio, left_pts = eye_open_ratio(face_landmarks.landmark, left_idx,w,h)
                right_ratio, right_pts = eye_open_ratio(face_landmarks.landmark, right_idx,w,h)
                avg_ratio = (left_ratio + right_ratio) / 2.0

                for pt in left_pts + right_pts:
                    cv2.circle(frame,pt,2,(0,255,255),-1)
                
                if avg_ratio < BLINK_RATIO_THRESEHOLD:
                    frames_below_thresh += 1
                else:
                    if frames_below_thresh >= MIN_CONSEC_FRAMES:
                        blink_count += 1
                    frames_below_thresh = 0
                
                cv2.putText(frame, f"Ratio: {avg_ratio:.2f}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,0),2)
                cv2.putText(frame, f"Blinks: {blink_count}",(10,70),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,200,0),2)

        c_time = time.time()
        fps = 1.0 / (c_time - p_time) if p_time else 0.0
        p_time = c_time
        cv2.putText(frame, f"FPS: {fps:.1f}",(10,110),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,200,255),2)

        cv2.imshow("Gercek Zamanli Yuz Algilama",frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
