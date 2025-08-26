import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_stylews = mp.solutions.drawing_styles

FINGER_TIPS= [8,12,16,20]
FINGER_PIPS= [6,10,14,18]
THUMB_TIP=4
THUMB_IP=3

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)as hands:
    
    while True:
        ret,frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame,1)

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        text_y = 30

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,results.multi_handedness):
                hand_label = handedness.classification[0].label
                lm = hand_landmarks.landmark

                cont = 0
                for tip_id, pip_id in zip(FINGER_TIPS,FINGER_PIPS):
                    if lm[tip_id].y < lm[pip_id].y:
                        cont += 1

                   
                if hand_label == "Right":
                    if lm[THUMB_TIP].x < lm[THUMB_IP].x:
                        cont += 1
                else:
                    if lm[THUMB_TIP].x > lm[THUMB_IP].x:
                        cont += 1

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_stylews.get_default_hand_landmarks_style(),
                    mp_stylews.get_default_hand_connections_style()
                )

                cv2.putText(frame, f"{hand_label} hand: {cont}",(10, text_y),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
                text_y += 40

        cv2.imshow("Gercek Zamanli Yuz Algilama",frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()