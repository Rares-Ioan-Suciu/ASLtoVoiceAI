import cv2
import pickle
import mediapipe as mp
from collections import deque, Counter


model_dict = pickle.load(open('../models/model.p', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)


cap = cv2.VideoCapture(0)
predictions_queue = deque(maxlen=18)

symbols = [chr(letter) for letter in range(ord('A'), ord('Z') + 1)] + ['1', '5']

def detect_symbol(frame, results, Height, Width):
    data_aux = []
    x_, y_ = [], []

    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

    for hand_landmarks in results.multi_hand_landmarks:
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            x_.append(x)
            y_.append(y)

        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append(x - min(x_))
            data_aux.append(y - min(y_))

    if len(data_aux) == 42:
        proba = model.predict_proba([data_aux])
        confidence = max(proba[0])
        pred = model.predict([data_aux])[0]

        print(pred + ': ' + str(confidence))
        if confidence > 0.25 and pred in symbols:
            predictions_queue.append(pred)
        else:
            predictions_queue.append('NULL')

    most_common_pred = Counter(predictions_queue).most_common(1)[0][0]

    x1 = int(min(x_) * Width) - 10
    y1 = int(min(y_) * Height) - 10
    x2 = int(max(x_) * Width) + 10
    y2 = int(max(y_) * Height) + 10

    return most_common_pred, (x1, y1, x2, y2)

def inference():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        Height, Width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            predicted_char, (x1, y1, x2, y2) = detect_symbol(frame, results, Height, Width)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, f"Sign: {predicted_char}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3, cv2.LINE_AA)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    inference()
