import cv2
import pickle
import mediapipe as mp
from collections import deque, Counter

from gtts import gTTS
import pygame
import tempfile

def do_voiceover(sentence):
    print("Speaking:", sentence)
    tts = gTTS(text=sentence, lang='en')

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        pygame.mixer.init()
        pygame.mixer.music.load(fp.name)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

csv = '/prediction.csv'

def load_model(path='../models/model.p'):
    model_dict = pickle.load(open(path, 'rb'))
    return model_dict['model']

def initialize_hand_detector():
    return mp.solutions.hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

def process_frame(frame, hands, model, predictions_queue):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    data_aux = []
    x_, y_ = [], []
    pred_char = 'NULL'
    Height, Width, _ = frame.shape

    if results.multi_hand_landmarks:
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

            with open(csv, 'a') as f:
                f.write(f'{pred},{confidence}\n')

            if confidence > 0.15 and pred in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ15':
                predictions_queue.append(pred)
            else:
                predictions_queue.append('NULL')

            pred_char = Counter(predictions_queue).most_common(1)[0][0]

        x1 = int(min(x_) * Width) - 10
        y1 = int(min(y_) * Height) - 10
        x2 = int(max(x_) * Width) - 10
        y2 = int(max(y_) * Height) - 10
        box_coords = (x1, y1, x2, y2)

        return pred_char, results.multi_hand_landmarks, box_coords
    return pred_char, None, None

def update_sentence(sentence, predicted_char):
    if predicted_char == '1':
        return sentence + ' '
    elif predicted_char == '5':
        return sentence[:-1]
    elif predicted_char != 'NULL':
        return sentence + predicted_char
    return sentence

def draw_ui(frame, landmarks, box_coords, predicted_char, sentence, mp_hands, mp_drawing, mp_drawing_styles):
    if landmarks:
        for hand_landmarks in landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
        if box_coords:
            x1, y1, x2, y2 = box_coords
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, f"Sign: {predicted_char}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3, cv2.LINE_AA)

    cv2.putText(frame, f"Sentence: {sentence}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    return frame

def main():
    model = load_model()
    hands = initialize_hand_detector()
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(0)
    predictions_queue = deque(maxlen=20)

    sentence = ''
    no_hand_counter = 0
    counter_limit = 30

    stable_pred = 'NULL'
    stable_count = 0
    cooldown_counter = 0
    cooldown_time = 50

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        predicted_char, landmarks, box_coords = process_frame(frame, hands, model, predictions_queue)

        if landmarks:
            no_hand_counter = 0

            if predicted_char == stable_pred:
                stable_count += 1
            else:
                stable_pred = predicted_char
                stable_count = 1

            if stable_count >= 45 and cooldown_counter == 0:

                sentence = update_sentence(sentence, predicted_char)
                cooldown_counter = cooldown_time
        else:
            no_hand_counter += 1
            predictions_queue.clear()
            if no_hand_counter >= counter_limit and sentence != '':
                do_voiceover(sentence)
                sentence = ''

        if cooldown_counter > 0:
            cooldown_counter -= 1

        frame = draw_ui(frame, landmarks, box_coords, predicted_char, sentence,
                        mp_hands, mp_drawing, mp_drawing_styles)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
