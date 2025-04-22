import os
import pickle
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = '../test_folder'
sorted_dirs = sorted(os.listdir(DATA_DIR))

data = []
labels = []

def process_image(img_path, dir_, count):
    data_aux = []

    x_ = []
    y_ = []

    img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(img_rgb)
    count += 1
    if count % 1000 == 0:
        print("Processed: ", count, " images for class: ", dir_, "")
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
        if data_aux.__len__() == 42:
            data.append(data_aux)
            labels.append(dir_)

def save_data():
    f = open('../models/data_night.pickle', 'wb')
    pickle.dump({'data': data, 'labels': labels}, f)
    f.close()

def process_data():
    for dir_ in sorted_dirs:
        count = 0
        print("Start processing for: ", dir_, "")
        for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
            process_image(img_path, dir_, count)

if __name__ == "__main__":
    process_data()