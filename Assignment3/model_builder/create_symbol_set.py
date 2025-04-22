import os
import cv2
import shutil

import numpy as np


#script used to capture data from ypu live cam, use your own hand to do the symbol
DATA_DIR = '../capture_data' # change with personal directory
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 0
dataset_size = 1500

def capture_images():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Press a letter or number key to start data capture. Press ESC to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read from camera.")
            break

        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('frame', 1280, 720)

        frame =  255 * np.zeros(shape=[720, 1280, 3], dtype=np.uint8)
        cv2.putText(frame, 'Ready? Press your letter/number!', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        key = cv2.waitKey(0)
        if key == 27:  # ESC
            print("Exiting...")
            break

        if key in range(ord('a'), ord('z') + 1):
            key -= 32

        if chr(key).isalnum():
            label = chr(key)
            class_dir = os.path.join(DATA_DIR, label)
            os.makedirs(class_dir, exist_ok=True)

            print(f"Collecting data for class '{label}'...")
            counter = 0
            existing_files = os.listdir(class_dir)
            existing_indices = [int(f.split('.')[0]) for f in existing_files if
                                f.endswith('.jpg') and f.split('.')[0].isdigit()]
            start_index = max(existing_indices, default=-1) + 1

            while counter < dataset_size:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to read from camera.")
                    break

                cv2.imshow('frame', frame)
                cv2.waitKey(25)
                img_path = os.path.join(class_dir, f'{start_index + counter}.jpg')
                cv2.imwrite(img_path, frame)
                counter += 1

            print(f"Finished capturing {dataset_size} images for '{label}'.")

    cap.release()
    cv2.destroyAllWindows()




if __name__ == "__main__":
    capture_images()


