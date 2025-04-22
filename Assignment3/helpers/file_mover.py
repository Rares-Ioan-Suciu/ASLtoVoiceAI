import os
import shutil

# used this script to move some files from one dataset to another

DATA_DIR = '../capture_data' # change with personal directory
dataset1 = "/home/rares655/Sem_4/AI/Assignment3/dataset1/asl_alphabet_train/asl_alphabet_train"
dataset2 = "/home/rares655/Sem_4/AI/Assignment3/dataset2"
dataset3 = "/home/rares655/Sem_4/AI/Assignment3/dataset2/asl_dataset"

dataset_2_and_3 = [dataset2, dataset3]

print("Moving data from dataset1 to capture_data...")

for dir_ in os.listdir(dataset1):

    if dir_.isalpha() and len(dir_) == 1 and dir_.isupper():
        src_dir = os.path.join(dataset1, dir_)
        dst_dir = os.path.join(DATA_DIR, dir_)

        try:
            if not os.listdir(src_dir):
                os.rmdir(src_dir)
        except FileNotFoundError:
            pass

        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        try:
            for img in os.listdir(src_dir):
                shutil.move(os.path.join(src_dir, img), os.path.join(dst_dir, img))
        except FileNotFoundError:
            print("File not found! Skipping...")

print("All data moved from dataset1!")

for dataset in dataset_2_and_3:
    print(f"Moving data from {dataset} to capture_data...")
    for dir_ in os.listdir(dataset):
        if (dir_.isalpha() and len(dir_) == 1 and dir_.isupper()) or dir_.isdigit():
            if dir_.isalpha():
                dir_ = dir_.upper()

            src_dir = os.path.join(dataset, dir_)
            dst_dir = os.path.join(DATA_DIR, dir_)

            try:
                if not os.listdir(src_dir):
                    os.rmdir(src_dir)
            except FileNotFoundError:
                pass

            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)

            try:
                for img in os.listdir(src_dir):
                    shutil.move(os.path.join(src_dir, img), os.path.join(dst_dir, img))
            except FileNotFoundError:
                print("File not found! Skipping...")

    print(f"All data moved from {dataset}!")

do_symbol_dir = os.path.join(DATA_DIR, 'symbols')

if not os.path.exists(do_symbol_dir):
    os.makedirs(do_symbol_dir)
    for dir_ in os.listdir(DATA_DIR):
        current_dir = os.path.join(DATA_DIR, dir_)
        if dir_ == 'symbols' or not os.path.isdir(current_dir):
            continue
        for img in os.listdir(current_dir):
            src_path = os.path.join(current_dir, img)
            dst_path = os.path.join(do_symbol_dir, f"{dir_}_{img}")
            shutil.copy(src_path, dst_path)
            break