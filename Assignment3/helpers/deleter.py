import os

big_dir = "../test_folder"

symbols = "../symbols"
for dir_ in os.listdir(big_dir):
    if dir_ in ['D', 'E', 'F', 'G']:
        count = 4400
        for img in os.listdir(os.path.join(big_dir, dir_)):
            if count == 0:
                break
            os.remove(os.path.join(big_dir, dir_, img))
            count -= 1

for img_path in os.listdir(symbols):
    if img_path[0].isupper():
        os.rename(os.path.join(symbols, img_path), os.path.join(symbols, img_path[0] + '.jpg'))