import pickle
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from collections import Counter


data_dict = pickle.load(open('../models/data_night.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])


def get_confusion(y_test, y_pred, model):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap='Greens', values_format='d', xticks_rotation='vertical')
    plt.title("Confusion Matrix")
    plt.show()


def save_model(model, path='../models/model.p'):
    with open(path, 'wb') as f:
        pickle.dump({'model': model}, f)


def print_report(model, score, x_test, y_test):
    print(f'{score * 100:.2f}% of samples were classified correctly!\n')

    label_counts = Counter(y_test)
    print("Test Set Label Distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  Label '{label}': {count} samples")

    y_pred = model.predict(x_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    get_confusion(y_test, y_pred, model)


def train():
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

    model = RandomForestClassifier(class_weight='balanced', random_state=42)
    model.fit(x_train, y_train)

    score = accuracy_score(y_test, model.predict(x_test))

    save_model(model)
    print_report(model, score, x_test, y_test)


if __name__ == "__main__":
    train()
