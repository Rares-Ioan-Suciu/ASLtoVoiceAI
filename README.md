#Hand Gesture to Speech Translator

A computer vision-based Python application that translates real-time hand gestures (ASL letters) into English text and then speaks the resulting sentence aloud using a text-to-speech engine.

---
## 📸 [Demo](https://youtu.be/gl8E00bKbqU)
https://youtu.be/gl8E00bKbqU
---

## Features

- Real-time ASL alphabet recognition using MediaPipe.
- Machine learning model trained to classify 26 English letters and control gestures (`1 = Space`, `5 = Backspace`).
- Noise filtering when no hand is detected.
- Text-to-speech support via Google Text-to-Speech (gTTS) and `pygame`.
- Automatically speaks the full sentence when no hand is detected for a few seconds.
---

## Requirements
Install the required Python packages:

```bash
pip install opencv-python mediapipe scikit-learn gTTS pygame
```
Or use:
```bash
pip install -r requirements.txt
```


#  How It Works

### Hand Detection
MediaPipe detects **hand landmarks** (21 key points).
### Feature Extraction
Extracts **relative coordinates** of landmarks for model input.
### Prediction
A trained **classifier** predicts the letter or gesture.
### Sentence Formation
Stabilized letters form a sentence, with special gestures for:
- **1** → Space  
- **5** → Backspace
### Speech Output
When **no hand** is detected for a short period, the **sentence is spoken aloud**.

---

## Training the Model (Optional)

If you want to train your own model:
1. **Collect data** using your webcam and save **landmark data** for each letter.
2. **Train a classifier** (e.g. RandomForest) on the dataset.
3. **Save it** as `model.p` using `pickle`.
---

## 🎮 Controls & Gestures

| Gesture | Action        |
|---------|---------------|
| A–Z     | Add letter    |
| 1       | Add space     |
| 5       | Backspace     |
| No hand | Speak sentence |
