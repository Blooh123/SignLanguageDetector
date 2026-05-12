import os
import pickle
import mediapipe as mp
import cv2

# ================= MEDIAPIPE SETUP =================
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3
)

# ================= DATASET DIRECTORY =================
DATA_DIR = './data'

data = []
labels = []

# ================= PROCESS DATASET =================
# Expected structure:
# data/
#   A/
#      RIGHT_VARIATION_1/
#      RIGHT_VARIATION_2/
#      RIGHT_VARIATION_3/
#      LEFT_VARIATION_1/
#      LEFT_VARIATION_2/
#      LEFT_VARIATION_3/

for letter_dir in os.listdir(DATA_DIR):

    letter_path = os.path.join(DATA_DIR, letter_dir)

    if not os.path.isdir(letter_path):
        continue

    print(f"Processing letter: {letter_dir}")

    # Loop through all variations
    for variation_dir in os.listdir(letter_path):

        variation_path = os.path.join(letter_path, variation_dir)

        if not os.path.isdir(variation_path):
            continue

        print(f"   Variation: {variation_dir}")

        for img_name in os.listdir(variation_path):

            img_path = os.path.join(variation_path, img_name)

            # ================= READ IMAGE =================
            img = cv2.imread(img_path)

            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # ================= HAND DETECTION =================
            results = hands.process(img_rgb)

            if not results.multi_hand_landmarks:
                continue

            hand_landmarks = results.multi_hand_landmarks[0]

            data_aux = []
            x_ = []
            y_ = []

            # ================= LANDMARK EXTRACTION =================
            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            x_min = min(x_)
            y_min = min(y_)

            x_range = max(x_) - x_min
            y_range = max(y_) - y_min

            # Prevent division by zero
            if x_range == 0 or y_range == 0:
                continue

            # ================= NORMALIZATION =================
            for lm in hand_landmarks.landmark:
                normalized_x = (lm.x - x_min) / x_range
                normalized_y = (lm.y - y_min) / y_range

                data_aux.append(normalized_x)
                data_aux.append(normalized_y)

            # ================= VALIDATION =================
            if len(data_aux) == 42:
                data.append(data_aux)

                # Use only the letter as label
                labels.append(letter_dir)

# ================= SAVE DATASET =================
with open('data.pickle', 'wb') as f:
    pickle.dump(
        {
            'data': data,
            'labels': labels
        },
        f
    )

# ================= SUMMARY =================
print("\nDataset creation complete!")
print(f"Total samples collected: {len(data)}")
print(f"Total labels collected: {len(labels)}")