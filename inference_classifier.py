import pickle
import cv2
import mediapipe as mp
import numpy as np

# ================= LOAD MODEL =================
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

# ================= LABEL MAP =================
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
    5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N',
    14: 'O', 15: 'P', 16: 'Q', 17: 'R',
    18: 'S', 19: 'T', 20: 'U', 21: 'V',
    22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}

# Letters requiring motion
motion_letters = ['J', 'Z']

# ================= CAMERA =================
cap = cv2.VideoCapture(0)

# Optional resolution boost
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ================= MEDIAPIPE =================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ================= VARIABLES =================
trajectory = []
last_prediction = ""
stable_prediction = ""
counter = 0


# ================= MOTION DETECTION =================
def detect_motion(trajectory):
    """
    Detect J and Z movement.
    """
    if len(trajectory) < 10:
        return None

    xs = [p[0] for p in trajectory]
    ys = [p[1] for p in trajectory]

    dy = max(ys) - min(ys)

    direction_changes = 0
    for i in range(1, len(xs) - 1):
        if (xs[i] - xs[i - 1]) * (xs[i + 1] - xs[i]) < 0:
            direction_changes += 1

    # Z motion
    if direction_changes >= 2:
        return "Z"

    # J motion
    elif dy > 0.15:
        return "J"

    return None


# ================= MAIN LOOP =================
while True:

    ret, frame = cap.read()
    if not ret:
        break

    # Flip for mirror effect
    frame = cv2.flip(frame, 1)

    H, W, _ = frame.shape

    data_aux = []
    x_ = []
    y_ = []

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    predicted_character = "No hand"

    if results.multi_hand_landmarks:

        hand_landmarks = results.multi_hand_landmarks[0]

        # ================= DRAW HAND =================
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        # ================= FEATURE EXTRACTION =================
        for lm in hand_landmarks.landmark:
            x_.append(lm.x)
            y_.append(lm.y)

        x_min = min(x_)
        y_min = min(y_)

        x_range = max(x_) - x_min
        y_range = max(y_) - y_min

        if x_range != 0 and y_range != 0:

            for lm in hand_landmarks.landmark:
                data_aux.append((lm.x - x_min) / x_range)
                data_aux.append((lm.y - y_min) / y_range)

            # ================= BOUNDING BOX =================
            x1 = int(x_min * W) - 20
            y1 = int(y_min * H) - 20
            x2 = int(max(x_) * W) + 20
            y2 = int(max(y_) * H) + 20

            # ================= MODEL PREDICTION =================
            prediction = model.predict([np.asarray(data_aux)])

            # Your model already returns the actual label (like 'A', 'B', etc.)
            predicted_character = str(prediction[0])

            # If prediction is unknown
            if predicted_character not in labels_dict.values():
                predicted_character = "Unknown"

            # ================= CONFIDENCE SCORE =================
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(
                    [np.asarray(data_aux)]
                )[0]

                confidence = np.max(probabilities) * 100
            else:
                confidence = 0

            # ================= MOTION TRACKING =================
            if predicted_character in motion_letters:

                x_tip = hand_landmarks.landmark[8].x
                y_tip = hand_landmarks.landmark[8].y

                trajectory.append((x_tip, y_tip))

                if len(trajectory) > 20:
                    trajectory.pop(0)

                motion_prediction = detect_motion(trajectory)

                if motion_prediction:
                    predicted_character = motion_prediction

            else:
                trajectory.clear()

            # ================= STABILITY FILTER =================
            if predicted_character == last_prediction:
                counter += 1
            else:
                counter = 0

            if counter > 5:
                stable_prediction = predicted_character

            last_prediction = predicted_character

            display_character = (
                stable_prediction
                if stable_prediction
                else predicted_character
            )

            # ================= DRAW UI =================
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 0, 0),
                3
            )

            # Prediction label
            cv2.putText(
                frame,
                f'Letter: {display_character}',
                (x1, y1 - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (0, 0, 0),
                3,
                cv2.LINE_AA
            )

            # Confidence
            cv2.putText(
                frame,
                f'Confidence: {confidence:.2f}%',
                (x1, y1 - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 100, 255),
                2,
                cv2.LINE_AA
            )

            # ================= TRAJECTORY =================
            if predicted_character in motion_letters:
                for i in range(1, len(trajectory)):
                    cv2.line(
                        frame,
                        (
                            int(trajectory[i - 1][0] * W),
                            int(trajectory[i - 1][1] * H)
                        ),
                        (
                            int(trajectory[i][0] * W),
                            int(trajectory[i][1] * H)
                        ),
                        (0, 255, 0),
                        3
                    )

    else:
        trajectory.clear()

    # ================= GAME-STYLE HUD =================
    cv2.rectangle(frame, (10, 10), (350, 80), (255, 255, 255), -1)
    cv2.rectangle(frame, (10, 10), (350, 80), (0, 0, 0), 3)

    cv2.putText(
        frame,
        "SIGNQUEST DETECTOR",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        "ESC to Exit",
        (20, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    # ================= DISPLAY =================
    cv2.imshow("Sign Language Detector", frame)

    # ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break


# ================= CLEANUP =================
cap.release()
cv2.destroyAllWindows()