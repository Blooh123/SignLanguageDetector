import os
import cv2

# ================= CONFIG =================
DATA_DIR = './data'

# 26 letters A-Z
number_of_classes = 26

# 6 variations per letter:
# 1-3 Right hand
# 4-6 Left hand
variations_per_class = 6

# Images per variation
dataset_size = 100

letters = [chr(ord('A') + i) for i in range(number_of_classes)]

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# ================= CAMERA =================
cap = cv2.VideoCapture(0)

for class_idx in range(number_of_classes):

    letter = letters[class_idx]

    # Create main folder for the letter
    class_dir = os.path.join(DATA_DIR, letter)
    os.makedirs(class_dir, exist_ok=True)

    print(f"\nCollecting data for letter {letter}")

    # ================= 6 VARIATIONS =================
    for variation in range(1, variations_per_class + 1):

        # Determine hand type
        if variation <= 3:
            hand_type = "RIGHT"
            variation_num = variation
        else:
            hand_type = "LEFT"
            variation_num = variation - 3

        variation_folder = os.path.join(
            class_dir,
            f"{hand_type}_VARIATION_{variation_num}"
        )
        os.makedirs(variation_folder, exist_ok=True)

        print(
            f"Collecting {hand_type} hand variation {variation_num} "
            f"for letter {letter}"
        )

        # ================= READY SCREEN =================
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            display_frame = frame.copy()

            cv2.putText(
                display_frame,
                f'Letter: {letter}',
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                3,
                cv2.LINE_AA
            )

            cv2.putText(
                display_frame,
                f'Hand: {hand_type}',
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                3,
                cv2.LINE_AA
            )

            cv2.putText(
                display_frame,
                f'Variation: {variation_num}/3',
                (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                3,
                cv2.LINE_AA
            )

            cv2.putText(
                display_frame,
                'Press Q when ready',
                (50, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3,
                cv2.LINE_AA
            )

            cv2.imshow('Data Collection', display_frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # ================= CAPTURE IMAGES =================
        counter = 0

        while counter < dataset_size:
            ret, frame = cap.read()
            if not ret:
                continue

            display_frame = frame.copy()

            # On-screen identifiers
            cv2.putText(
                display_frame,
                f'{letter} | {hand_type} | Var {variation_num}',
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 0),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                display_frame,
                f'Image: {counter + 1}/{dataset_size}',
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            cv2.imshow('Data Collection', display_frame)

            # Save original frame
            cv2.imwrite(
                os.path.join(
                    variation_folder,
                    f'{counter}.jpg'
                ),
                frame
            )

            counter += 1

            # Small delay for better variation
            cv2.waitKey(50)

# ================= CLEANUP =================
cap.release()
cv2.destroyAllWindows()

print("\nDataset collection complete!")