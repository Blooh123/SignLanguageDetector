import pickle
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ================= LOAD DATASET =================
data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.array(
    data_dict['data'],
    dtype=np.float32
)

labels = np.asarray(
    data_dict['labels']
)

# ================= DATA SUMMARY =================
print("Dataset loaded successfully!")
print(f"Total samples: {len(data)}")
print(f"Total labels: {len(labels)}")
print(f"Unique classes: {sorted(set(labels))}")

# ================= TRAIN / TEST SPLIT =================
x_train, x_test, y_train, y_test = train_test_split(
    data,
    labels,
    test_size=0.2,
    shuffle=True,
    stratify=labels,
    random_state=42
)

print("\nTraining samples:", len(x_train))
print("Testing samples:", len(x_test))

# ================= MODEL CONFIGURATION =================
model = RandomForestClassifier(
    n_estimators=300,        # More trees for better accuracy
    max_depth=None,          # Full depth
    random_state=42,
    n_jobs=-1                # Faster training
)

# ================= TRAIN MODEL =================
print("\nTraining classifier...")
model.fit(x_train, y_train)

# ================= PREDICTIONS =================
y_predict = model.predict(x_test)

# ================= ACCURACY =================
score = accuracy_score(
    y_test,
    y_predict
)

print(
    '\n{:.2f}% of samples were classified correctly!'.format(
        score * 100
    )
)

# ================= DETAILED REPORT =================
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_predict
    )
)

# ================= CONFUSION MATRIX =================
print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_predict
    )
)

# ================= FEATURE IMPORTANCE =================
print("\nModel training complete.")

# ================= SAVE MODEL =================
with open('model.p', 'wb') as f:
    pickle.dump(
        {
            'model': model
        },
        f
    )

print("\nModel saved successfully as model.p")