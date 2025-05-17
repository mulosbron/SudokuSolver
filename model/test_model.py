import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_all = np.concatenate((X_train, X_test), axis=0)
y_all = np.concatenate((y_train, y_test), axis=0)

valid_indices = y_all != 0
X_all = X_all[valid_indices]
y_all = y_all[valid_indices]

X_all = X_all.astype('float32') / 255.0
IMG_SIZE = 100
X_all_resized = np.array([cv2.resize(img, (IMG_SIZE, IMG_SIZE)) for img in X_all])
X_all_resized = X_all_resized.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = tf.keras.models.load_model('sudoku/model.keras')

predictions = model.predict(X_all_resized)
predicted_indices = np.argmax(predictions, axis=1)

actual_classes = y_all.astype(int)
predicted_classes = np.array(predicted_indices).astype(int)

print(classification_report(actual_classes, predicted_classes))
print(confusion_matrix(actual_classes, predicted_classes))

wrong_indices = np.where(predicted_classes != actual_classes)[0]
print(len(wrong_indices))

num_examples = 5
for i in range(min(num_examples, len(wrong_indices))):
    idx = wrong_indices[i]
    img = X_all[idx]
    plt.imshow(img, cmap='gray')
    plt.title(f"Actual: {actual_classes[idx]}, Prediction: {predicted_classes[idx]}")
    plt.show()
