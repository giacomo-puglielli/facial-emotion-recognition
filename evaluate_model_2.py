# evaluate_model.py

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from sklearn.metrics import f1_score, classification_report

# 1) Percorsi
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
WEBCAM_DIR   = os.path.join(BASE_DIR, 'webcam')
MODEL_DIR    = os.path.join(WEBCAM_DIR, 'models')
PREPRO_DIR   = os.path.join(BASE_DIR, 'data', 'preprocessed')

ARCH_PATH    = os.path.join(MODEL_DIR, 'model.json')
WEIGHTS_PATH = os.path.join(MODEL_DIR, 'model_3.h5')
X_TEST_PATH  = os.path.join(PREPRO_DIR, 'x_test.npy')
Y_TEST_PATH  = os.path.join(PREPRO_DIR, 'y_test.npy')

# 2) Carica test set pre-processato
print("Loading test data…")
x_test = np.load(X_TEST_PATH)     # shape (N,48,48,1)
y_test = np.load(Y_TEST_PATH)     # one-hot shape (N,7)

# 3) Normalizza esattamente come in training
x_test = x_test.astype('float32') / 255.0

# 4) Ricostruisci il modello da JSON e carica i pesi
print("Loading model architecture and weights…")
with open(ARCH_PATH, 'r') as f:
    model = model_from_json(f.read())
model.load_weights(WEIGHTS_PATH)

# 5) Compila (optimizer ininfluente per valutazione, serve solo loss/metriche)
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 6) Valuta sul test set
print("Evaluating on test set…")
test_loss, test_acc = model.evaluate(
    x_test, y_test,
    batch_size=64,
    verbose=1
)
print(f"\nTest Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}\n")

# 7) Calcola F1-score
#    y_true dalle etichette one-hot
y_true = np.argmax(y_test, axis=1)

#    y_pred dalle predizioni
y_pred_proba = model.predict(x_test, batch_size=64, verbose=1)
y_pred       = np.argmax(y_pred_proba, axis=1)

f1_macro = f1_score(y_true, y_pred, average='macro')
f1_micro = f1_score(y_true, y_pred, average='micro')
print(f"F1-score (macro): {f1_macro:.4f}")
print(f"F1-score (micro): {f1_micro:.4f}\n")

# 8) Report dettagliato per classe
print("Classification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=['angry','disgust','fear','happy','sad','surprise','neutral']
))

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns  # opzionale, per una heatmap più carina

# … dopo aver calcolato y_true e y_pred …

# 1) Costruisci la confusion matrix
cm = confusion_matrix(y_true, y_pred)

# 2) Plot (versione vanilla con matplotlib)
plt.figure(figsize=(8,8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
classes = ['angry','disgust','fear','happy','sad','surprise','neutral']
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

# numeri nella matrice
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'confusion_matrix.png'))
plt.show()

