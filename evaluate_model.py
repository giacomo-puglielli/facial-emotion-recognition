import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json

# 1) Percorsi
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
WEBCAM_DIR = os.path.join(BASE_DIR, 'webcam')
MODEL_DIR    = os.path.join(WEBCAM_DIR, 'models')
PREPRO_DIR   = os.path.join(BASE_DIR, 'data', 'preprocessed')

ARCH_PATH    = os.path.join(MODEL_DIR, 'model.json')
WEIGHTS_PATH = os.path.join(MODEL_DIR, 'model_3.h5')
X_TEST_PATH  = os.path.join(PREPRO_DIR, 'x_test.npy')
Y_TEST_PATH  = os.path.join(PREPRO_DIR, 'y_test.npy')

# 2) Carica test set pre‐processato
print("Loading test data…")
x_test = np.load(X_TEST_PATH)
y_test = np.load(Y_TEST_PATH)

# 3) Normalizza esattamente come in training
x_test = x_test.astype('float32') / 255.0

# 4) Ricostruisci il modello da JSON e carica i pesi
print("Loading model architecture and weights…")
with open(ARCH_PATH, 'r') as f:
    model = model_from_json(f.read())
model.load_weights(WEIGHTS_PATH)

# 5) Compila (con lo stesso optimizer e loss usati in training)
model.compile(
    optimizer=tf.keras.optimizers.Adam(),            # l’importante è il loss
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
print(f"Test Accuracy: {test_acc:.4f}")

from sklearn.metrics import f1_score, classification_report

# ... dopo model.evaluate(test_ds) ...

# 1. Prendi tutte le immagini e le label vere
y_true = []
for _, y in test_ds.unbatch():
    y_true.append(np.argmax(y.numpy()))
y_true = np.array(y_true)

# 2. Prendi tutte le predizioni
y_pred_proba = model.predict(test_ds)
y_pred = np.argmax(y_pred_proba, axis=1)

# 3. Calcola macro e micro F1
f1_macro = f1_score(y_true, y_pred, average='macro')
f1_micro = f1_score(y_true, y_pred, average='micro')

print(f"F1 (macro): {f1_macro:.4f}")
print(f"F1 (micro): {f1_micro:.4f}")

# (opzionale) report dettagliato
print(classification_report(y_true, y_pred, target_names=[
    'angry','disgust','fear','happy','sad','surprise','neutral'
]))

