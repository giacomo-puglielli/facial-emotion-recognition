import os
import pandas as pd
import numpy as np
from preprocessing.cleaning import preprocess_x, preprocess_y

# 1. Set up paths
BASE_DIR = os.path.dirname(os.path.abspath("pipeline.py"))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DIR =  os.path.join(DATA_DIR, 'raw')        # CSVs located here
PREPRO_DIR = os.path.join(DATA_DIR, 'preprocessed')
MODEL_DIR  = os.path.join(BASE_DIR, "models")

# 2. Load raw CSVs
train_df = pd.read_csv(os.path.join(RAW_DIR, 'train.csv'))
val_df   = pd.read_csv(os.path.join(RAW_DIR, 'val.csv'))
test_df  = pd.read_csv(os.path.join(RAW_DIR, 'test.csv'))

# Optional: quick check on raw data
#print("Raw data previews:")
#print("Train CSV head:")
#print(train_df.head().to_csv(index=False))
#print("Validation CSV head:")
#print(val_df.head().to_csv(index=False))
#print("Test CSV head:")
#print(test_df.head().to_csv(index=False))

# 3. Preprocess features and labels
x_train = preprocess_x(train_df)
y_train = preprocess_y(train_df)

x_val   = preprocess_x(val_df)
y_val   = preprocess_y(val_df)

x_test  = preprocess_x(test_df)
y_test  = preprocess_y(test_df)

# 4. Save preprocessed arrays
#np.save(os.path.join(PREPRO_DIR, 'x_train.npy'), x_train)
#np.save(os.path.join(PREPRO_DIR, 'y_train.npy'), y_train)

#np.save(os.path.join(PREPRO_DIR, 'x_val.npy'), x_val)
#np.save(os.path.join(PREPRO_DIR, 'y_val.npy'), y_val)

#np.save(os.path.join(PREPRO_DIR, 'x_test.npy'), x_test)
#np.save(os.path.join(PREPRO_DIR, 'y_test.npy'), y_test)

# 5. Ready for modeling and training
print(f"Data loaded and preprocessed.\n"
      f"x_train: {x_train.shape}, y_train: {y_train.shape}\n"
      f"x_val:   {x_val.shape}, y_val:   {y_val.shape}\n"
      f"x_test:  {x_test.shape}, y_test:  {y_test.shape}")

# 6. Setting the Generator for training 
from keras.preprocessing.image import ImageDataGenerator 
datagen = ImageDataGenerator( 
    rescale=1./255,
    rotation_range = 10,
    horizontal_flip = True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode = 'nearest') # augmented generator for training
testgen = ImageDataGenerator(rescale=1./255) # just normalizing for test
datagen.fit(x_train) # applying augmented generator to training set
batch_size = 64

train_flow = datagen.flow(x_train, y_train, batch_size=batch_size) 
test_flow = testgen.flow(x_test, y_test, batch_size=batch_size)
val_flow = testgen.flow(x_val, y_val, batch_size=batch_size)


from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam
from models.creation_model import FER_Model

# 7. Setting schedule for training
lr_schedule = ExponentialDecay(
    initial_learning_rate=0.001,  # learning rate iniziale
    decay_steps=1000,             # ogni quanti steps deve decrescere
    decay_rate=0.96,              # quanto scende ogni volta
    staircase=True                # se vuoi farlo "a gradini"
)

opt = Adam(learning_rate=lr_schedule)

# 8. Creation and compilation model
model = FER_Model()
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# 9. Training the model
from keras.callbacks import ModelCheckpoint

# Defining the callback
checkpoint = ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, "best_model.h5"),
    monitor="val_loss",
    save_best_only=True,
    mode="min",
    verbose=1
)

steps_per_epoch  = len(x_train) // batch_size
validation_steps = len(x_val)   // batch_size
num_epochs = 100  
history = model.fit(train_flow, 
                    steps_per_epoch=steps_per_epoch, 
                    epochs=num_epochs,  
                    verbose=1,  
                    validation_data=val_flow,
                    validation_steps=validation_steps,
                    callbacks=[checkpoint])

# 10. Saving the model
model_json = model.to_json()
json_path = os.path.join(MODEL_DIR, "model.json")
with open(json_path, "w") as json_file:
    json_file.write(model_json)

# 2. Salva i pesi in HDF5 dentro models/
weights_path = os.path.join(MODEL_DIR, "model.h5")
model.save_weights(weights_path)

print(f"Model saved:\n • architecture → {json_path}\n • weights      → {weights_path}")
