import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 

import tensorflow as tf
from PIL import Image
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, Input
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator

'''
import kagglehub

# Download latest version
path = kagglehub.dataset_download("idrisskh/obstacles-dataset")
print("Path to dataset files:", path)
'''

EPOCHS = 10
LR = 0.001
BATCH_SIZE = 32
IMAGE_SIZE = (64, 64)
NUM_CLASSES = 2
KERNEL_SIZE = (3, 3)
POOL_SIZE = (2, 2)


train_dir = 'dataset_detec/train'
val_dir = 'dataset_detec/val'
test_dir = 'dataset_detec/test'

datagen = ImageDataGenerator(rescale=1./255)
'''
tr_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
'''

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size= IMAGE_SIZE,
    batch_size= BATCH_SIZE,
    class_mode='categorical',
)
val_generator = datagen.flow_from_directory(    
    val_dir,
    target_size= IMAGE_SIZE,
    batch_size= BATCH_SIZE,
    class_mode='categorical'
)

test_generator = datagen.flow_from_directory(
    test_dir,
    target_size= IMAGE_SIZE,
    batch_size= BATCH_SIZE,
    class_mode='categorical'
)

# Define the CNN model
'''
model = Sequential([
    Input(shape=(*IMAGE_SIZE, 3)),
    Conv2D(64, KERNEL_SIZE, activation='relu'),
    BatchNormalization(),
    Conv2D(64, KERNEL_SIZE, activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=POOL_SIZE),  
    Dropout(0.25), 
    
    Conv2D(128, KERNEL_SIZE, activation='relu'),
    BatchNormalization(),
    Conv2D(128, KERNEL_SIZE, activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=POOL_SIZE),
    Dropout(0.25),
    
    GlobalAveragePooling2D(),
    
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])
'''

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])


'''
lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=LR,
    decay_steps=10000,
    decay_rate=0.9
)
'''

Optimizer = tf.keras.optimizers.Adam(
    learning_rate= LR,
    beta_1=0.9, 
    beta_2=0.999, 
    epsilon=1e-08, 
    name= 'adam',
)

model.compile(optimizer=Optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

'''
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
]
'''

model.fit(
    train_generator,
    validation_data=val_generator,
    epochs= EPOCHS,
    #callbacks= callbacks
)

loss, accuracy = model.evaluate(test_generator)

print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
