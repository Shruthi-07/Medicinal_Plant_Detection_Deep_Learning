from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.layers import Activation, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Average
import sys
import shutil
from glob import glob
import json
import math
import os
import cv2
import glob as gb
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.models import Sequential
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Input, Lambda
from tensorflow.keras.applications import ResNet50, InceptionV3, EfficientNetB3

# Import additional libraries for confusion matrix and classification report
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd

train_path = "./Train"
test_path = "./Test"
IMAGE_SIZE = [256,256]

# Scaling all the images between 0 to 1
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=False)

# Performing only scaling on the test dataset
test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory(train_path,
                                              target_size=(256,256),
                                              batch_size=2,
                                              class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(test_path,
                                            target_size=(256,256),
                                            batch_size=2,
                                            class_mode='categorical')

from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential, Model

learning_rate_reduction = ReduceLROnPlateau(
    monitor="val_accuracy", patience=3, verbose=1, factor=0.3, min_lr=0.0000001
)
early_stop = EarlyStopping(
    patience=10,
    verbose=1,
    monitor="val_accuracy",
    mode="max",
    min_delta=0.001,
    restore_best_weights=True,
)

from tensorflow.keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

base_model = Xception(input_shape = IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(43, activation='softmax')(x)

# this is the model we will train
model4 = Model(inputs=base_model.input, outputs=predictions)
model4.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=["accuracy", f1_m, precision_m, recall_m])
model4.summary()

hist4 = model4.fit(train_set, validation_data=test_set, epochs=30, steps_per_epoch=len(train_set), validation_steps=len(test_set))
model4.save('xception.h5')

dl_acc = hist4.history["val_accuracy"][-1]  # Use -1 to get the last epoch value
dl_prec = hist4.history["val_precision_m"][-1]
dl_rec = hist4.history["val_recall_m"][-1]
dl_f1 = hist4.history["val_f1_m"][-1]

print(f"Validation Accuracy: {dl_acc:.4f}")
print(f"Validation Precision: {dl_prec:.4f}")
print(f"Validation Recall: {dl_rec:.4f}")
print(f"Validation F1-Score: {dl_f1:.4f}")

# ========== ADDED: Confusion Matrix and Classification Report ==========

# Get true labels and predictions
print("\nGenerating confusion matrix and classification report...")

# Reset test generator and get all predictions
test_set.reset()
Y_pred = model4.predict(test_set, steps=len(test_set))
y_pred = np.argmax(Y_pred, axis=1)

# Get true labels
true_classes = test_set.classes
class_labels = list(test_set.class_indices.keys())

# Generate confusion matrix
cm = confusion_matrix(true_classes, y_pred)

# Plot confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_labels, 
            yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Generate classification report
print("\n" + "="*50)
print("CLASSIFICATION REPORT")
print("="*50)
cr = classification_report(true_classes, y_pred, target_names=class_labels)
print(cr)

# Save classification report to file
with open('classification_report.txt', 'w') as f:
    f.write("Classification Report\n")
    f.write("="*50 + "\n")
    f.write(cr)
    
# Calculate and print overall metrics from confusion matrix
total = np.sum(cm)
accuracy = np.trace(cm) / total
print(f"\nOverall Accuracy from Confusion Matrix: {accuracy:.4f}")

# Calculate precision, recall, F1 for each class
print("\n" + "="*50)
print("PER-CLASS METRICS")
print("="*50)

for i, class_name in enumerate(class_labels):
    tp = cm[i, i]
    fp = np.sum(cm[:, i]) - tp
    fn = np.sum(cm[i, :]) - tp
    tn = total - tp - fp - fn
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"{class_name:20s} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f}")

print("\nConfusion matrix saved as 'confusion_matrix.png'")
print("Classification report saved as 'classification_report.txt'")