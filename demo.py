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
from tensorflow.keras.models import Model
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

def ensemble():
    # Load models
    model1 = load_model('xception.h5', compile=False)
    model2 = load_model('efficient.h5', compile=False)
    
    # Create a simple Sequential ensemble model
    ensemble_model = tf.keras.Sequential(name='Ensemble')
    
    # Create input layer
    ensemble_model.add(tf.keras.layers.InputLayer(input_shape=(256, 256, 3), name='input'))
    
    # Custom layer that averages predictions from both models
    class ModelAveraging(tf.keras.layers.Layer):
        def __init__(self, model1, model2, **kwargs):
            super(ModelAveraging, self).__init__(**kwargs)
            self.model1 = model1
            self.model2 = model2
            
        def call(self, inputs):
            pred1 = self.model1(inputs)
            pred2 = self.model2(inputs)
            return (pred1 + pred2) / 2.0
            
        def get_config(self):
            config = super().get_config()
            return config
    
    # Add the averaging layer
    ensemble_model.add(ModelAveraging(model1, model2, name='model_averaging'))
    
    return ensemble_model

model = ensemble()
model.compile(optimizer='sgd', 
              loss = 'categorical_crossentropy', 
              metrics=["accuracy",f1_m,precision_m, recall_m])
model.summary()
hist = model.fit(train_set, validation_data=test_set, epochs=30, steps_per_epoch=len(train_set), validation_steps=len(test_set))
model.save('hybridmodel.h5')

# ========== ADDED: Confusion Matrix and Classification Report for Ensemble Model ==========

print("\n" + "="*70)
print("ENSEMBLE MODEL EVALUATION - CONFUSION MATRIX AND CLASSIFICATION REPORT")
print("="*70)

# Get training metrics
train_acc = hist.history["accuracy"][-1] if "accuracy" in hist.history else 0
val_acc = hist.history["val_accuracy"][-1] if "val_accuracy" in hist.history else 0
val_prec = hist.history["val_precision_m"][-1] if "val_precision_m" in hist.history else 0
val_rec = hist.history["val_recall_m"][-1] if "val_recall_m" in hist.history else 0
val_f1 = hist.history["val_f1_m"][-1] if "val_f1_m" in hist.history else 0

print(f"Training Accuracy: {train_acc:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")
print(f"Validation Precision: {val_prec:.4f}")
print(f"Validation Recall: {val_rec:.4f}")
print(f"Validation F1-Score: {val_f1:.4f}")

# Reset test generator and get all predictions
test_set.reset()
print("\nGenerating ensemble model predictions...")
Y_pred_ensemble = model.predict(test_set, steps=len(test_set))
y_pred_ensemble = np.argmax(Y_pred_ensemble, axis=1)

# Get true labels
true_classes = test_set.classes
class_labels = list(test_set.class_indices.keys())

print(f"True classes shape: {true_classes.shape}")
print(f"Predicted classes shape: {y_pred_ensemble.shape}")
print(f"Number of classes: {len(class_labels)}")

# Generate confusion matrix
cm_ensemble = confusion_matrix(true_classes, y_pred_ensemble)

# Plot confusion matrix
plt.figure(figsize=(16, 14))
sns.heatmap(cm_ensemble, annot=True, fmt='d', cmap='viridis', 
            xticklabels=class_labels, 
            yticklabels=class_labels,
            cbar_kws={'shrink': 0.8},
            annot_kws={'size': 8})
plt.title('Confusion Matrix - Ensemble Model (Xception + EfficientNetB3)', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig('ensemble_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Generate classification report
print("\n" + "="*70)
print("CLASSIFICATION REPORT - ENSEMBLE MODEL")
print("="*70)
cr_ensemble = classification_report(true_classes, y_pred_ensemble, target_names=class_labels, digits=4)
print(cr_ensemble)

# Save classification report to file
with open('ensemble_classification_report.txt', 'w') as f:
    f.write("Classification Report - Ensemble Model (Xception + EfficientNetB3)\n")
    f.write("="*70 + "\n")
    f.write(cr_ensemble)
    f.write("\n\nTraining Metrics Summary:\n")
    f.write(f"Training Accuracy: {train_acc:.4f}\n")
    f.write(f"Validation Accuracy: {val_acc:.4f}\n")
    f.write(f"Validation Precision: {val_prec:.4f}\n")
    f.write(f"Validation Recall: {val_rec:.4f}\n")
    f.write(f"Validation F1-Score: {val_f1:.4f}\n")

# Calculate and print overall metrics from confusion matrix
total = np.sum(cm_ensemble)
accuracy_from_cm = np.trace(cm_ensemble) / total if total > 0 else 0

print(f"\nOverall Accuracy from Confusion Matrix: {accuracy_from_cm:.4f}")
print(f"Total test samples: {total}")

# Calculate precision, recall, F1 for each class
print("\n" + "="*70)
print("PER-CLASS METRICS - ENSEMBLE MODEL")
print("="*70)

class_metrics_ensemble = []
for i, class_name in enumerate(class_labels):
    tp = cm_ensemble[i, i]
    fp = np.sum(cm_ensemble[:, i]) - tp
    fn = np.sum(cm_ensemble[i, :]) - tp
    tn = total - tp - fp - fn
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    support = np.sum(cm_ensemble[i, :])
    
    class_metrics_ensemble.append({
        'class': class_name,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'tp': tp,
        'fp': fp,
        'fn': fn
    })
    
    print(f"{class_name:25s} | Prec: {precision:.4f} | Rec: {recall:.4f} | F1: {f1:.4f} | Support: {support:3d}")

# Calculate macro and weighted averages
macro_precision = np.mean([m['precision'] for m in class_metrics_ensemble])
macro_recall = np.mean([m['recall'] for m in class_metrics_ensemble])
macro_f1 = np.mean([m['f1'] for m in class_metrics_ensemble])

weighted_precision = np.average([m['precision'] for m in class_metrics_ensemble], weights=[m['support'] for m in class_metrics_ensemble])
weighted_recall = np.average([m['recall'] for m in class_metrics_ensemble], weights=[m['support'] for m in class_metrics_ensemble])
weighted_f1 = np.average([m['f1'] for m in class_metrics_ensemble], weights=[m['support'] for m in class_metrics_ensemble])

print("\n" + "="*70)
print("OVERALL METRICS SUMMARY - ENSEMBLE MODEL")
print("="*70)
print(f"Macro Average Precision:    {macro_precision:.4f}")
print(f"Macro Average Recall:       {macro_recall:.4f}")
print(f"Macro Average F1-Score:     {macro_f1:.4f}")
print(f"Weighted Average Precision: {weighted_precision:.4f}")
print(f"Weighted Average Recall:    {weighted_recall:.4f}")
print(f"Weighted Average F1-Score:  {weighted_f1:.4f}")

# Plot training history for ensemble model
if len(hist.history) > 0:
    plt.figure(figsize=(15, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    if 'accuracy' in hist.history:
        plt.plot(hist.history['accuracy'], label='Training Accuracy', linewidth=2)
    if 'val_accuracy' in hist.history:
        plt.plot(hist.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    plt.title('Ensemble Model Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    if 'loss' in hist.history:
        plt.plot(hist.history['loss'], label='Training Loss', linewidth=2)
    if 'val_loss' in hist.history:
        plt.plot(hist.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Ensemble Model Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ensemble_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

# Create a summary comparison table
print("\n" + "="*70)
print("ENSEMBLE MODEL PERFORMANCE SUMMARY")
print("="*70)
print(f"{'Metric':<25} {'Value':<10}")
print(f"{'-'*35}")
print(f"{'Training Accuracy':<25} {train_acc:.4f}")
print(f"{'Validation Accuracy':<25} {val_acc:.4f}")
print(f"{'Validation Precision':<25} {val_prec:.4f}")
print(f"{'Validation Recall':<25} {val_rec:.4f}")
print(f"{'Validation F1-Score':<25} {val_f1:.4f}")
print(f"{'CM Accuracy':<25} {accuracy_from_cm:.4f}")
print(f"{'Macro F1-Score':<25} {macro_f1:.4f}")
print(f"{'Weighted F1-Score':<25} {weighted_f1:.4f}")

print("\n" + "="*70)
print("FILES SAVED")
print("="*70)
print("✓ Ensemble confusion matrix saved as: 'ensemble_confusion_matrix.png'")
print("✓ Ensemble classification report saved as: 'ensemble_classification_report.txt'")
print("✓ Ensemble training history saved as: 'ensemble_training_history.png'")
print("✓ Ensemble model saved as: 'hybridmodel.h5'")

# Additional: Show top performing classes
print("\n" + "="*70)
print("TOP 10 BEST PERFORMING CLASSES (by F1-Score)")
print("="*70)
top_classes = sorted(class_metrics_ensemble, key=lambda x: x['f1'], reverse=True)[:10]
for i, class_metric in enumerate(top_classes):
    print(f"{i+1:2d}. {class_metric['class']:25s} | F1: {class_metric['f1']:.4f} | Precision: {class_metric['precision']:.4f} | Recall: {class_metric['recall']:.4f}")

print("\nEnsemble model evaluation completed successfully!")