import numpy as np
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import metrics
import tensorflow as tf

# Import additional libraries for confusion matrix and classification report
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

train_path = "./Train"
test_path = "./Test"
IMAGE_SIZE = [256, 256]

# Define custom metrics
def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

def precision_m(y_true, y_pred):
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    return precision

def recall_m(y_true, y_pred):
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
    return recall

# Enhanced preprocessing function
def ensure_3_channels(img):
    """
    Ensure the image has exactly 3 channels
    """
    # Convert to float32 first
    img = img.astype('float32')
    
    # Handle different channel cases
    if len(img.shape) == 2:  # Grayscale (H, W)
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[-1] == 1:  # Grayscale with channel dimension (H, W, 1)
        img = np.concatenate([img, img, img], axis=-1)
    elif img.shape[-1] == 4:  # RGBA
        img = img[:, :, :3]
    elif img.shape[-1] > 3:  # More than 4 channels, take first 3
        img = img[:, :, :3]
    
    # Ensure the output has exactly 3 channels
    assert img.shape[-1] == 3, f"Expected 3 channels, got {img.shape[-1]}"
    return img

# Create data generators
train_datagen = ImageDataGenerator(
    rescale=1./255, 
    shear_range=0.2, 
    zoom_range=0.2, 
    horizontal_flip=False,
    preprocessing_function=ensure_3_channels
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=ensure_3_channels
)

print("Creating training dataset...")
train_set = train_datagen.flow_from_directory(
    train_path,
    target_size=(256, 256),
    batch_size=2,
    class_mode='categorical',
    color_mode='rgb',  # Force RGB loading
    shuffle=True
)

print("Creating test dataset...")
test_set = test_datagen.flow_from_directory(
    test_path,
    target_size=(256, 256),
    batch_size=2,
    class_mode='categorical',
    color_mode='rgb',  # Force RGB loading
    shuffle=False  # Important: Don't shuffle test set for evaluation
)

print(f"Training samples: {train_set.samples}")
print(f"Test samples: {test_set.samples}")

# Verify the image shape
for x_batch, y_batch in train_set:
    print(f"Batch shape: {x_batch.shape}")
    print(f"Data type: {x_batch.dtype}")
    print(f"Min value: {x_batch.min()}, Max value: {x_batch.max()}")
    break

# Create model without pre-trained weights to avoid shape mismatch
print("Creating EfficientNetB3 model...")
try:
    base_model = EfficientNetB3(
        input_shape=(256, 256, 3),
        weights='imagenet', 
        include_top=False
    )
    print("Model created with pre-trained weights successfully!")
except ValueError as e:
    print(f"Error with pre-trained weights: {e}")
    print("Creating model with random initialization...")
    base_model = EfficientNetB3(
        input_shape=(256, 256, 3),
        weights=None,  # Random initialization
        include_top=False
    )
    print("Model created with random initialization successfully!")

# Add custom layers
x = Flatten()(base_model.output)
predictions = Dense(43, activation='softmax')(x)
model5 = Model(inputs=base_model.inputs, outputs=predictions)

print("Final model created successfully!")
model5.summary()

# Define proper callbacks
learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_accuracy',
    patience=3,
    verbose=1,
    factor=0.5,
    min_lr=0.00001
)

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    verbose=1,
    restore_best_weights=True
)

# Compile the model
model5.compile(
    loss='categorical_crossentropy', 
    optimizer='adam', 
    metrics=["accuracy", f1_m, precision_m, recall_m]
)

print("Starting training...")
hist5 = model5.fit(
    train_set, 
    validation_data=test_set, 
    epochs=30, 
    steps_per_epoch=len(train_set), 
    validation_steps=len(test_set),
    callbacks=[learning_rate_reduction, early_stop]
)

model5.save('efficient.h5')

# Get metrics
dl_acc = hist5.history["val_accuracy"][-1]
dl_prec = hist5.history["val_precision_m"][-1]
dl_rec = hist5.history["val_recall_m"][-1]
dl_f1 = hist5.history["val_f1_m"][-1]

print(f"Validation Accuracy: {dl_acc:.4f}")
print(f"Validation Precision: {dl_prec:.4f}")
print(f"Validation Recall: {dl_rec:.4f}")
print(f"Validation F1-Score: {dl_f1:.4f}")

# ========== ADDED: Confusion Matrix and Classification Report ==========

print("\n" + "="*60)
print("GENERATING CONFUSION MATRIX AND CLASSIFICATION REPORT")
print("="*60)

# Reset test generator and get all predictions
test_set.reset()
print("Generating predictions...")
Y_pred = model5.predict(test_set, steps=len(test_set))
y_pred = np.argmax(Y_pred, axis=1)

# Get true labels
true_classes = test_set.classes
class_labels = list(test_set.class_indices.keys())

print(f"True classes shape: {true_classes.shape}")
print(f"Predicted classes shape: {y_pred.shape}")
print(f"Number of classes: {len(class_labels)}")

# Generate confusion matrix
cm = confusion_matrix(true_classes, y_pred)

# Plot confusion matrix
plt.figure(figsize=(15, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_labels, 
            yticklabels=class_labels,
            cbar_kws={'shrink': 0.8})
plt.title('Confusion Matrix - EfficientNetB3', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('efficientnet_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Generate classification report
print("\n" + "="*60)
print("CLASSIFICATION REPORT - EfficientNetB3")
print("="*60)
cr = classification_report(true_classes, y_pred, target_names=class_labels, digits=4)
print(cr)

# Save classification report to file
with open('efficientnet_classification_report.txt', 'w') as f:
    f.write("Classification Report - EfficientNetB3\n")
    f.write("="*60 + "\n")
    f.write(cr)
    f.write("\n\nValidation Metrics Summary:\n")
    f.write(f"Accuracy: {dl_acc:.4f}\n")
    f.write(f"Precision: {dl_prec:.4f}\n")
    f.write(f"Recall: {dl_rec:.4f}\n")
    f.write(f"F1-Score: {dl_f1:.4f}\n")

# Calculate and print overall metrics from confusion matrix
total = np.sum(cm)
accuracy = np.trace(cm) / total if total > 0 else 0

print(f"\nOverall Accuracy from Confusion Matrix: {accuracy:.4f}")
print(f"Total samples: {total}")

# Calculate precision, recall, F1 for each class
print("\n" + "="*60)
print("PER-CLASS METRICS")
print("="*60)

class_metrics = []
for i, class_name in enumerate(class_labels):
    tp = cm[i, i]
    fp = np.sum(cm[:, i]) - tp
    fn = np.sum(cm[i, :]) - tp
    tn = total - tp - fp - fn
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    class_metrics.append({
        'class': class_name,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': np.sum(cm[i, :])
    })
    
    print(f"{class_name:20s} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f} | Support: {np.sum(cm[i, :]):3d}")

# Calculate macro and weighted averages
macro_precision = np.mean([m['precision'] for m in class_metrics])
macro_recall = np.mean([m['recall'] for m in class_metrics])
macro_f1 = np.mean([m['f1'] for m in class_metrics])

weighted_precision = np.average([m['precision'] for m in class_metrics], weights=[m['support'] for m in class_metrics])
weighted_recall = np.average([m['recall'] for m in class_metrics], weights=[m['support'] for m in class_metrics])
weighted_f1 = np.average([m['f1'] for m in class_metrics], weights=[m['support'] for m in class_metrics])

print("\n" + "="*60)
print("OVERALL METRICS SUMMARY")
print("="*60)
print(f"Macro Average Precision:  {macro_precision:.4f}")
print(f"Macro Average Recall:     {macro_recall:.4f}")
print(f"Macro Average F1-Score:   {macro_f1:.4f}")
print(f"Weighted Average Precision: {weighted_precision:.4f}")
print(f"Weighted Average Recall:    {weighted_recall:.4f}")
print(f"Weighted Average F1-Score:  {weighted_f1:.4f}")

print("\n" + "="*60)
print("FILES SAVED")
print("="*60)
print("✓ Confusion matrix saved as: 'efficientnet_confusion_matrix.png'")
print("✓ Classification report saved as: 'efficientnet_classification_report.txt'")
print("✓ Model saved as: 'efficient.h5'")

# Plot training history
plt.figure(figsize=(15, 5))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(hist5.history['accuracy'], label='Training Accuracy')
plt.plot(hist5.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(hist5.history['loss'], label='Training Loss')
plt.plot(hist5.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('efficientnet_training_history.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Training history plot saved as: 'efficientnet_training_history.png'")