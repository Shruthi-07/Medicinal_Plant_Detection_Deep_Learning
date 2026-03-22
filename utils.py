import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# Load the trained model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_XCEPTION = os.path.join(BASE_DIR, "xception.h5")
MODEL_EFFICIENT = os.path.join(BASE_DIR, "efficient.h5")
model1 = tf.keras.models.load_model(MODEL_XCEPTION, compile=False)
model2 = tf.keras.models.load_model(MODEL_EFFICIENT, compile=False)

print("✅ Loaded base models")

# -----------------------------
# Ensemble function (average)
# -----------------------------
def ensemble_predict(img_array):
    pred1 = model1.predict(img_array)
    pred2 = model2.predict(img_array)
    final_pred = (pred1 + pred2) / 2.0
    return final_pred

# Plant class labels
PLANT_CLASSES = {
    0: 'Aloevera', 1: 'Amla', 2: 'Amruthaballi', 3: 'Arali', 4: 'Astma_weed', 
    5: 'Badipala', 6: 'Balloon_Vine', 7: 'Bamboo', 8: 'Beans', 9: 'Betel', 
    10: 'Bhrami', 11: 'Bringaraja', 12: 'Caricature', 13: 'Castor', 14: 'Catharanthus', 
    15: 'Chakte', 16: 'Chilly', 17: 'Citron lime (herelikai)', 18: 'Coffee', 
    19: 'Common rue(naagdalli)', 20: 'Coriender', 21: 'Curry', 22: 'Doddpathre', 
    23: 'Drumstick', 24: 'Ekka', 25: 'Eucalyptus', 26: 'Ganigale', 27: 'Ganike', 
    28: 'Gasagase', 29: 'Ginger', 30: 'Globe Amarnath', 31: 'Guava', 32: 'Henna', 
    33: 'Hibiscus', 34: 'Honge', 35: 'Insulin', 36: 'Jackfruit', 37: 'Jasmine', 
    38: 'Kambajala', 39: 'Kasambruga', 40: 'Kohlrabi', 41: 'Lantana', 42: 'Lemon', 
    43: 'Lemongrass', 44: 'Malabar_Nut', 45: 'Malabar_Spinach', 46: 'Mango', 
    47: 'Marigold', 48: 'Mint', 49: 'Neem', 50: 'Nelavembu', 51: 'Nerale', 
    52: 'Nooni', 53: 'Onion', 54: 'Padri', 55: 'Palak(Spinach)', 56: 'Papaya', 
    57: 'Parijatha', 58: 'Pea', 59: 'Pepper', 60: 'Pomoegranate', 61: 'Pumpkin', 
    62: 'Raddish', 63: 'Rose', 64: 'Sampige', 65: 'Sapota', 66: 'Seethaashoka', 
    67: 'Seethapala', 68: 'Spinach1', 69: 'Tamarind', 70: 'Taro', 71: 'Tecoma', 
    72: 'Thumbe', 73: 'Tomato', 74: 'Tulsi', 75: 'Turmeric', 76: 'ashoka', 
    77: 'camphor', 78: 'kamakasturi', 79: 'kepala'
}

# Load model (lazy loading)
_model = None

def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model

def predict_plant(img_path):
    img = load_img(img_path, target_size=(256, 256))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    final_pred = ensemble_predict(img)[0]
    class_idx = np.argmax(final_pred)
    confidence = np.max(final_pred)

    print("\n✅ Prediction Successful!")
    print(f"🌿 Predicted Plant: {PLANT_CLASSES[class_idx]}")
    print(f"🔍 Confidence: {confidence:.2%}\n")
    return PLANT_CLASSES[class_idx], confidence  # Fixed the syntax error here