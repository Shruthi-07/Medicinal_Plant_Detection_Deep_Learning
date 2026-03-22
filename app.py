from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
from datetime import datetime
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import io
import base64

from utils import predict_plant
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///medicinal_plants.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# Prediction history model
class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    predicted_class = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    user = db.relationship('User', backref=db.backref('predictions', lazy=True))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Plant class labels (from your training code)
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

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if password != confirm_password:
            flash('Passwords do not match!', 'danger')
            return render_template('register.html')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists!', 'danger')
            return render_template('register.html')
        
        if User.query.filter_by(email=email).first():
            flash('Email already exists!', 'danger')
            return render_template('register.html')
        
        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = bool(request.form.get('remember'))
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user, remember=remember)
            next_page = request.args.get('next')
            flash('Login successful!', 'success')
            return redirect(next_page) if next_page else redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password!', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    user_predictions = PredictionHistory.query.filter_by(user_id=current_user.id)\
        .order_by(PredictionHistory.created_at.desc()).limit(10).all()
    return render_template('upload.html', predictions=user_predictions)

def load_plant_data():
    try:
        df = pd.read_excel('F:/webapp/webapp/medicinal_plants_new.xlsx', sheet_name='Sheet1')
        plant_data = {}
        for _, row in df.iterrows():
            plant_data[row['Plant Name'].strip().lower()] = {
                'botanical_name': row['Botanical Name'],
                'common_name': row['Common Name'],
                'medicinal_usage': row['Medicinal Usage (Detailed)'],
                'regions_grown': row['Regions / Countries Where Grown'],
                'how_to_use': row['How to Use']  # Add this line
            }
        return plant_data
    except Exception as e:
        print(f"Error loading plant data: {e}")
        return {}

# Load plant data once at startup
PLANT_DATA = load_plant_data()

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files:
        flash('No file selected!', 'danger')
        return redirect(url_for('dashboard'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected!', 'danger')
        return redirect(url_for('dashboard'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Preprocess and predict
            #processed_image = preprocess_image(filepath)
            prediction, confidence = predict_plant(filepath)
            
            # Save prediction to database
            prediction_record = PredictionHistory(
                user_id=current_user.id,
                filename=filename,
                predicted_class=prediction,
                confidence=confidence
            )
            db.session.add(prediction_record)
            db.session.commit()
            
            # Convert image to base64 for display
            with open(filepath, 'rb') as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Get plant information
            plant_info = PLANT_DATA.get(prediction.lower().strip())
            print("plant_info==",plant_info)
            
            return render_template('result.html', 
                                prediction=prediction,
                                confidence=confidence,
                                image_data=img_data,
                                filename=filename,
                                plant_info=plant_info)
            
        except Exception as e:
            flash(f'Error processing image: {str(e)}', 'danger')
            return redirect(url_for('dashboard'))
    
    else:
        flash('Invalid file type! Please upload an image.', 'danger')
        return redirect(url_for('dashboard'))

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

@app.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            #processed_image = preprocess_image(filepath)
            prediction, confidence = predict_plant(filepath)
            
            # Save prediction to database
            prediction_record = PredictionHistory(
                user_id=current_user.id,
                filename=filename,
                predicted_class=prediction,
                confidence=confidence
            )
            db.session.add(prediction_record)
            db.session.commit()
            
            return jsonify({
                'prediction': prediction,
                'confidence': float(confidence),
                'filename': filename
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/history')
@login_required
def prediction_history():
    page = request.args.get('page', 1, type=int)
    per_page = 10
    
    predictions = PredictionHistory.query.filter_by(user_id=current_user.id)\
        .order_by(PredictionHistory.created_at.desc())\
        .paginate(page=page, per_page=per_page, error_out=False)
    
    return render_template('history.html', predictions=predictions)

# Initialize database and load model
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)