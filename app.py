from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import uuid
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('models/model.h5')

# Class labels
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

# Define static/uploads folder
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Helper function to predict tumor type
def predict_tumor(image_path):
    IMAGE_SIZE = 128
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions, axis=1)[0]

    if class_labels[predicted_class_index] == 'notumor':
        return "No Tumor", confidence_score
    else:
        return f"Tumor: {class_labels[predicted_class_index]}", confidence_score

# Route for the main page
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    confidence = None
    file_path = None
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html')
        
        file = request.files['file']
        
        if file.filename == '':
            return render_template('index.html')
        
        if file:
            # Generate unique filename
            original_filename = secure_filename(file.filename)
            unique_id = uuid.uuid4().hex[:8]
            filename = f"{unique_id}_{original_filename}"
            
            # Save file to static/uploads
            file_location = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_location)

            # Predict
            result, confidence_score = predict_tumor(file_location)
            confidence = f"{confidence_score*100:.2f}%"
            
            # Only uploads/filename - Flask will serve from static automatically
            file_path = f'uploads/{filename}'
    
    return render_template('index.html', 
                          result=result, 
                          confidence=confidence, 
                          file_path=file_path)

if __name__ == '__main__':
    app.run(debug=True)