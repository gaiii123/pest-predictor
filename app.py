from flask import Flask, request, render_template, redirect, url_for, session
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import base64
import io

app = Flask(__name__)
app.secret_key = 'your_secret_key'  #secret key
model = load_model('my_model.h5')

# class labels
class_labels = ['cordana', 'healthy', 'Unknown image', 'sigatoka']

def prepare_image(file, target_size):
    # Convert the file to a byte stream
    image = load_img(io.BytesIO(file.read()), target_size=target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

def is_confident(prediction, threshold=0.5):
    max_prob = np.max(prediction)
    if max_prob < threshold:
        return False
    return True

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    result = None
    uploaded_image = None
    if 'username' in session:
        if request.method == 'POST':
            file = request.files['file']
            if file:
                # Process image for prediction
                image = prepare_image(file, target_size=(224, 224))
                prediction = model.predict(image)
                predicted_class = np.argmax(prediction, axis=1)[0]

                # Check confidence
                if is_confident(prediction):
                    if predicted_class < len(class_labels):
                        result = class_labels[predicted_class]
                    else:
                        result = "Unknown class"
                else:
                    result = "Not confident enough to classify"

    

        return render_template('upload_and_result.html', result=result, uploaded_image=uploaded_image)
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        session.pop('username', None)
        username = request.form['username']
        password = request.form['password']

        #actual authentication logic 
        if username == 'user' and password == 'password':
            session['username'] = username
            return redirect(url_for('upload_image'))
        else:
            return render_template('login.html', error="Invalid username or password")

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
