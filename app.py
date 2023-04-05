import os
import os.path
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)
Bootstrap(app)
app.config['UPLOAD_FOLDER'] = 'uploads'

loaded_model = load_model('spoon_identifier_model.h5')

def is_spoon(model, image_path):
    image = load_img(image_path, target_size=(224, 224))
    image_array = img_to_array(image) / 255.0
    image_batch = np.expand_dims(image_array, axis=0)
    prediction = model.predict(image_batch)
    return prediction[0] > 0.5


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        if is_spoon(loaded_model, file_path):
            result = "The uploaded image is classified as a spoon."
        else:
            result = "The uploaded image is not classified as a spoon."

        return render_template('result.html', result=result, image_path=file_path)

    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.template_filter('basename')
def basename_filter(s):
    return os.path.basename(s)



if __name__ == '__main__':
    app.run(debug=True)
