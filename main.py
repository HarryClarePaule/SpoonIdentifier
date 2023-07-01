import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

loaded_model = load_model('spoon_identifier_model.h5')

def classify_images(model, validation_images_dir):
    image_files = os.listdir(validation_images_dir)
    spoon_count = 0
    not_spoon_count = 0

    for image_file in image_files:
        image_path = os.path.join(validation_images_dir, image_file)
        image = load_img(image_path, target_size=(224, 224))
        image_array = img_to_array(image) / 255.0
        image_batch = np.expand_dims(image_array, axis=0)

        prediction = model.predict(image_batch)

        if prediction[0] > 0.5:
            print(f"{image_file} is classified as a spoon.")
            spoon_count += 1
        else:
            print(f"{image_file} is not classified as a spoon.")
            not_spoon_count += 1

    print(f"\nTotal spoons: {spoon_count}")
    print(f"Total not spoons: {not_spoon_count}")

# Usage example:
validation_images_dir = 'ValidationImages'
classify_images(loaded_model, validation_images_dir)

