import os
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet import preprocess_input
from tqdm import tqdm  

emotion_model = tf.keras.models.load_model('./pretrained_models/late_fusion2_model.h5')

labels = ['amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness']

def process_folder(folder_path, labels):
    folder_name = os.path.basename(folder_path)

    results = pd.DataFrame(columns=['Brand'] + ['Filename'] + labels + ['Predicted Emotion'])  

    for filename in tqdm(os.listdir(folder_path), desc=f"Processing {folder_name}"):  
        if filename.endswith('.jpg'):
            image = Image.open(os.path.join(folder_path, filename))
            size = (224, 224)
            im = ImageOps.fit(image, size, Image.ANTIALIAS)
            im = img_to_array(im)
            im = im.reshape((1, im.shape[0], im.shape[1], im.shape[2]))
            im = preprocess_input(im)

            preds = emotion_model.predict([im, im])
            predicted_index = np.argmax(preds, axis=1)[0]
            predicted_emotion = labels[predicted_index]

            result_row = [folder_name, filename] + preds.tolist()[0] + [predicted_emotion]
            results.loc[len(results)] = result_row

    subfolder_result_path = os.path.join('sub_result', f'{folder_name}_result.csv')
    results.to_csv(subfolder_result_path, index=False)

sub_result_folder = 'sub_result'
if not os.path.exists(sub_result_folder):
    os.mkdir(sub_result_folder)

merged_results = pd.DataFrame(columns=['Brand'] + ['Filename'] + labels + ['Predicted Emotion'])

root_folder = './data'
for subfolder in os.listdir(root_folder):
    if os.path.isdir(os.path.join(root_folder, subfolder)):
        subfolder_path = os.path.join(root_folder, subfolder)
        process_folder(subfolder_path, labels)
        subfolder_result_path = os.path.join('sub_result', f'{subfolder}_result.csv')

        subfolder_results = pd.read_csv(subfolder_result_path)
        merged_results = pd.concat([merged_results, subfolder_results], ignore_index=True)

merged_results.to_csv('merged_result.csv', index=False)
