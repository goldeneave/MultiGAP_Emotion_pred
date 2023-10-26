import gradio as gr
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet import preprocess_input

# Load pretrained model
emotion_model = tf.keras.models.load_model('./pretrained_models/late_fusion2_model.h5')

# Define the emotion labels
labels = ['amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness']

# Define a function to make predictions
def predict_emotion(image):
    image = ImageOps.fit(image, (224, 224), Image.ANTIALIAS)
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)

    # Predict emotions
    preds = emotion_model.predict([image, image])
    predicted_index = np.argmax(preds, axis=1)[0]
    predicted_emotion = labels[predicted_index]

    return {predicted_emotion: preds.tolist()[0]}

# Create a Gradio interface
iface = gr.Interface(fn=predict_emotion,
                     inputs=gr.inputs.Image(type="pil", label="Upload an image"),
                     outputs="json")

# Launch the Gradio interface on a local web server
iface.launch(share=True)
