import streamlit as st
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

@st.cache_resource  # 👈 Add the caching decorator
def load_tfmodel():
    return load_model("keras_model.h5", compile=False)



st.title("Mask Detector")
st.write("Take photo to see the ML model identfying whether you have worn a mask or not")
col1,col2 = st.columns(2)
with col1:
    with st.container(border=True):
        input_img = st.camera_input(label="input camera")

if input_img:
    # st.image(input_img,width=200)
    # print(type(input_img))
    np.set_printoptions(suppress=True)
    # Load the model
    model = load_tfmodel()

    # Load the labels
    class_names = open("labels.txt", "r").readlines()

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    # image = Image.open().convert("RGB")
    image = Image.open(input_img)
    
    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    
    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", confidence_score)
    with col2:
        with st.container(border=True):
            st.write(f"**Predicted class:** {class_name[2:]} ")
            st.write(f"**Confidence Score:** {round(confidence_score*100,2)}%")