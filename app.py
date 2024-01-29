import gradio as gr
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

def mask_detector(img):
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)
    # Load the model
    model = load_model("keras_model.h5", compile=False)

    # Load the labels
    class_names = open("labels.txt", "r").readlines()

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    # image = Image.open().convert("RGB")
    image = img
  
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
    # print("Class:", class_name[2:], end="")
    # print("Confidence Score:", confidence_score)
    return(img,class_name[2:],f"{round(confidence_score*100,2)}%")


# interface =gr.Interface(fn= mask_detector, inputs=gr.Image(sources=['webcam'],type='pil'),outputs=[gr.Textbox(),gr.Textbox()])
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            img_in = gr.Image(type='pil',label='input',sources=['webcam'],streaming=True)
            btn = gr.Button(value='Capture')

        with gr.Column():
            out_img = gr.Image(width=200)
            out1 = gr.Textbox(label='Class')
            out2 = gr.Textbox(label='Confidence')
            

    btn.click(mask_detector,inputs=img_in, outputs=[out_img,out1,out2,])

demo.launch()