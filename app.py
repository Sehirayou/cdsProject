import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import os
import random
import glob
import re
from io import BytesIO
import cv2
import warnings
warnings.filterwarnings('ignore')
from PIL import Image
#import cv2  # Make sure to import cv2 if needed
import streamlit as st
import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

import tensorflow.keras as k
k.utils.set_random_seed(42) # idem keras

from keras.backend import manual_variable_initialization
manual_variable_initialization(True) # https://github.com/keras-team/keras/issues/4875#issuecomment-296696536

from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.applications.xception import Xception
import json


file_path = '.vscode/inputs/' # folder with files
Dis_percentage = pd.read_csv(os.path.join(file_path,'Spots_Percentage_results.csv'))
Details = pd.read_csv(os.path.join(file_path,'Plant_details.csv'))
# Domain Name Suggestion
domain_name = "cds.plantdiseasealert.com"
# Load the TensorFlow Lite model
model_path = '.vscode/model/model.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

with open('.vscode/inputs/0420_labels.json', 'r') as file:
    loaded_class_indices = {k: int(v) for k, v in json.load(file).items()}
    class_labels = {value: key for key, value in loaded_class_indices.items()} # Convert keys to int


# Identify extent of spot or lesion coverage on leaf
def identify_spots_or_lesions(cv_image):
    
    lab_image = cv2.cvtColor(np.array(cv_image), cv2.COLOR_BGR2Lab)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    blur = cv2.GaussianBlur(a_channel,(3,3),0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    # Morphological clean-up
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1) # Opening = erosion followed by dilation
    edges = cv2.Canny(cleaned,100,300)

    # Filter and contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 18000
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) < max_area]

    # Calculate the percentage of spots/lesions
    spot_pixels = sum(cv2.contourArea(cnt) for cnt in filtered_contours)
    total_pixels = edges.shape[0] * edges.shape[1]
    percentage_spots = (spot_pixels / total_pixels)*100
    print(f"Percentage of spots/lesions: {percentage_spots:.2f}%")

    # Draw filtered contours
    contoured_image = cv2.drawContours(cv_image.copy(), filtered_contours, -1, (0, 255, 0), 1)

    # Visualization
    plt.figure(figsize=(25, 8))
    plt.subplot(1, 5, 1)
    plt.imshow(cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB))
    plt.title('Original Image')

    plt.subplot(1, 5, 2)
    plt.imshow(a_channel, cmap='gray')
    plt.title('LAB - A channel')

    plt.subplot(1, 5, 3)
    plt.imshow(edges, cmap='gray')
    plt.title('Edge Detection')

    plt.subplot(1, 5, 4)
    plt.imshow(cleaned, cmap='gray')
    plt.title('Thresholded & Cleaned')

    plt.subplot(1, 5, 5)
    plt.imshow(cv2.cvtColor(contoured_image, cv2.COLOR_BGR2RGB))
    plt.title('Spots or Lesions Identified')
    plt.show()
    return(percentage_spots)

# Plot disease percentage
def plot_dis_percentage(row, percentage):
   # Determine the range category for the title
   if percentage < row['Q1']:
       category = 'Low'
       color = 'yellow'
   elif row['Q1'] <= percentage <= row['Q3']:
       category = 'Medium'
       color = 'orange'
   else:
       category = 'High'
       color = 'darkred'

   # Normalize the data to the range of [0, 1]
   min_val = row['min']
   max_val = row['max']
   range_val = max_val - min_val
   percentage_norm = (percentage - min_val) / range_val

   # Create a figure and a set of subplots
   fig, ax = plt.subplots(figsize=(6, 1))

   # Create the ranges for Low, Medium, and High
   ax.axhline(0, xmin=0, xmax=(row['Q1'] - min_val) / range_val, color='yellow', linewidth=4, label='Low')
   ax.axhline(0, xmin=(row['Q1'] - min_val) / range_val, xmax=(row['Q3'] - min_val) / range_val, color='orange', linewidth=4, label='Medium')
   ax.axhline(0, xmin=(row['Q3'] - min_val) / range_val, xmax=1, color='darkred', linewidth=4, label='High')

   # Plot the actual percentage as an arrow
   ax.annotate('', xy=(percentage_norm, 0.1), xytext=(percentage_norm, -0.1),
               arrowprops=dict(facecolor=color, shrink=0.05, width=1, headwidth=10))

   # Set display parameters
   ax.set_yticks([])  # No y-ticks
   ax.set_xticks([])  # Remove specific percentage figures from the x-axis
   ax.set_xlim([0, 1])  # Set x-limits to normalized range
   ax.set_title(f'{row["Plant"]} - {category}')
   ax.set_xlabel('Value (Normalized)')

   plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
   plt.tight_layout()
   st.pyplot(fig)
from PIL import Image

def resize_image(image, target_size=(224, 224)):
    return image.resize(target_size)

# Classify the image
def classify_image(image):
    # Preprocess the image as needed
    resized_image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)

    img_array = np.array(resized_image, dtype='float32')
    img_array = np.expand_dims(img_array, axis=0)

    # preprocess_input from Xception to scale the image to -1 to +1
    img_array = preprocess_input(img_array)

    # Perform inference using the TensorFlow Lite model
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_class_name = class_labels[predicted_class_index]  # direct integer index

    # Display predicted class name on the webpage
    st.write("Predicted class:", predicted_class_name)

    if "healthy" in predicted_class_name:
        st.write(f"{predicted_class_name} is healthy, skipping further analysis.")
        return

    else:
        spots_percentage = identify_spots_or_lesions(image)

        if predicted_class_name in Dis_percentage['Plant'].values:
            row = Dis_percentage.loc[Dis_percentage['Plant'] == predicted_class_name].iloc[0]
            plot_dis_percentage(row, spots_percentage)

            if predicted_class_name in Details['Plant'].values:
                row = Details.loc[Details['Plant'] == predicted_class_name].iloc[0]
                st.write("Disease Identification:", row[4])
                st.write("----------------------------------")
                st.write("Management:", row[5])
        else:
            st.write("No data available for this plant disease in DataFrame.")

        return predicted_class_name

    
# Streamlit app
st.title('Plant Disease Identification')
# Display Plant Care Icon
st.image(".vscode/inputs/plantIcon.jpg", width=100)
st.write("""
Plant diseases are a significant threat to agricultural productivity worldwide, causing substantial crop losses and economic damage. These diseases can be caused by various factors, including fungi, bacteria, viruses, and environmental stressors. Recognizing the symptoms of plant diseases early is crucial for implementing effective management strategies and minimizing the impact on crop yield and quality.
""")

# Importance of Early Detection
st.write("""
### Importance of Early Detection

Early detection of plant diseases is paramount for farmers to protect their crops and livelihoods. By identifying diseases at their onset, farmers can implement timely interventions, such as targeted pesticide applications or cultural practices, to prevent the spread of diseases and reduce crop losses. Early detection also reduces the need for excessive chemical inputs, promoting sustainable agriculture practices and environmental stewardship.
""")


# Types of Plant Diseases Detected
st.image(".vscode/inputs/Plant-disease-classifier-with-ai-blog-banner.jpg", width=700)





st.write("With more than 50% of the population in India still relying on agriculture and with the average farm sizes and incomes being very small, we believe that cost effective solutions for early detection and treatment solutions for disease could significantly improve the quality of produce and lives of farmers. With smartphones being ubiquitous, we believe providing solutions to farmers over a smartphone is the most penetrative form.")
st.write('### Plant Disease Prediction')
# Load and display the image
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"], key="uploader")
if uploaded_file is not None:
    
    print("Image successfully uploaded!")
    # Read the uploaded image file
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    image = Image.open(uploaded_file)
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # Perform classification
    result = classify_image(cv_image)
else:
    print("No file uploaded.")
# Disclaimer
st.write("""
### Disclaimer

While our disease identification system strives for accuracy and reliability, it is essential to note its limitations. False positives or false negatives may occur, and users are encouraged to consult with agricultural experts for professional advice and decision-making.
""")