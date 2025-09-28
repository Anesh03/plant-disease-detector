import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import io
# Explicitly import Image and ImageOps from PIL
from PIL import Image as PILImage 
import streamlit as st
import pandas as pd
# Explicitly import Keras components from tf.keras
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from matplotlib.lines import Line2D
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase 

# Set a professional plotting style for the chart
# We'll customize the colors later for the dark theme
plt.style.use('seaborn-v0_8-darkgrid')

# --- Configuration (No change) ---
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
] 
IMAGE_SIZE = (224, 224)
MIN_CONFIDENCE_THRESHOLD = 70.0 
MAX_TOP_DIFFERENCE = 10.0 
MODEL_PATH = 'best_model.keras'

# --- Custom Green/Black Colors & Professional Text Color ---
PRIMARY_COLOR = '#006400'  # Dark Green (Used for backgrounds, like the diagnosis box)
SECONDARY_COLOR = '#90EE90' # Light Green (Used for accents/borders/headers)
BACKGROUND_COLOR = '#000000' # BLACK background
TEXT_COLOR = '#F0F0F0'      # Off-White/Light Gray text for contrast
ACCENT_COLOR = '#4CAF50'    # A moderate, clean green

# --- Validation/Core Functions (No change) ---

def is_valid_plant_photo(img_buffer, min_pixels=50000):
    try:
        img_stream = io.BytesIO(img_buffer)
        img = PILImage.open(img_stream)
        
        width, height = img.size
        
        if (width * height) < min_pixels:
            return False, "Image resolution is too low."
        
        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio > 3.5: 
             return False, "Image aspect ratio is unusual (too long/thin)."
             
        return True, None
    except Exception as e:
        return False, f"Could not read image file. Error: {e}"

def get_plant_name(class_name):
    return class_name.split('___')[0]

def is_consistent_prediction(all_predictions, confidence):
    if confidence >= 90.0:
        return True, None 

    top_idx = np.argsort(all_predictions)[-2:][::-1]
    
    top1_conf = all_predictions[top_idx[0]] * 100
    top2_conf = all_predictions[top_idx[1]] * 100
    
    top1_class = CLASS_NAMES[top_idx[0]]
    top2_class = CLASS_NAMES[top_idx[1]]

    top1_plant = get_plant_name(top1_class)
    top2_plant = get_plant_name(top2_class)

    if top1_conf - top2_conf < MAX_TOP_DIFFERENCE and top1_conf < 90.0:
        return False, "Prediction scores are too close, suggesting model confusion."
        
    if top1_conf < 85.0 and top1_plant != top2_plant:
        return False, f"Top predictions ({top1_plant} vs {top2_plant}) are conflicting."

    return True, None

@st.cache_resource
def load_classification_model(model_path):
    if not os.path.exists(model_path):
        return None
        
    try:
        model = load_model(model_path, compile=False) 

        for layer in model.layers[-30:]:
            layer.trainable = True

        model.compile(
            optimizer=Adam(learning_rate=1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    except Exception as e:
        return None

def preprocess_image(img_path, target_size=IMAGE_SIZE):
    # Use explicit tf.keras methods
    img = tf.keras.utils.load_img(img_path, target_size=target_size)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.keras.applications.mobilenet_v3.preprocess_input(img_array) 
    return np.expand_dims(img_array, axis=0)

def predict_image(model, img_path):
    try:
        processed_img = preprocess_image(img_path)
        predictions = model.predict(processed_img, verbose=0)
        all_predictions = predictions[0]
        pred_class_idx = np.argmax(all_predictions)
        confidence = np.max(all_predictions) * 100
        return CLASS_NAMES[pred_class_idx], confidence, all_predictions
    except Exception as e:
        return None, None, None

# --- Streamlit Web App Interface ---

def main():
    st.set_page_config(
        page_title="AI Plant Disease Diagnostic 🌿",
        layout="centered",
        initial_sidebar_state="auto",
    )

    # 1. ADD CUSTOM CSS FOR DARK MODE
    st.markdown(
        f"""
        <style>
            /* Apply a professional, sans-serif font like Google uses */
            @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
            
            /* Main Content Styling - BLACK BACKGROUND */
            .stApp {{
                background-color: {BACKGROUND_COLOR};
                font-family: 'Roboto', sans-serif;
                color: {TEXT_COLOR}; /* OFF-WHITE/LIGHT TEXT */
            }}
            /* All text should inherit the dark mode color */
            body, p, label, .stMarkdown, .stText, .stAlert, div[data-testid*="stMarkdownContainer"] {{
                color: {TEXT_COLOR} !important;
                font-weight: 400;
            }}
            /* Sidebar is often darker by default, but we ensure its text is light */
            .css-e37064, .css-1d3w5wq {{ /* Specific Streamlit classes for sidebar/main container */
                color: {TEXT_COLOR} !important;
            }}
            
            /* Title/Header Styling - Use bright green on black */
            h1 {{
                color: {SECONDARY_COLOR}; /* Light Green */
                text-align: center;
                font-size: 2.5em;
                padding-bottom: 0.5em;
                border-bottom: 3px solid {ACCENT_COLOR};
                font-weight: 700; 
            }}
            h2, h3, h4 {{
                color: {SECONDARY_COLOR}; /* Light Green */
                font-weight: 500;
            }}
            /* Streamlit Blocks (Success, Error, Info) - Dark backgrounds, light text */
            .stSuccess > div {{
                background-color: #003300; /* Very Dark Green */
                border-left: 5px solid {ACCENT_COLOR};
                color: {TEXT_COLOR}; /* Light Text */
            }}
            .stError > div {{
                background-color: #330000; /* Dark Red */
                border-left: 5px solid #FF4B4B;
                color: {TEXT_COLOR}; /* Light Text */
            }}
            .stWarning > div {{
                background-color: #333300; /* Dark Yellow */
                border-left: 5px solid #FFCD00;
                color: {TEXT_COLOR}; /* Light Text */
            }}
            
            .stRadio > label {{
                color: {SECONDARY_COLOR} !important; /* Emphasize radio button labels */
            }}
            .stRadio > div {{
                background-color: #111111; /* Dark Gray for inputs */
                padding: 10px;
                border-radius: 10px;
                border: 1px solid {PRIMARY_COLOR}; /* Dark Green border */
            }}
            /* Code Block Styling */
            .stCode pre {{
                background-color: #111111 !important;
                color: {TEXT_COLOR} !important;
            }}
            .stCode blockquote {{
                border-left: 5px solid {SECONDARY_COLOR};
                color: {TEXT_COLOR} !important;
            }}
            /* Prediction Banner Styling - Uses DARK GREEN background (PRIMARY) but must keep WHITE text */
            .diagnosis-container {{
                background-color: {PRIMARY_COLOR};
                color: white;
                padding: 20px;
                border-radius: 10px;
                margin-top: 15px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.4);
            }}
            .diagnosis-container h2, .diagnosis-container h3 {{
                color: white; /* Ensure white text on dark green */
                font-weight: 700;
            }}
            /* Info Box - Darker background, light text */
            .stAlert.info {{
                background-color: #0A140A; 
                border-left: 5px solid {ACCENT_COLOR};
                color: {TEXT_COLOR};
            }}
            /* Custom class for bold white text in warnings/errors */
            .bold-white-text {{
                color: white !important; /* Force white text */
                font-weight: 700 !important;
                font-size: 1.1em;
            }}

        </style>
        """,
        unsafe_allow_html=True
    )

    # --- UI Elements ---
    
    st.title("🔬 AI-Powered Plant Disease Diagnostic Dashboard 🌿")
    st.markdown("---")
    # Instruction text
    st.markdown(f'<div style="text-align: center; color: {TEXT_COLOR}; font-size: 1.1em; font-weight: 400;">Upload a **clear, focused photo** of the affected plant leaf for immediate AI analysis.</div>', unsafe_allow_html=True)
    st.markdown("---")


    # 1. Load Model
    model = load_classification_model(MODEL_PATH)
    
    if model is None:
        st.error(f"❌ CRITICAL ERROR: Could not load the diagnostic model from `{MODEL_PATH}`. Ensure the file is present.")
        st.stop() 

    st.success("✅ Model loaded successfully and ready for prediction!")

    # 2. User Input Choice
    st.subheader("🖼️ Image Input Source")
    source_choice = st.radio("Choose image source:", ('Upload File', 'Take Photo with Camera'), horizontal=True)
    uploaded_file = None
    
    if source_choice == 'Upload File':
        uploaded_file = st.file_uploader("Upload a Plant Leaf Image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    
    elif source_choice == 'Take Photo with Camera':
        uploaded_file = st.camera_input("Click 'Take Photo' to capture a leaf image.")


    # 3. Process and Predict
    if uploaded_file is not None:
        
        uploaded_file.seek(0)
        is_valid_sanity, validation_msg = is_valid_plant_photo(uploaded_file.getvalue())

        # The bold white text instruction template
        bold_white_instruction = f'<p class="bold-white-text">Please upload a clear, focused photo of the leaf or plant area.</p>'

        if not is_valid_sanity:
            st.error(f"🚫 Invalid Photo (Quality Check): {validation_msg}")
            # The instruction is displayed using a custom white bold class.
            st.markdown(bold_white_instruction, unsafe_allow_html=True) 
            return

        st.markdown("---")
        st.subheader("📸 Image for Analysis")
        st.image(uploaded_file, caption='Image for Analysis', use_container_width=True) 
        st.markdown("---")

        with st.spinner('🔍 Analyzing image and making a diagnosis...'):
            temp_img_path = os.path.join(os.getcwd(), "temp_upload.jpg")
            with open(temp_img_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            disease, confidence, all_predictions = predict_image(model, temp_img_path)
            
            os.remove(temp_img_path)

        # 4. Display Results
        if disease:
            
            # --- STRICT RESTRICTION LOGIC ---
            
            if confidence < MIN_CONFIDENCE_THRESHOLD:
                 st.error("⚠️ Photo Not Valid (Low Confidence)")
                 st.warning(f"The model's highest confidence prediction was only {confidence:.2f}%. This is below the required **{MIN_CONFIDENCE_THRESHOLD:.0f}%** threshold, highly suggesting the photo is **not a valid plant leaf** or is too poor quality.")
                 st.markdown(bold_white_instruction, unsafe_allow_html=True) # Display bold white text
                 return
                 
            is_consistent, consistency_msg = is_consistent_prediction(all_predictions, confidence)
            
            if not is_consistent:
                 st.error("⚠️ Photo Not Valid (Inconsistent Features)")
                 st.warning(f"The model's top predictions are highly confused ({consistency_msg}). This strongly suggests the photo is **not a valid plant leaf** from the training set.")
                 st.markdown(bold_white_instruction, unsafe_allow_html=True) # Display bold white text
                 return
             
            # --- If valid, proceed to display professional results ---
            clean_disease = disease.replace('___', ' ').replace('_', ' ').title()
            
            # A. Primary Diagnosis Section (Styled with custom CSS)
            st.markdown(
                f"""
                <div class="diagnosis-container">
                    <h2>🌿 Diagnosis Confirmed!</h2>
                    <h3>Primary Diagnosis: <strong>{clean_disease}</strong></h3>
                    <h3>Confidence: <strong>{confidence:.2f}%</strong></h3>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown("---")

            # B. Get Top 5 Predictions Data
            top_k = 5
            top_idx = np.argsort(all_predictions)[-top_k:][::-1]
            top_confidences = [all_predictions[i] * 100 for i in top_idx]
            
            top_names_detailed = []
            for i in top_idx:
                 parts = CLASS_NAMES[i].split('___')
                 plant = parts[0].replace('_', ' ').title()
                 condition = parts[-1].replace('_', ' ').title()
                 top_names_detailed.append(f"{plant} ({condition})")
            
            
            # C. Detailed Top Predictions (Text Block)
            st.subheader("🏆 Detailed Top Predictions:")
            
            text_output = "\n".join([
                 f"  {i+1}. {top_names_detailed[i]:<60} [{top_confidences[i]:.2f}%]"
                 for i in range(top_k)
            ])
            st.code(text_output, language='text')
            st.markdown("---")


            # D. Visualization (Top 5 Bar Chart)
            st.subheader("📊 Confidence Distribution Visualization (Top 5)")
            
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Explicitly set the Matplotlib figure background for dark mode
            fig.patch.set_facecolor(BACKGROUND_COLOR)
            ax.set_facecolor('#111111') # Dark gray plot area

            bar_colors = [PRIMARY_COLOR] + [SECONDARY_COLOR] * (top_k - 1) 

            ax.barh(top_names_detailed, top_confidences, color=bar_colors, height=0.7)
            ax.invert_yaxis()
            
            # Set all Matplotlib text to the light theme color
            ax.set_title("Top 5 Prediction Confidence", fontsize=16, pad=15, fontweight='bold', color=TEXT_COLOR)
            ax.set_xlabel("Confidence Score (%)", fontsize=12, color=TEXT_COLOR)
            ax.set_xlim(0, 100)
            
            plt.xticks(color=TEXT_COLOR)
            plt.yticks(color=TEXT_COLOR)
            # Set grid/spines to a dark accent color
            ax.tick_params(axis='x', colors=TEXT_COLOR)
            ax.tick_params(axis='y', colors=TEXT_COLOR)
            ax.spines['left'].set_color(SECONDARY_COLOR)
            ax.spines['bottom'].set_color(SECONDARY_COLOR)
            
            for index, value in enumerate(top_confidences):
                 # White text on the primary green bar, light gray text on the other bars
                 text_color = 'white' if index == 0 else TEXT_COLOR
                 ax.text(value - 4.5, index, f'{value:.2f}%', va='center', color=text_color, fontweight='bold')

            st.pyplot(fig) 
            plt.close(fig) 
            
            st.markdown("---")
            st.info("💡 Recommendation: The model shows high confidence in the primary diagnosis. For treatment options, consult a local agricultural extension specialist.")

        else:
            st.error("🚫 Analysis failed. Check the image quality, file format, or model setup.")

if __name__ == '__main__':
    main()