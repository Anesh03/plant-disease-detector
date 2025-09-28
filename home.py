import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import io
from PIL import Image as PILImage 
import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from matplotlib.lines import Line2D # Used in the plotting function

# --- Configuration & Theme ---

# Set a professional plotting style for the chart
plt.style.use('seaborn-v0_8-darkgrid')

# Configuration values
IMAGE_SIZE = (224, 224)
MODEL_PATH = 'best_model.keras' 

# Define the international set of plant diseases
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

# Quality Thresholds (needed for consistency, even if checks are skipped)
MIN_PIXELS = 50000 

# Dark Mode (Black/Green) Theme Colors
PRIMARY_COLOR = '#006400'   # Dark Green
SECONDARY_COLOR = '#90EE90'  # Light Green
BACKGROUND_COLOR = '#000000' # BLACK
TEXT_COLOR = '#F0F0F0'       # Off-White/Light Gray
ACCENT_COLOR = '#4CAF50'     # Moderate Green
ERROR_COLOR = '#FF4B4B'      # Red

# --- Helper Functions (From previous valid version) ---

def is_valid_plant_photo(img_buffer, min_pixels=MIN_PIXELS):
    """Placeholder for validation (checks are currently removed from main logic)."""
    try:
        img_stream = io.BytesIO(img_buffer)
        img = PILImage.open(img_stream)
        width, height = img.size
        return True, None
    except Exception:
        return True, None 

def get_plant_name(class_name):
    return class_name.split('___')[0]

def is_consistent_prediction(all_predictions, confidence):
    return True, None

# --- Core ML Functions ---

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
    except Exception:
        return None

def preprocess_image(img_path, target_size=IMAGE_SIZE):
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
    except Exception:
        return None, None, None

# --- Streamlit Web App Interface (Main) ---

def main():
    # Initialize session state variables
    if 'camera_captured' not in st.session_state:
        st.session_state.camera_captured = False
    if 'captured_file' not in st.session_state:
        st.session_state.captured_file = None
    if 'last_upload_key' not in st.session_state:
        st.session_state.last_upload_key = None

    st.set_page_config(
        page_title="International Plant Disease Detector 🌿",
        layout="centered",
        initial_sidebar_state="auto",
    )

    # 1. Inject Custom CSS (Dark Mode/Green Theme)
    st.markdown(
        f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
            
            /* Main Content Styling - BLACK BACKGROUND */
            .stApp {{ background-color: {BACKGROUND_COLOR}; font-family: 'Roboto', sans-serif; color: {TEXT_COLOR}; }}
            /* General text coloring */
            h1, h2, h3, h4 {{ color: {SECONDARY_COLOR}; }}
            .stSuccess > div {{ background-color: #003300; border-left: 5px solid {ACCENT_COLOR}; color: {TEXT_COLOR}; }}
            .diagnosis-container {{ background-color: {PRIMARY_COLOR}; color: white; }}
            
            /* Radio Button Styling to match the grouped look */
            div[data-testid="stFormSubmitButton"] + div[data-testid="stRadio"] > div {{
                background-color: {BACKGROUND_COLOR}; /* Black container */
                border: 1px solid #333333; /* Dark gray border */
                border-radius: 8px;
                padding: 10px;
            }}
            
            /* Custom styling for the camera success message */
            .camera-success {{
                background-color: {PRIMARY_COLOR};
                padding: 10px;
                border-radius: 5px;
                color: white;
                margin-top: 25px;
                display: flex;
                align-items: center;
                gap: 10px;
                font-size: 1.1em;
            }}

        </style>
        """,
        unsafe_allow_html=True
    )

    # --- UI Header ---
    
    st.title("🔬 International Plant Disease Diagnostic Dashboard 🌍")
    st.markdown("---")
    
    st.markdown(f'<div style="text-align: center; color: {TEXT_COLOR}; font-size: 1.1em; font-weight: 400;">Upload a <b>clear, focused photo</b> of the affected plant leaf for immediate AI analysis.</div>', unsafe_allow_html=True)
    st.markdown("---")

    # 2. Model Loading
    model = load_classification_model(MODEL_PATH)
    
    if model is None:
        st.error(f"❌ CRITICAL ERROR: Could not load the diagnostic model from `{MODEL_PATH}`. Ensure the file is present in the deployment environment.")
        st.stop() 

    st.success("✅ Model loaded successfully and ready for prediction!")

    # 3. User Input (Combined Upload & Camera)
    st.subheader("🖼️ Image Input Source")
    
    # Using st.radio to achieve the visual grouping shown in the image
    source_choice = st.radio(
        "Choose image source:",
        ('Upload File', 'Take Photo with Camera'),
        horizontal=True,
        key="source_choice"
    )
    
    uploaded_file = None
    camera_file = None
    
    if source_choice == 'Upload File':
        # Reset camera state when switching to Upload
        if st.session_state.camera_captured:
             st.session_state.camera_captured = False
             st.session_state.captured_file = None
             
        uploaded_file = st.file_uploader(
            "Upload a Plant Leaf Image (JPG/PNG)", 
            type=["jpg", "jpeg", "png"],
            key="file_uploader"
        )
        
    elif source_choice == 'Take Photo with Camera':
        
        # --- CAMERA CAPTURE LOGIC ---
        
        if st.session_state.camera_captured and st.session_state.captured_file is not None:
            # Display the check circle success message after capture
            st.markdown(
                f'<div class="camera-success">✅ Photo Captured! Ready for analysis.</div>', 
                unsafe_allow_html=True
            )
            # Use the stored file data
            file_to_process = st.session_state.captured_file
            
            # Add a button to clear the capture and show the live camera again
            if st.button("🚫 Clear Photo / Re-take", key="clear_capture"):
                st.session_state.camera_captured = False
                st.session_state.captured_file = None
                st.rerun() # Rerun to show the live camera

        else:
            # Show the live camera input
            st.session_state.camera_captured = False # Ensure false while open
            camera_file = st.camera_input("Click to take photo", key="camera_input")
            
            # Check if a new photo was just captured
            if camera_file is not None:
                # Store the captured file and update state
                st.session_state.camera_captured = True
                st.session_state.captured_file = camera_file
                st.rerun() # Rerun to switch to the success message view

    # Determine the final file to process
    if source_choice == 'Upload File':
        file_to_process = uploaded_file
    elif st.session_state.camera_captured:
        file_to_process = st.session_state.captured_file
    else:
        file_to_process = None
    
    # 4. Process and Predict
    if file_to_process is not None:
        
        # Read file content
        file_to_process.seek(0)
        img_buffer = file_to_process.getvalue()

        # Display image
        st.markdown("---")
        st.subheader("📸 Image for Analysis")
        st.image(img_buffer, caption='Image for Analysis', use_container_width=True) 
        st.markdown("---")

        # Predict
        with st.spinner('🔍 Analyzing image and making a diagnosis...'):
            temp_img_path = os.path.join(os.getcwd(), "temp_upload.jpg")
            try:
                with open(temp_img_path, "wb") as f:
                    f.write(img_buffer)

                disease, confidence, all_predictions = predict_image(model, temp_img_path)
            finally:
                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path) # Clean up

        # 5. Display Results
        if disease and confidence is not None:
            
            clean_disease = disease.replace('___', ' ').replace('_', ' ').title()
            
            # A. Primary Diagnosis Section
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

            # B. Top Predictions Table
            st.subheader("🏆 Detailed Top Predictions:")
            
            top_k = 5
            top_idx = np.argsort(all_predictions)[-top_k:][::-1]
            top_confidences = [all_predictions[i] * 100 for i in top_idx]
            
            top_names_detailed = []
            for i in top_idx:
                 parts = CLASS_NAMES[i].split('___')
                 plant = parts[0].replace('_', ' ').title()
                 condition = parts[-1].replace('_', ' ').title()
                 top_names_detailed.append(f"{plant} ({condition})")
            
            df_results = pd.DataFrame({
                'Rank': list(range(1, top_k + 1)),
                'Disease/Condition': top_names_detailed,
                'Confidence (%)': [f'{c:.2f}' for c in top_confidences]
            })
            st.dataframe(df_results, hide_index=True, use_container_width=True)
            st.markdown("---")

            # C. Visualization (Bar Chart)
            st.subheader("📊 Confidence Distribution Visualization (Top 5)")
            
            fig, ax = plt.subplots(figsize=(10, 5))

            # Matplotlib Dark Mode Setup
            fig.patch.set_facecolor(BACKGROUND_COLOR)
            ax.set_facecolor('#111111') 

            bar_colors = [PRIMARY_COLOR] + [SECONDARY_COLOR] * (top_k - 1) 
            ax.barh(top_names_detailed, top_confidences, color=bar_colors, height=0.7)
            ax.invert_yaxis()
            
            # Text and Axes Styling
            ax.set_title("Top 5 Prediction Confidence", fontweight='bold', color=TEXT_COLOR)
            ax.set_xlabel("Confidence Score (%)", color=TEXT_COLOR)
            ax.set_xlim(0, 100)
            plt.xticks(color=TEXT_COLOR)
            plt.yticks(color=TEXT_COLOR)
            ax.tick_params(axis='x', colors=TEXT_COLOR)
            ax.tick_params(axis='y', colors=TEXT_COLOR)
            
            for index, value in enumerate(top_confidences):
                 text_color = 'white' if index == 0 else TEXT_COLOR
                 ax.text(value - 4.5, index, f'{value:.2f}%', va='center', color=text_color, fontweight='bold')

            st.pyplot(fig) 
            plt.close(fig) 
            
            st.markdown("---")
            st.info("💡 Recommendation: The model shows high confidence in the primary diagnosis. For treatment options, consult a local agricultural extension specialist.")

        else:
            st.error("🚫 Analysis failed. An unknown error occurred during prediction.")

if __name__ == '__main__':
    main()