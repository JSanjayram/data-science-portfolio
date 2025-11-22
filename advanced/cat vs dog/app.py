import json
import streamlit as st
from cat_dog_model import CatDogClassifier
import tempfile
import os
from PIL import Image
import requests
from io import BytesIO

# Initialize the classifier
@st.cache_resource
def load_model():
    classifier = CatDogClassifier(confidence_threshold=0.9)
    classifier.build_model()
    return classifier

def process_image(image, classifier):
    """Process image and return prediction results"""
    with st.spinner('Analyzing image...'):
        result = classifier.predict_image(image)
    return result

def main():
    # Enhanced UI/UX styling
    st.markdown("""
    <style>
    .stApp {
        background: radial-gradient(circle farthest-corner at -24.7% -47.3%, rgba(6,130,165,1) 0%, rgba(34,48,86,1) 66.8%, rgba(15,23,42,1) 100.2%) !important;
    }
    .main {
        background: transparent !important;
    }
    .block-container {
        background: transparent !important;
    }
    .stTitle { text-align: center; }
    .stMarkdown { text-align: center; }
    .stRadio > div { display: flex; justify-content: center; gap: 30px; align-items: center; flex-wrap: wrap; }
    .stRadio > div > label { background: rgba(0,0,0,0.9); color: white; padding: 15px 25px; border-radius: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); transition: all 0.3s; margin: 0 auto; }
    .stRadio > div > label:hover { transform: translateY(-2px); box-shadow: 0 6px 12px rgba(0,0,0,0.15); }
    .stRadio { text-align: center; }
    .stRadio > div > div { margin: 0 auto; }
    @media (max-width: 768px) {
        .stColumn { padding: 2px !important; min-width: 30% !important; flex: 1 1 30% !important; }
        .stButton button { font-size: 12px !important; padding: 5px 8px !important; }
        div[data-testid="column"] { min-width: 30% !important; flex: 1 1 30% !important; }
    }
    header[data-testid="stHeader"] {
        display: none;
    }
    .stDeployButton {
        display: none;
    }
    footer {
        display: none;
    }
    .stApp > footer {
        display: none;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Load animations if available
    dog_data = {}
    cat_data = {}
    
    try:
        with open('Happy Dog.json', 'r') as f:
            dog_data = json.load(f)
    except FileNotFoundError:
        pass
        
    try:
        with open('cute-cat (2).json', 'r') as f:
            cat_data = json.load(f)
    except FileNotFoundError:
        pass
    
    st.markdown("<h1 style='text-align: center;'> Cat or Dog Classifier</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 18px; margin-bottom: 30px;'>Upload an image or provide URL to classify if it's a cat or dog with 90%+ accuracy!</p>", unsafe_allow_html=True)
    
    classifier = load_model()
    
    if dog_data or cat_data:
        html = '<script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>'
        
        if dog_data:
            html += f'''
            <div style="display: flex; justify-content: center;">
                <lottie-player id="happy-dog" background="transparent" speed="1" 
                               style="width: 400px; height: 200px;" loop autoplay>
                </lottie-player>
            </div>
            '''
            
        if cat_data:
            html += f'''
            <div style="display: flex; justify-content: center; margin-top: -120px; margin-left: -60px;">
                <lottie-player id="cute-cat" background="transparent" speed="1" 
                               style="width: 200px; height: 130px;" loop autoplay>
                </lottie-player>
            </div>
            '''
            
        html += '<script>'
        if dog_data:
            html += f'document.getElementById("happy-dog").load({json.dumps(dog_data)});'
        if cat_data:
            html += f'document.getElementById("cute-cat").load({json.dumps(cat_data)});'
        html += '</script>'
        
        st.components.v1.html(html, height=200)
    else:
        st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)
    
    # Enhanced input method selection
    st.markdown("<h3 style='text-align: center; margin-bottom: 20px;'>Choose Input Method<br style='font-size:10px;'><span style='color: #FFD700;font-size:10px'>Note: This model is trained specifically to recognize animals found in Tamil Nadu.</span></br></h3>", unsafe_allow_html=True)
    
    # Center the radio buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        input_method = st.radio("", ["📁 Upload File", "🔗 Image URL"], horizontal=True)
        
        # Sample images section in 3x2 grid
        st.markdown("<p style='text-align: center; margin: 10px 0;'>Or try sample images:</p>", unsafe_allow_html=True)
        
        sample_images = {
            "Cat 1": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=200&h=200&fit=crop",
            "Cat 2": "https://images.unsplash.com/photo-1573865526739-10659fec78a5?w=200&h=200&fit=crop", 
            "Cat 3": "https://images.unsplash.com/photo-1592194996308-7b43878e84a6?w=200&h=200&fit=crop",
            "Dog 1": "https://images.unsplash.com/photo-1568572933382-74d440642117?w=200&h=200&fit=crop",
            "Dog 2": "https://images.unsplash.com/photo-1583337130417-3346a1be7dee?w=200&h=200&fit=crop",
            "Dog 3": "https://images.unsplash.com/photo-1587300003388-59208cc962cb?w=200&h=200&fit=crop"
        }
        
        selected_sample = None
        
        # 3x2 grid for sample images
        row1_col1, row1_col2, row1_col3 = st.columns(3)
        with row1_col1:
            st.image(sample_images["Cat 1"], use_column_width=True)
            if st.button("🐱 Cat 1", key="cat1", use_container_width=True):
                selected_sample = sample_images["Cat 1"]
        with row1_col2:
            st.image(sample_images["Cat 2"], use_column_width=True)
            if st.button("🐱 Cat 2", key="cat2", use_container_width=True):
                selected_sample = sample_images["Cat 2"]
        with row1_col3:
            st.image(sample_images["Cat 3"], use_column_width=True)
            if st.button("🐱 Cat 3", key="cat3", use_container_width=True):
                selected_sample = sample_images["Cat 3"]
        
        row2_col1, row2_col2, row2_col3 = st.columns(3)
        with row2_col1:
            st.image(sample_images["Dog 1"], use_column_width=True)
            if st.button("🐶 Dog 1", key="dog1", use_container_width=True):
                selected_sample = sample_images["Dog 1"]
        with row2_col2:
            st.image(sample_images["Dog 2"], use_column_width=True)
            if st.button("🐶 Dog 2", key="dog2", use_container_width=True):
                selected_sample = sample_images["Dog 2"]
        with row2_col3:
            st.image(sample_images["Dog 3"], use_column_width=True)
            if st.button("🐶 Dog 3", key="dog3", use_container_width=True):
                selected_sample = sample_images["Dog 3"]
    
    image = None
    
    # Handle sample image selection with session state
    if selected_sample:
        st.session_state.selected_image_url = selected_sample
    
    # Load image from session state if available
    if 'selected_image_url' in st.session_state and st.session_state.selected_image_url:
        try:
            response = requests.get(st.session_state.selected_image_url)
            image = Image.open(BytesIO(response.content))
        except Exception as e:
            st.error("Failed to load sample image")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if input_method == "📁 Upload File":
            uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
            if uploaded_file:
                image = Image.open(uploaded_file)
                
        else:  # Image URL
            url = st.text_input("Enter image URL:")
            if url:
                try:
                    response = requests.get(url)
                    image = Image.open(BytesIO(response.content))
                except Exception as e:
                    st.error("Failed to load image from URL")
        
        if image:
            st.image(image, caption="Input Image", width=400)
            
            # Load and predict with model
            if st.button("🔍 Classify Image", type="primary"):
                result = process_image(image, classifier)
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col2:
                    if result['prediction'] == 'Dog':
                        if result['confidence'] >= 0.9:
                            st.success(f"Hey it's **DOG** WOW WOW! ({result['confidence']:.1%} confidence)")
                        else:
                            st.warning(f"Hey it's **DOG** WOW WOW! ({result['confidence']:.1%} confidence)")
                    else:  # Cat
                        if result['confidence'] >= 0.9:
                            st.success(f"Hey it's **CAT** MEOW MEOW! ({result['confidence']:.1%} confidence)")
                        else:
                            st.warning(f"Hey it's **CAT** MEOW MEOW! ({result['confidence']:.1%} confidence)")
                
                # Show confidence breakdown
                if result['prediction'] == 'Dog':
                    st.subheader("🐶 Woof! Prediction Confidence")
                else:
                    st.subheader("🐱 Meow! Prediction Confidence")
                
                if result['prediction'] == 'Cat':
                    st.progress(result['confidence'], text=f"Cat: {result['confidence']:.1%}")
                    st.progress(1-result['confidence'], text=f"Dog: {1-result['confidence']:.1%}")
                else:
                    st.progress(1-result['confidence'], text=f"Cat: {1-result['confidence']:.1%}")
                    st.progress(result['confidence'], text=f"Dog: {result['confidence']:.1%}")

if __name__ == "__main__":
    main()