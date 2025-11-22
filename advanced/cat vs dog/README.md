# Cat vs Dog Image Classifier

A minimal image recognition model that classifies images as cats or dogs with 90% confidence threshold.

## Features
- Uses pre-trained MobileNetV2 for high accuracy
- 90% confidence threshold requirement
- Web interface with Streamlit
- Handles JPG, JPEG, PNG images

## Installation
```bash
pip install -r requirements.txt
```

## Usage

### Web App
```bash
streamlit run app.py
```

### Python Script
```python
from cat_dog_model import CatDogClassifier

classifier = CatDogClassifier(confidence_threshold=0.9)
classifier.build_model()
result = classifier.predict_image("your_image.jpg")
print(result)
```

## Model Behavior
- **Confident**: Returns prediction when confidence ≥ 90%
- **Uncertain**: Returns "uncertain" when confidence < 90%