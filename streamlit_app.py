import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path

st.set_page_config(page_title='AgriAssist', layout='wide')

MODEL_PATH = Path('results/initial_model.pt')

ADVISORIES = {
    'leaf_spot': {
        'english': 'Remove infected leaves and avoid overhead irrigation.',
        'hindi': 'संक्रमित पत्तियों को हटाएं और ऊपर से सिंचाई से बचें।',
        'telugu': 'సంక్రమిత ఆకులను తొలగించి పై నుండి నీరు పోయడం నివారించండి.'
    },
    'leaf_blight': {
        'english': 'Use disease-resistant seeds and apply fungicide carefully.',
        'hindi': 'रोग प्रतिरोधी बीजों का उपयोग करें और सावधानी से फफूंदनाशक डालें।',
        'telugu': 'రోగ నిరోధక విత్తనాలు ఉపయోగించి జాగ్రత్తగా ఫంగిసైడ్ వాడండి.'
    },
    'healthy_leaf': {
        'english': 'Crop appears healthy. Continue regular monitoring.',
        'hindi': 'फसल स्वस्थ दिख रही है। नियमित निगरानी जारी रखें।',
        'telugu': 'పంట ఆరోగ్యంగా ఉంది. సాధారణ పర్యవేక్షణ కొనసాగించండి.'
    }
}

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])


class SmallCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(32 * 16 * 16, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)


@st.cache_resource

def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    label_to_idx = checkpoint['label_to_idx']
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    model = SmallCNN(len(label_to_idx))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, idx_to_label


st.title('AgriAssist - AI Crop Disease Diagnosis')
st.write('Upload a crop leaf image for disease prediction and localized advisory.')

language = st.selectbox('Select advisory language', ['english', 'hindi', 'telugu'])
uploaded = st.file_uploader('Upload crop image', type=['jpg', 'jpeg', 'png'])

if uploaded:
    image = Image.open(uploaded).convert('RGB')
    st.image(image, caption='Uploaded Image', width=300)

    if MODEL_PATH.exists():
        model, idx_to_label = load_model()
        x = transform(image).unsqueeze(0)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1).item()
            confidence = probs[0][pred].item()

        label = idx_to_label[pred]

        st.subheader(f'Prediction: {label}')
        st.write(f'Confidence: {confidence:.2f}')
        st.success(ADVISORIES[label][language])

        st.warning('AI-generated guidance should be verified with local agricultural experts before chemical usage.')
    else:
        st.error('Model file not found. Train the model first.')
