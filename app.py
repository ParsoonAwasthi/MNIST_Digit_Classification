import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

# Define Model Architecture

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(32,64,kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
                
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(64*5*5,128),
            nn.ReLU(),

            nn.Linear(128,10)
        )

    def forward(self,x):
        x = self.conv_layers(x)
        x = x.view(x.size(0),-1)
        x = self.fc_layers(x)
        return x

 # Load the model

@st.cache_resource
def load_model():
    model = CNN() # Instantiate your model class
    # Load the saved weights
    model = torch.load('mnist_model.pth', map_location=torch.device('cpu'))
    model.eval() # Set model to evaluation mode
    return model

model = load_model()

# Build Streamlit UI

st.title("PyTorch MNIST Digit Recognizer ✍️")
st.write("Draw a digit (0-9) in the box below and the model will guess it!")

# Create a canvas component
canvas_result = st_canvas(
    fill_color="black",  
    stroke_width=20,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button('Predict'):
    if canvas_result.image_data is not None:
        # 1. Get the numpy array from the canvas (RGBA)
        img_array = canvas_result.image_data.astype('uint8')
        
        # 2. Convert to a PIL Image (Grayscale)
        img_pil = Image.fromarray(img_array, 'RGBA').convert('L')
        
        # 3. Apply PyTorch transformations (Resize to 28x28 and convert to Tensor)
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(), # Automatically scales pixels to [0.0, 1.0]
            # Optional: Add transforms.Normalize if you used it during training
            # transforms.Normalize((0.1307,), (0.3081,)) 
        ])
        
        # Add batch dimension [1, 1, 28, 28]
        input_tensor = transform(img_pil).unsqueeze(0) 
        
        # 4. Make Prediction
        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.argmax(output, dim=1).item()
            
        st.success(f"**Predicted Digit:** {prediction}")
