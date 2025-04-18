
import streamlit as st 
import torch
import torch.nn as nn
from torchvision import transforms 
import time 
from PIL import Image
import matplotlib.pyplot as plt
def load_model(model_path):
    class PlantDiseasesCNN(nn.Module):
         def __init__(self):
             super(PlantDiseasesCNN, self).__init__()
             self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
             self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
             self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
             self.pool_2 = nn.AvgPool2d(kernel_size=2, stride=2)
             self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
             self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dynamically calculate the input neurons
             self._to_linear = None
             self._get_conv_output()

             self.fc1 = nn.Linear(self._to_linear, 512)
             self.fc2 = nn.Linear(512, 256)
             self.fc3 = nn.Linear(256, 128)
             self.fc4 = nn.Linear(128, 64)
             self.fc5 = nn.Linear(64, 38)  
             self.dropout = nn.Dropout(0.2)
             self.relu = nn.ReLU()

         def _get_conv_output(self):
             dummy_input = torch.randn(1, 3, 224, 224) 
             out = self.conv1(dummy_input)
             out = self.pool_1(out)
             out = self.conv2(out)
             out = self.pool_2(out)
             out = self.conv3(out)
             out = self.pool_3(out)
             self._to_linear = out.numel() 

         def forward(self, x):
             x = self.pool_1(self.conv1(x))
             x = self.pool_2(self.conv2(x))
             x = self.pool_3(self.conv3(x))

             x = x.view(-1, self._to_linear)  
             x = self.relu(self.fc1(x))
             x = self.dropout(x)  
             x = self.relu(self.fc2(x))
             x = self.relu(self.fc3(x))
             x = self.relu(self.fc4(x))
             x = self.fc5(x)  
             return x
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PlantDiseasesCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Load model
model = load_model("plant_disease_model.pth")
    
                   
class_names={0:'Apple___Apple_scab',1:'Apple___Black_rot',2:'Apple___Cedar_apple_rust',3:'Apple___healthy',4:'Blueberry___healthy',5:'Cherry_(including_sour)___Powdery_mildew',
            6:'Cherry_(including_sour)___healthy',7:'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',8:'Corn_(maize)___Common_rust_',9:'Corn_(maize)___Northern_Leaf_Blight',
            10:'Corn_(maize)___healthy',11:'Grape___Black_rot',12:'Grape___Esca_(Black_Measles)',13:'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',14:'Grape___healthy',15:'Orange___Haunglongbing_(Citrus_greening)',
            16:'Peach___Bacterial_spot',17:'Peach___healthy',18:'Pepper,_bell___Bacterial_spot',19:'Pepper,_bell___healthy',20:'Potato___Early_blight',
            21:'Potato___healthy',22:'Potato___Late_blight',23:'Raspberry___healthy',24:'Soybean___healthy',25:'Squash___Powdery_mildew',
            26:'Strawberry___healthy',27:'Strawberry___Leaf_scorch',28:'Tomato___Bacterial_spot',29:'Tomato___Early_blight',30:'Tomato___healthy',31:'Tomato___Late_blight',
            32:'Tomato___Leaf_Mold',33:'Tomato___Septoria_leaf_spot',34:'Tomato___Spider_mites Two-spotted_spider_mite',35:'Tomato___Target_Spot',
            36:'Tomato___Tomato_Yellow_Leaf_Curl_Virus',37:'Tomato___Tomato_mosaic_virus'}
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])  

p = st.sidebar.radio('Plant Disease prediction', ['Plant Disease Detection', 'Graphical Representation of Models'])

if p == 'Plant Disease Detection':
    g = """
    <style>
    .st-emotion-cache-bm2z3a{
        background-image: url("https://media.istockphoto.com/id/1402801804/photo/closeup-nature-view-of-palms-and-monstera-and-fern-leaf-background.jpg?s=612x612&w=0&k=20&c=0pX8CbzsrqvMQKMa4853JRUw8oGy8NnsOC812H3o9Xo=");
        background-attachment: fixed;
        background-size: cover;
    }
    </style>
    """
    st.markdown(g, unsafe_allow_html=True)
    st.subheader(' Plant Disease Detection Web Application')
    st.write(" Upload a leaf image for prediction:")

    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        with st.spinner("Uploading and processing image..."):
            time.sleep(2)
        st.success(" Image has been uploaded for prediction.")
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        img=transform(image).unsqueeze(0)
        with st.spinner('predicting......'):
           with torch.no_grad():
              output=model(img)
              _,predicted=torch.max(output,1)
              prediction=class_names[predicted.item()]
        st.success(f"Predicted Disease: **{prediction}**")
elif p == 'Graphical Representation of Models':
    st.subheader(' Model Performance Comparison')
    model_names = ['CNN', 'ResNet18']
    accuracy = [83, 96]  
    fig,x = plt.subplots(figsize=(8,5))
    x.barh(model_names, accuracy, color=['green', 'skyblue'])
    x.set_xlabel('Accuracy (%)')
    x.set_title('Model Accuracy Comparison')
    st.pyplot(fig)

    st.markdown("**CNN** is performing very well, but pre-trained models like **ResNet18** achieve even higher accuracy.")
    