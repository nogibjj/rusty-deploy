import torch
from torchvision import transforms
from PIL import Image

# Load the model
model = torch.jit.load("model/resnet34.ot")

# Load and preprocess the image
image = Image.open("lion.jpg")
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])
image = transform(image)
image = image.unsqueeze(0)  # Add batch dimension

# Debug information
print("Input tensor shape:", image.shape)
print("Input tensor dtype:", image.dtype)

# Get input tensor as a numpy array
input_np = image.numpy()
print("Input tensor as array:")
print(input_np)

# Make a prediction
with torch.no_grad():
    output = model(image)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    _, predicted_class = torch.max(probabilities, 0)

# Debug information
print("Output tensor shape:", output.shape)
print("Output tensor dtype:", output.dtype)

# Print the predicted class
print("Predicted class:", predicted_class.item())
