import torch
import torchvision
from torchvision import transforms
from PIL import Image

# Load ResNet34 model
model = torchvision.models.resnet34(weights='IMAGENET1K_V1')
model.eval()

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize adversarial patch
patch_size = (50, 50)
ad_patch = torch.rand(3, *patch_size, requires_grad=True)

# Load ImageNet classes
with open('imagenet_classes.txt') as f:
    class_labels = [line.strip() for line in f.readlines()]

# Set target class (Pomeranian, 260)
target_class_idx = 260

# Load and preprocess image
image = preprocess(Image.open('Dalgom.jpg')).unsqueeze(0)

# Optimizer
optimizer = torch.optim.Adam([ad_patch], lr=0.01)

# Training loop
for _ in range(100):
    optimizer.zero_grad()
    patched_image = image.clone()
    patched_image[:, :, :patch_size[0], :patch_size[1]] = ad_patch
    output = model(patched_image)
    loss = -output[0, target_class_idx]
    loss.backward()
    optimizer.step()

# Get the predicted class
_, predicted_class_idx = torch.max(output, 1)
predicted_class_idx = predicted_class_idx.item()  # Convert tensor to integer

# Print predicted class label
print(f'Predicted class: {class_labels[predicted_class_idx]}')

# Top 5 predictions with probabilities
softmax_probs = torch.nn.functional.softmax(output[0], dim=0)
top5_probs, top5_class_indices = torch.topk(softmax_probs, 5)

print("Top 5 predicted classes:")
for i in range(5):
    print(f"{class_labels[top5_class_indices[i]]}: {top5_probs[i].item() * 100:.2f}%")
