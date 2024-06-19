import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTImageProcessor
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()

train_directory = './dataset/train'
test_directory = './dataset/test'

# 하이퍼파라미터 설정
img_size = 224
num_classes = 2
batch_size = 32
learning_rate = 1e-4
num_epochs = 10
image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
])

test_dataset = ImageFolder(root=test_directory, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Grad-CAM 클래스 정의
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        target_layer = dict(self.model.named_modules())[self.target_layer]
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def generate(self, input_image, target_class, criterion, device):
        self.model.eval()
        input_image = input_image.unsqueeze(0).to(device)
        
        # Forward pass
        output = self.model(input_image).logits
        loss = criterion(output, torch.tensor([target_class]).to(device))
        
        # Backward pass
        self.model.zero_grad()
        loss.backward()

        gradients = self.gradients.detach().cpu().numpy()
        activations = self.activations.detach().cpu().numpy()

        if gradients.ndim == 4:  # Ensure gradients and activations are 4D tensors
            gradients = gradients[0]
            activations = activations[0]
        elif gradients.ndim == 3:  # Handle the case where they are 3D tensors
            pass
        else:
            raise ValueError("Unexpected gradients dimension")

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_image.shape[2], input_image.shape[3]))
        cam -= np.min(cam)
        cam /= np.max(cam)

        return cam

    def visualize(self, input_image, cam, alpha=0.5):
        input_image_np = input_image.detach().cpu().numpy().transpose(1, 2, 0)
        input_image_np = (input_image_np - np.min(input_image_np)) / (np.max(input_image_np) - np.min(input_image_np))

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam_image = heatmap + np.float32(input_image_np)
        cam_image = cam_image / np.max(cam_image)

        # Find the hottest area
        hot_x, hot_y = np.unravel_index(np.argmax(cam), cam.shape)

        # Resize coordinates to match original image size
        hot_x = int(hot_x * input_image_np.shape[0] / cam.shape[0])
        hot_y = int(hot_y * input_image_np.shape[1] / cam.shape[1])

        # Draw a rectangle around the hottest area
        cv2.rectangle(cam_image, (hot_y - 5, hot_x - 5), (hot_y + 5, hot_x + 5), (1, 0, 0), 2)

        plt.imshow(cam_image)
        plt.axis('off')
        plt.show()
# Grad-CAM 사용 예시
def predict_and_visualize(model, grad_cam, dataloader, criterion, device):
    model.eval()
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images).logits
        _, predicted = torch.max(outputs.data, 1)
        
        for i in range(images.size(0)):
            input_image = images[i].clone().detach()
            input_image.requires_grad = True
            cam = grad_cam.generate(input_image, predicted[i].item(), criterion, device)
            grad_cam.visualize(input_image, cam)

model_path = './models/model_epoch_7.pth'
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=num_classes, ignore_mismatched_sizes=True)
model.load_state_dict(torch.load(model_path))
model.to(device)

target_layer = 'vit.encoder.layer.11.output'
grad_cam = GradCAM(model, target_layer)

predict_and_visualize(model, grad_cam, test_loader, criterion, device)
