# ICAO_Passport-Photo_Compliance_Detector

# Description:

This AI model classifies passport photos as ICAO-compliant or non-compliant using binary image classification. Built with PyTorch, it leverages a ResNet50 backbone (pretrained on ImageNet) for robust feature extraction, fine-tuned with a custom classifier head. The pipeline includes data augmentation (rotations, shifts) to enhance generalization and GPU-accelerated training for efficiency.

# Tools & Technologies:

- Framework: PyTorch

- Backbone: ResNet50 (transfer learning)

 - Data Processing: TorchVision (transforms, ImageFolder, DataLoader)

- Training: Binary Cross-Entropy Loss, Adam Optimizer

- Hardware: CUDA GPU support (if available)

- Output: Saved model weights (.pth) for deployment

# Use Case:

Automate passport photo validation for government agencies, photo studios, or mobile apps, ensuring adherence to ICAO standards (e.g., lighting, pose, background).
