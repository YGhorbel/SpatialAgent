import os
import json
import torch
import numpy as np
from PIL import Image
import io
import boto3
from torchvision import transforms
from torchvision.transforms import functional as F
import pycocotools.mask as mask_utils
import torchvision.models as models
import torch.nn as nn

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Model Definitions (Copied from original repo for self-containment) ---

class ResNetDistanceRegressor(nn.Module):
    def __init__(self, input_channels=5, backbone='resnet50', pretrained=False):
        super().__init__()
        self.resnet = getattr(models, backbone)(weights=None)
        old_conv = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(input_channels, old_conv.out_channels,
                                      kernel_size=old_conv.kernel_size,
                                      stride=old_conv.stride,
                                      padding=old_conv.padding,
                                      bias=old_conv.bias is not None)
        num_feats = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_feats, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.resnet(x).squeeze(1)

class ResNet50Binary(nn.Module):
    def __init__(self, in_channels=5):
        super().__init__()
        self.resnet = models.resnet50(weights=None)
        self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)
        
    def forward(self, x):
        x = self.resnet(x)
        return x.squeeze(1)

# --- SageMaker Interface Functions ---

def model_fn(model_dir):
    """Load the model for inference."""
    print(f"Loading model from {model_dir}")
    
    # Determine which model to load based on file existence
    # We assume the user packages the specific .pth file for the endpoint
    
    # Check for distance model
    dist_path = os.path.join(model_dir, 'model.pth') # Generic name, rename before upload
    
    # Heuristic: Try to load as Distance model, if fails/logic needed, can use env vars
    # For simplicity, we assume we deploy separate endpoints and this script handles both 
    # if we know which class to instantiate. 
    # Ideally, pass a hyperparameter or env var. 
    # Here, we will try to load state_dict and infer or just default to Distance for now.
    # BETTER APPROACH: Use an environment variable 'MODEL_TYPE' set during deployment.
    
    model_type = os.environ.get('MODEL_TYPE', 'distance') # 'distance' or 'inside'
    
    if model_type == 'distance':
        model = ResNetDistanceRegressor(input_channels=5)
    else:
        model = ResNet50Binary(in_channels=5)
        
    with open(dist_path, 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=DEVICE))
    
    model.to(DEVICE)
    model.eval()
    return model

def input_fn(request_body, request_content_type):
    """Parse input data."""
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        return data
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Run inference."""
    # input_data expects:
    # {
    #   "image_s3_uri": "s3://...",
    #   "pairs": [
    #       {"mask1": {...RLE...}, "mask2": {...RLE...}},
    #       ...
    #   ]
    # }
    
    s3 = boto3.client('s3')
    
    # 1. Download Image
    s3_uri = input_data['image_s3_uri']
    bucket, key = s3_uri.replace("s3://", "").split("/", 1)
    
    img_bytes = io.BytesIO()
    s3.download_fileobj(bucket, key, img_bytes)
    img = Image.open(img_bytes).convert('RGB')
    
    # Resize logic (matches original tools.py)
    resize_shape = (360, 640) # H, W
    img = F.resize(img, resize_shape)
    rgb = np.array(img).astype(np.float32) / 255.0
    
    results = []
    
    for pair in input_data['pairs']:
        mask1_rle = pair['mask1']
        mask2_rle = pair['mask2']
        
        # Decode masks
        # Ensure RLE strings are bytes if needed by pycocotools
        if isinstance(mask1_rle['counts'], str):
             mask1_rle['counts'] = mask1_rle['counts'].encode('utf-8')
        if isinstance(mask2_rle['counts'], str):
             mask2_rle['counts'] = mask2_rle['counts'].encode('utf-8')

        mask1_arr = mask_utils.decode(mask1_rle).astype(np.float32)
        mask2_arr = mask_utils.decode(mask2_rle).astype(np.float32)
        
        mask1_img = Image.fromarray(mask1_arr)
        mask2_img = Image.fromarray(mask2_arr)
        
        mask1_img = F.resize(mask1_img, resize_shape, interpolation=Image.NEAREST)
        mask2_img = F.resize(mask2_img, resize_shape, interpolation=Image.NEAREST)
        
        mask1_resized = np.array(mask1_img).astype(np.float32)
        mask2_resized = np.array(mask2_img).astype(np.float32)
        
        # Stack
        components = [rgb, mask1_resized[..., None], mask2_resized[..., None]]
        input_tensor = np.concatenate(components, axis=-1)
        input_tensor = torch.tensor(input_tensor).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output = model(input_tensor)
            
            # Post-process based on model type (heuristic or env var)
            # Distance model returns raw float
            # Inside model returns logits -> sigmoid -> round
            
            if isinstance(model, ResNetDistanceRegressor):
                val = output.item() / 100.0 # Scale back
            else:
                val = torch.sigmoid(output).item()
                
            results.append(val)
            
    return results

def output_fn(prediction, response_content_type):
    """Format output."""
    if response_content_type == 'application/json':
        return json.dumps(prediction)
    raise ValueError(f"Unsupported content type: {response_content_type}")
