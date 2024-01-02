import sys
import torch
import cv2
import numpy as np
from torchvision.transforms import Compose
from midas.midas_net import MidasNet
from midas.transforms import Resize, NormalizeImage, PrepareForNet

def load_model(model_path):
    model = MidasNet(model_path, non_negative=True)
    model.eval()
    return model

def read_image(image_path):
    # Read the image using OpenCV
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def preprocess_image(img, scale_factor=384):
    # Define transformations
    transform = Compose([
        Resize(
            256,  # Resize the image to 256x256 pixels
            256,
            resize_target=None,
            keep_aspect_ratio=True,
            ensure_multiple_of=32,
            resize_method="upper_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet()
    ])

    # Apply the transformations
    img_input = transform({"image": img})["image"]

    # Convert to PyTorch tensor and add batch dimension
    img_input = torch.from_numpy(img_input).unsqueeze(0)

    return img_input

def process_image(model, image_path):
    # Load and preprocess the image
    img = read_image(image_path)
    input_tensor = preprocess_image(img)

    # Perform inference
    with torch.no_grad():
        prediction = model(input_tensor)

    # Convert prediction to a numpy array
    prediction = prediction.cpu().numpy().squeeze()

    # Normalize the output depth map
    depth_map = cv2.normalize(prediction, None, 0, 255, cv2.NORM_MINMAX)
    depth_map = np.uint8(depth_map)
    depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)

    return depth_map

if __name__ == "__main__":
    model_path = sys.argv[1]
    image_path = sys.argv[2]
    model = load_model(model_path)
    depth_map = process_image(model, image_path)

    # Saving the depth map
    output_path = image_path.replace('.jpg', '_depth.jpg')
    cv2.imwrite(output_path, depth_map)
    print(f"Depth map saved to {output_path}")
