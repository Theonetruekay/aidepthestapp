import cv2
import numpy as np
import os

def read_image(image_path, in_color=True):
    """
    Reads an image from a given path.
    """
    if in_color:
        return cv2.imread(image_path, cv2.IMREAD_COLOR)
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

def save_image(image, save_path):
    """
    Saves an image to a specified path.
    """
    cv2.imwrite(save_path, image)

def resize_image(image, size):
    """
    Resizes an image to a given size.
    """
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

def convert_to_grayscale(image):
    """
    Converts an image to grayscale.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def calculate_fps(start_time, frame_count):
    """
    Calculates the frames per second (FPS).
    """
    current_time = time.time()
    return frame_count / (current_time - start_time)

def list_files_in_directory(directory, file_extension=None):
    """
    Lists all files in a directory with an optional file extension filter.
    """
    files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    if file_extension:
        return [f for f in files if f.endswith(file_extension)]
    return files

def normalize_array(arr):
    """
    Normalizes a numpy array to have values between 0 and 1.
    """
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    return (arr - arr_min) / (arr_max - arr_min)

# Add more utility functions as needed for your specific application.
