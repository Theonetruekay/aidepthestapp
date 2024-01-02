# config.py

# General Camera settings
CAMERA_DEVICE = 0  # Default camera index for cv2.VideoCapture
CAMERA_RESOLUTION = (2560, 1440)  # 2K resolution
CAMERA_FRAME_RATE = 30  # Desired frame rate

# MiDaS model settings
MIDAS_MODEL_TYPE = 'MiDaS_small'  # Model type: 'MiDaS' or 'MiDaS_small'
# Note: MiDaS_small is optimized for speed but less accurate than the larger MiDaS model.

# Path settings
MODEL_PATH = 'models/midas/model.pt'  # Path to the MiDaS model file
SAVED_FRAMES_PATH = 'data/saved_frames/'  # Directory to save captured frames
SAVED_DEPTH_MAPS_PATH = 'data/saved_depth_maps/'  # Directory to save depth maps

# Display and Processing settings
DEFAULT_COLOR_MAP = 'COLORMAP_MAGMA'  # Default color map for depth visualization
DEPTH_MAP_SCALE = 1.0  # Scale factor for resizing depth map if necessary
DISPLAY_SCALE = 0.5  # Scale factor for downsizing the display window

# Performance settings
FPS_CALC_WINDOW = 60  # Number of frames to average for FPS calculation

# Advanced Camera settings (if needed)
CAMERA_SETTINGS = {
    'brightness': 50,  # Adjust if the image is too dark or too bright
    'contrast': 10,    # Adjust the contrast level
    'saturation': 15,  # Adjust the color saturation level
    # Add more settings as required
}


# Additional imports if needed
# from some_module import some_function

# Depth Estimation Parameters
DEPTH_THRESHOLD = 0.5
POST_PROCESSING_FILTER_SIZE = 3

# User Interface Settings
UI_WINDOW_SIZE = (800, 600)
UI_THEME_COLOR = '#123456'

# Logging and Debugging
LOG_LEVEL = 'DEBUG'
LOG_FILE = 'logs/application.log'
DEBUG_MODE = True

# Networking Settings
STREAM_IP = '192.168.1.2'
STREAM_PORT = 8080
PROTOCOL = 'TCP'

# Hardware Acceleration
USE_GPU = True
GPU_ID = 0

# External Dependencies
EXTERNAL_TOOL_PATH = 'tools/external_tool/'

# Security and Authentication
API_KEY = 'your_api_key_here'
ENCRYPTION_KEY = 'your_encryption_key_here'

# Customizable Shortcuts
SHORTCUTS = {
    'save_frame': 'Ctrl+S',
    'exit': 'Ctrl+Q'
}

# Localization and Accessibility
LANGUAGE = 'English'
TEXT_SIZE = 12

# Extension and Plugin Settings
ENABLED_PLUGINS = ['plugin1', 'plugin2']
