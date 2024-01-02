import cv2

def load_video(video_path):
    """
    Loads a video from a file.
    """
    return cv2.VideoCapture(video_path)

def read_frame(video_capture):
    """
    Reads a single frame from the video capture object.
    """
    ret, frame = video_capture.read()
    return ret, frame

def save_video(output_path, frame_size, fps=30):
    """
    Creates a VideoWriter object to save output video.
    """
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    return cv2.VideoWriter(output_path, fourcc, fps, frame_size)

def display_frame(window_name, frame):
    """
    Displays a single frame in a named window.
    """
    cv2.imshow(window_name, frame)

def release_video(video_capture):
    """
    Releases the video capture object.
    """
    video_capture.release()

def write_frame(video_writer, frame):
    """
    Writes a frame to the output video.
    """
    video_writer.write(frame)

def close_window(window_name):
    """
    Closes the specified window.
    """
    cv2.destroyWindow(window_name)

# Example usage
# Loading and displaying video
video_capture = load_video('path/to/your/video.mp4')
while True:
    ret, frame = read_frame(video_capture)
    if not ret:
        break
    display_frame('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press Q to quit
        break
release_video(video_capture)
close_window('Video')

# Saving video
frame_size = (640, 480)  # Example frame size
video_writer = save_video('path/to/save/output.avi', frame_size)
# Assume you have a loop where you process and write frames
# write_frame(video_writer, processed_frame)
video_writer.release()
