import cv2
import numpy as np
import os

class StreamingVideoRender:
    """
    A class to stream and save a sequence of (H, W, C) colored images
    as a video file using OpenCV's VideoWriter.
    """
    def __init__(self, save_path: str, fps: int = 30):
        """
        Initializes the video renderer.

        Args:
            save_path (str): The full path to save the output video file (e.g., 'output.mp4').
            fps (int): Frames per second for the output video. Defaults to 30.
        """
        self.save_path = save_path
        self.fps = fps
        self.video_writer = None
        self.is_running = False

    def start(self, frame_shape: tuple):
        """
        Starts the video rendering process by initializing the VideoWriter.

        Args:
            frame_shape (tuple): The shape of the input frames (H, W, C).
                                 E.g., (480, 640, 3) for a 480p color image.
        
        Raises:
            ValueError: If the process is already running.
        """
        if self.is_running:
            raise ValueError("Renderer is already running. Call stop() before starting again.")

        if len(frame_shape) != 3 or frame_shape[2] != 3:
            # OpenCV expects (H, W) or (H, W, 3) for BGR color images
            raise ValueError(f"Input frame shape must be (H, W, 3) for a color image, but got {frame_shape}")

        # Extract dimensions (OpenCV uses W, H order for size)
        height, width, channels = frame_shape
        size = (width, height)

        # Define the codec and create a VideoWriter object
        # NOTE: Codec (FOURCC) and file extension must match.
        # - 'mp4v' or 'XVID' are often used for .mp4 or .avi respectively.
        # - You might need to experiment with codecs depending on your OS/OpenCV build.
        # 'mp4v' is generally a good choice for cross-platform .mp4
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        
        # Check if the output directory exists
        output_dir = os.path.dirname(self.save_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Initialize VideoWriter
        self.video_writer = cv2.VideoWriter(
            self.save_path, 
            fourcc, 
            self.fps, 
            size, 
            isColor=True # Explicitly state that the video is color (3 channels)
        )

        if not self.video_writer.isOpened():
            print(f"ERROR: Could not open VideoWriter at {self.save_path}")
            print("Try changing the FOURCC codec (e.g., from 'mp4v' to 'XVID' or 'MJPG').")
            self.video_writer = None # Ensure it's None if it failed
            return

        self.is_running = True
        print(f"Video rendering started: {self.save_path} ({width}x{height} @ {self.fps} FPS)")

    def step(self, frame_array: np.ndarray):
        """
        Writes a single frame to the video file.

        Args:
            frame_array (np.ndarray): The (H, W, C) colored image frame (RGB or BGR).
        
        Raises:
            RuntimeError: If the renderer has not been started.
        """
        if not self.is_running or self.video_writer is None:
            raise RuntimeError("Renderer is not running. Call start() first.")

        # OpenCV expects BGR format. If your input is RGB (common for deep learning/webcams), 
        # you need to convert it. We assume the input is RGB for this example.
        if frame_array.shape[2] == 3:
            # Convert RGB (or whatever format your input is) to BGR
            frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR) 
        else:
            frame_bgr = frame_array # Assume it's already BGR or a non-standard color frame

        self.video_writer.write(frame_bgr)

    def stop(self):
        """
        Stops the video rendering process and releases the VideoWriter resource.
        """
        if self.is_running and self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            self.is_running = False
            print(f"Video rendering stopped and saved to {self.save_path}")
        elif self.is_running:
             # Should only happen if video_writer failed to open in start()
             self.is_running = False
             print("Renderer was running but VideoWriter was not initialized. Stopped cleanly.")
        else:
            print("Renderer is already stopped.")

    def __del__(self):
        """Ensure resources are released when the object is garbage collected."""
        self.stop()