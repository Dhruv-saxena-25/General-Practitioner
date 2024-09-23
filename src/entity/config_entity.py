from datetime import datetime
from src.constant import ARTIFACT_DIR

import os



class DirPath:
    def __init__(self, timestamp=datetime.now()):
        """
        Creates a timestamped directory inside the ARTIFACT_DIR.
        """
        self.timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.artifact_dir = os.path.join(ARTIFACT_DIR, self.timestamp)

        # Create base artifact directory
        os.makedirs(self.artifact_dir, exist_ok=True)

        
    def get_images_dir(self):
        # Create subdirectories for images
        self.images_dir = os.path.join(self.artifact_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)
        return self.images_dir

    def get_chroma_dir(self):
        # Create subdirectories for chroma
        self.chroma_dir = os.path.join(self.artifact_dir, "chroma")
        os.makedirs(self.chroma_dir, exist_ok=True)
        return self.chroma_dir


