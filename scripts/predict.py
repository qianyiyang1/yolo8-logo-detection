"""
Inference Script for YOLOv8 Model
"""

from ultralytics import YOLO
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def main():
    """Main prediction function"""
    try:
        # Load trained model
        model_path = Path("runs/detect/train/weights/best.pt")
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
            
        model = YOLO(model_path)
        
        # Get all jpg files from the data/samples/ directory
        input_dir = Path("data/samples/")
        jpg_files = list(input_dir.glob("*.jpg"))  # Get all .jpg files
        
        if not jpg_files:
            raise FileNotFoundError("No .jpg files found in the 'data/samples/' directory.")
        
        # Perform prediction for each image in the directory
        for image_path in jpg_files:
            logging.info(f"Predicting on {image_path}")
            
            results = model.predict(
                source=str(image_path),
                conf=0.25,
                iou=0.5,
                device="mps",
                save=True,
                show_labels=True,
                show_conf=True
            )
            
            logging.info(f"Prediction completed for {image_path}")
        
    except Exception as e:
        logging.error(f"Prediction failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
