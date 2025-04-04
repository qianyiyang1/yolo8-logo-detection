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
        
        # Perform prediction
        results = model.predict(
            source="data/samples/test.jpg",
            conf=0.25,
            iou=0.5,
            device="mps",
            save=True,
            show_labels=True,
            show_conf=True
        )
        
        logging.info("Prediction completed successfully")
        
    except Exception as e:
        logging.error(f"Prediction failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()