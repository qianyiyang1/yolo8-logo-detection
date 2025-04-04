"""
YOLOv8 Training Script
"""

from ultralytics import YOLO
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def main():
    """Main training function"""
    try:
        # Initialize model
        model = YOLO("yolov8n.pt")  # Load pretrained model
        
        # Training configuration
        results = model.train(
            data="configs/dataset.yaml",
            epochs=150,
            batch=8,
            device="mps",  # Use MPS for Apple Silicon
            imgsz=640,
            workers=4,
            augment=True,
            lr0=0.01,
            weight_decay=0.0005
        )
        
        logging.info("Training completed successfully")
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()