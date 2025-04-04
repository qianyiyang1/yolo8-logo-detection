"""
Visualize Ground Truth Annotations
"""

import cv2
import xml.etree.ElementTree as ET
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def visualize_annotations(xml_dir: Path, img_dir: Path, output_dir: Path):
    """
    Visualize annotations on images
    
    Args:
        xml_dir (Path): Directory containing XML annotations
        img_dir (Path): Directory containing images
        output_dir (Path): Output directory for visualized images
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for xml_path in xml_dir.glob("*.xml"):
        try:
            # Parse XML
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Get image path
            img_name = root.find('filename').text
            img_path = img_dir / img_name
            
            if not img_path.exists():
                logging.warning(f"Image missing: {img_path}")
                continue
            
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                raise IOError(f"Failed to read image: {img_path}")
            
            # Draw bounding boxes
            for obj in root.findall('object'):
                bndbox = obj.find('bndbox')
                x1 = int(bndbox.find('xmin').text)
                y1 = int(bndbox.find('ymin').text)
                x2 = int(bndbox.find('xmax').text)
                y2 = int(bndbox.find('ymax').text)
                
                # Draw rectangle
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
            # Save result
            output_path = output_dir / img_name
            cv2.imwrite(str(output_path), img)
            
        except Exception as e:
            logging.error(f"Processing failed for {xml_path}: {str(e)}")

if __name__ == "__main__":
    # Configure paths
    dataset_root = Path("data/dataset")
    output_dir = dataset_root / "compare"
    
    # Visualize training set
    visualize_annotations(
        xml_dir=dataset_root/"annotations/train",
        img_dir=dataset_root/"images/train",
        output_dir=output_dir/"train"
    )
    
    # Visualize validation set
    visualize_annotations(
        xml_dir=dataset_root/"annotations/val",
        img_dir=dataset_root/"images/val",
        output_dir=output_dir/"val"
    )
    
    logging.info("Visualization completed")