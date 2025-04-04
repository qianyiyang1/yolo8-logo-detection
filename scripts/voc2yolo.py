"""
Convert PASCAL VOC format annotations to YOLO format
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def parse_voc_xml(xml_path: Path):
    """
    Parse VOC XML file and extract bounding box coordinates
    
    Args:
        xml_path (Path): Path to XML annotation file
        
    Returns:
        tuple: (width, height, boxes) where boxes is list of (x_center, y_center, width, height)
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Get image dimensions
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        
        # Extract all bounding boxes (force class_id=0 for logo)
        boxes = []
        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            
            # Normalize coordinates
            x_center = (xmin + xmax) / (2 * width)
            y_center = (ymin + ymax) / (2 * height)
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height
            
            boxes.append((x_center, y_center, w, h))
        
        return width, height, boxes
    except Exception as e:
        logging.error(f"XML parsing failed: {xml_path} - {str(e)}")
        return None, None, []

def convert_voc_to_yolo(xml_dir: Path, img_dir: Path, label_dir: Path):
    """
    Batch convert VOC annotations to YOLO format
    
    Args:
        xml_dir (Path): Directory containing VOC XML files
        img_dir (Path): Directory containing corresponding images
        label_dir (Path): Output directory for YOLO labels
    """
    # Statistics
    total_files = 0
    success_count = 0
    missing_images = 0
    
    # Create output directory
    label_dir.mkdir(parents=True, exist_ok=True)
    
    # Process XML files
    for xml_path in xml_dir.glob("*.xml"):
        total_files += 1
        logging.info(f"Processing: {xml_path.name} ({total_files})")
        
        # Get corresponding image path
        img_stem = xml_path.stem
        img_path = img_dir / f"{img_stem}.jpg"  # Assuming JPG format
        
        # Check image existence
        if not img_path.exists():
            logging.warning(f"Image missing: {img_path}")
            missing_images += 1
            continue
        
        # Parse XML
        width, height, boxes = parse_voc_xml(xml_path)
        if not boxes:
            continue
        
        # Write YOLO label file
        txt_path = label_dir / f"{img_stem}.txt"
        try:
            with open(txt_path, 'w') as f:
                for box in boxes:
                    f.write(f"0 {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")
            success_count += 1
        except IOError as e:
            logging.error(f"Failed to write {txt_path}: {str(e)}")
    
    # Print summary
    logging.info("\nConversion Summary:")
    logging.info(f"Total XML files: {total_files}")
    logging.info(f"Successful conversions: {success_count}")
    logging.info(f"Missing images: {missing_images}")
    logging.info(f"Failed conversions: {total_files - success_count - missing_images}")

if __name__ == "__main__":
    # Configure paths
    dataset_root = Path("data/dataset")
    
    # Convert training set
    convert_voc_to_yolo(
        xml_dir=dataset_root/"annotations/train",
        img_dir=dataset_root/"images/train",
        label_dir=dataset_root/"labels/train"
    )
    
    # Convert validation set
    convert_voc_to_yolo(
        xml_dir=dataset_root/"annotations/val",
        img_dir=dataset_root/"images/val",
        label_dir=dataset_root/"labels/val"
    )