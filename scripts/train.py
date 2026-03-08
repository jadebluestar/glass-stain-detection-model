"""
Model Training Script

This script fine-tunes a YOLOv8 model for glass stain detection.

Usage:
    python train.py --epochs 100 --batch 16 --device cuda
"""

import argparse
import logging
from pathlib import Path

from ultralytics import YOLO

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def train_model(
    data_yaml: str = "data/data.yaml",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "cpu",
    patience: int = 20,
    model_name: str = "yolov8n.pt"
):
    """
    Train a YOLOv8 model for glass stain detection.
    
    Args:
        data_yaml: Path to dataset YAML configuration
        epochs: Number of training epochs
        imgsz: Input image size
        batch: Batch size
        device: Device to train on (cpu/cuda)
        patience: Early stopping patience
        model_name: Base model to fine-tune
    """
    
    logger.info(" Starting YOLOv8 Model Training")
    logger.info(f"   Model: {model_name}")
    logger.info(f"   Dataset: {data_yaml}")
    logger.info(f"   Epochs: {epochs}")
    logger.info(f"   Batch Size: {batch}")
    logger.info(f"   Device: {device}")
    
    # Load model
    try:
        model = YOLO(model_name)
        logger.info(f"✓ Model loaded: {model_name}")
    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")
        raise
    
    # Train model
    try:
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            patience=patience,
            device=device,
            project="runs/detect",
            name="glass_stain_detector",
            save=True,
            save_period=10,
            verbose=True,
            device_ids=0 if device == "cuda" else None,
        )
        
        logger.info("✓ Training completed successfully")
        logger.info(f"   Results saved to: {results.save_dir}")
        
        return results
        
    except Exception as e:
        logger.error(f"✗ Training failed: {e}")
        raise


def validate_model(model_path: str):
    """
    Validate a trained model.
    
    Args:
        model_path: Path to trained model weights
    """
    logger.info(f" Validating model: {model_path}")
    
    try:
        model = YOLO(model_path)
        metrics = model.val()
        
        logger.info("✓ Validation completed")
        logger.info(f"   mAP50: {metrics.box.map50:.3f}")
        logger.info(f"   mAP50-95: {metrics.box.map:.3f}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"✗ Validation failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 for glass stain detection")
    parser.add_argument("--data", default="data/data.yaml", help="Dataset YAML path")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--model", default="yolov8n.pt", help="Base model")
    parser.add_argument("--validate-only", action="store_true", help="Only validate, don't train")
    parser.add_argument("--weights", help="Path to model weights for validation")
    
    args = parser.parse_args()
    
    if args.validate_only:
        if not args.weights:
            logger.error("--weights required for validation")
            return
        validate_model(args.weights)
    else:
        train_model(
            data_yaml=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            patience=args.patience,
            model_name=args.model
        )


if __name__ == "__main__":
    main()
