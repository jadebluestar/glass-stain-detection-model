#!/usr/bin/env python3
"""
Glass Stain Detection Model - Live Demo
Full pipeline execution showing real-time detection results
"""

import sys
import json
from pathlib import Path

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.glass_stain_detection.core.stain_classifier import StainClassifier
from src.glass_stain_detection.core.controller import CleaningController
from src.glass_stain_detection.core.pipeline import PerceptionPipeline

print("=" * 80)
print("GLASS STAIN DETECTION MODEL - LIVE DEMO")
print("=" * 80)

# Step 1: Check test images
test_images_dir = PROJECT_ROOT / "data" / "test_images"
print(f"\nStep 1: Checking test images in {test_images_dir}")

if not test_images_dir.exists():
    print("ERROR: Test images directory not found!")
    sys.exit(1)

test_images = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png"))
print(f"Found {len(test_images)} test images:")
for img in sorted(test_images):
    size_mb = img.stat().st_size / 1024 / 1024
    print(f"  ✓ {img.name} ({size_mb:.2f} MB)")

# Step 2: Initialize classifier
print(f"\n{'=' * 80}")
print("Step 2: Initializing Stain Classifier")
print("=" * 80)

try:
    classifier = StainClassifier()
    print("✓ Classifier initialized successfully")
except Exception as e:
    print(f"ERROR: {e}")
    print("\nMissing dependencies? Install with:")
    print("  pip install -r requirements.txt")
    sys.exit(1)

# Step 3: Run detection on first image
print(f"\n{'=' * 80}")
print("Step 3: Running Detection on Single Image")
print("=" * 80)

test_image = test_images[0]
print(f"\nAnalyzing: {test_image.name}")

try:
    analysis = classifier.analyze_image(str(test_image))
    
    print("\nDETECTION RESULTS:")
    print(f"  Status: {analysis['status']}")
    print(f"  Contamination: {analysis['contamination_ratio']}%")
    print(f"  Total Stains: {len(analysis['detections'])}")
    
    if analysis['detections']:
        print("\n  Details:")
        for i, det in enumerate(analysis['detections'], 1):
            print(f"    {i}. {det['class_name'].replace('_', ' ').title()}")
            print(f"       Confidence: {det['confidence']}")
            print(f"       Area: {det['area']} pixels")
            
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)

# Step 4: Generate cleaning plan
print(f"\n{'=' * 80}")
print("Step 4: Generating Cleaning Plan")
print("=" * 80)

try:
    controller = CleaningController()
    plan = controller.generate_cleaning_plan(
        analysis['detections'],
        analysis['image_size']['width'],
        analysis['image_size']['height']
    )
    
    print(f"\nCLEANING PLAN:")
    print(f"  Overall Strategy: {plan['overall_strategy']}")
    print(f"  Severity Score: {plan['highest_severity_score']:.2f}")
    
    if plan['actions_by_priority']:
        print("\n  Actions by Priority:")
        for i, action in enumerate(plan['actions_by_priority'], 1):
            print(f"    {i}. {action['class_name'].replace('_', ' ').title()}")
            print(f"       Strategy: {action['action']['strategy']}")
            
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)

# Step 5: Process all test images
print(f"\n{'=' * 80}")
print("Step 5: Processing All Test Images")
print("=" * 80)

try:
    print("\nRunning full pipeline...")
    pipeline = PerceptionPipeline()
    results = pipeline.process_batch(str(test_images_dir))
    
    print(f"✓ Processed {len(results)} images\n")
    
    # Save results
    results_file = PROJECT_ROOT / "results" / "demo_results.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    pipeline.save_results(results, str(results_file))
    print(f"✓ Results saved to: {results_file}")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 6: Print summary
print(f"\n{'=' * 80}")
print("ANALYSIS SUMMARY")
print("=" * 80)

total_stains = sum(len(r['detections']) for r in results)
avg_contamination = sum(
    r['contamination_analysis']['contamination_ratio'] for r in results
) / len(results) if results else 0

print(f"\nTotal Images: {len(results)}")
print(f"Total Stains Detected: {total_stains}")
print(f"Average Contamination: {avg_contamination:.1f}%")

print("\nPer-Image Results:")
for result in results:
    status = result['contamination_analysis']['status']
    contamination = result['contamination_analysis']['contamination_ratio']
    print(f"  • {result['image_file']:15} | {status:20} | {contamination:5.1f}% contaminated")

# Step 7: Show cleaning recommendations
print(f"\n{'=' * 80}")
print("CLEANING RECOMMENDATIONS")
print("=" * 80)

strategies = {
    "HIGH_INTENSITY": [],
    "MEDIUM": [],
    "QUICK_PASS": []
}

for result in results:
    ratio = result['contamination_analysis']['contamination_ratio']
    fname = result['image_file']
    
    if ratio > 25:
        strategies["HIGH_INTENSITY"].append((fname, ratio))
    elif ratio > 10:
        strategies["MEDIUM"].append((fname, ratio))
    else:
        strategies["QUICK_PASS"].append((fname, ratio))

for strategy in ["HIGH_INTENSITY", "MEDIUM", "QUICK_PASS"]:
    items = strategies[strategy]
    if items:
        print(f"\n{strategy}:")
        for fname, ratio in sorted(items, key=lambda x: x[1], reverse=True):
            print(f"  • {fname} ({ratio:.1f}% contamination)")

print(f"\n{'=' * 80}")
print("✓ DEMO COMPLETED SUCCESSFULLY!")
print("=" * 80)

print(f"\nResults saved to: {PROJECT_ROOT / 'results' / 'demo_results.json'}")
print(f"View results with: cat results/demo_results.json | jq .")
print("\n")
