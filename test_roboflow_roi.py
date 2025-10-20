import os
import json
from PIL import Image

from inference_sdk import InferenceHTTPClient

# === CONFIGURATION ===
API_URL = "https://serverless.roboflow.com"
API_KEY = "8a39908gCoGGj7iyCgcL"

WORKSPACE_NAME = "impacteddetectionmodel"
DETECTION_WORKFLOW = "custom-workflow-3"
CLASSIFICATION_WORKFLOW = "custom-workflow-6"

INPUT_FOLDER = r"C:\yolo_test"
OUTPUT_FOLDER = r"C:\yolo_cropped"

# Create output folder if not exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === CONNECT TO CLIENT ===
client = InferenceHTTPClient(api_url=API_URL, api_key=API_KEY)

# === LOOP THROUGH ALL IMAGES ===
for filename in os.listdir(INPUT_FOLDER):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(INPUT_FOLDER, filename)
    print(f"\n🔍 Processing: {filename}")

    try:
        # === STAGE 1: DETECTION ===
        detect_result = client.run_workflow(
            workspace_name=WORKSPACE_NAME,
            workflow_id=DETECTION_WORKFLOW,
            images={"image": image_path},
            use_cache=True
        )

        print("🧾 Raw detection output:")
        print(json.dumps(detect_result, indent=2))

        # === PARSE DETECTION ===
        predictions = detect_result[0]["predictions"].get("predictions", [])
        if not predictions:
            print(f"⚠️ No ROI detected for {filename}")
            continue

        # Use the first detected ROI
        box = predictions[0]
        x, y, w, h = box["x"], box["y"], box["width"], box["height"]

        # === STAGE 2: CROP IMAGE ===
        img = Image.open(image_path)
        left = int(x - w / 2)
        top = int(y - h / 2)
        right = int(x + w / 2)
        bottom = int(y + h / 2)

        cropped_img = img.crop((left, top, right, bottom))
        roi_path = os.path.join(
            OUTPUT_FOLDER, f"{os.path.splitext(filename)[0]}_ROI.jpg"
        )
        cropped_img.save(roi_path)
        print(f"🖼️ ROI saved: {roi_path}")

        # === STAGE 3: CLASSIFICATION ===
        classify_result = client.run_workflow(
            workspace_name=WORKSPACE_NAME,
            workflow_id=CLASSIFICATION_WORKFLOW,
            images={"image": roi_path},
            use_cache=True
        )

        print("🔎 Classification result:")
        print(json.dumps(classify_result, indent=2))

        # === PARSE CLASSIFICATION ===
        class_pred = classify_result[0]["predictions"]
        top_class = class_pred.get("top", "N/A")
        conf = class_pred.get("confidence", 0)

        print(f"✅ Final Prediction: {top_class} ({conf:.2f})")

    except Exception as e:
        print(f"❌ Error processing {filename}: {e}")

print("\n🎉 All images processed!")
print(f"📁 Check results in: {OUTPUT_FOLDER}")
