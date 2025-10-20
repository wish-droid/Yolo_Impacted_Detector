import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw
import io, tempfile, os, json

# ---------------------------------
# APP CONFIGURATION
# ---------------------------------
st.set_page_config(page_title="Panoramic Tooth Detection & Classification", layout="wide")
st.title("🦷 Panoramic ROI Detection & Classification")
st.write("Please upload a panoramic image.")

# ---------------------------------
# ROBOFLOW CLIENT SETUP
# ---------------------------------
API_KEY = "8a39908gCoGGj7iyCgcL"
API_URL = "https://serverless.roboflow.com"

WORKSPACE = "impacteddetectionmodel"
DETECTION_WORKFLOW = "custom-workflow-3"
CLASSIFICATION_WORKFLOW = "custom-workflow-6"

client = InferenceHTTPClient(api_url=API_URL, api_key=API_KEY)

# ---------------------------------
# FILE UPLOAD
# ---------------------------------
uploaded_file = st.file_uploader("📁 Upload a panoramic image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.subheader(f"🖼️ Uploaded Image: {uploaded_file.name}")
    image = Image.open(uploaded_file).convert("RGB")

    # Resize for consistency
    resized_image = image.resize((640, 640))
    st.image(resized_image, caption="Resized Image (640x640)", width="stretch")

    # Save temporarily for inference
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        resized_image.save(tmp.name, format="JPEG")
        image_path = tmp.name

    # ---------------------------------
    # DETECTION
    # ---------------------------------
    st.write("🔍 Running detection on the image...")
    try:
        detection_result = client.run_workflow(
            workspace_name=WORKSPACE,
            workflow_id=DETECTION_WORKFLOW,
            images={"image": image_path},
            use_cache=True
        )
    except Exception as e:
        st.error(f"❌ Detection workflow error: {e}")
        st.stop()

    # Parse predictions
    predictions = []
    if isinstance(detection_result, list) and len(detection_result) > 0:
        predictions = detection_result[0].get("predictions", {}).get("predictions", [])
    elif isinstance(detection_result, dict):
        predictions = detection_result.get("predictions", {}).get("predictions", [])

    if not predictions:
        st.warning("⚠️ No regions detected. Check if your workflow has a valid Detection Model linked.")
    else:
        st.success(f"✅ {len(predictions)} region(s) detected!")

        annotated = resized_image.copy()
        draw = ImageDraw.Draw(annotated)
        cropped_images = []

        for i, pred in enumerate(predictions):
            x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
            label = pred.get("class", "unknown")
            conf = pred.get("confidence", 0)

            x1, y1 = max(x - w / 2, 0), max(y - h / 2, 0)
            x2, y2 = min(x + w / 2, annotated.width), min(y + h / 2, annotated.height)

            draw.rectangle([(x1, y1), (x2, y2)], outline="lime", width=3)
            draw.text((x1, max(y1 - 20, 0)), f"{label} ({conf:.2f})", fill="lime")

            roi = resized_image.crop((x1, y1, x2, y2))
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as roi_tmp:
                roi.save(roi_tmp.name, format="JPEG")
                cropped_images.append((roi, roi_tmp.name))

        st.image(annotated, caption="Detection Results", width="stretch")

        # ---------------------------------
        # CLASSIFICATION (fixed JSON parsing)
        # ---------------------------------
        st.write("Please wait. It is currently running classification on detected regions...")

        for i, (roi, roi_path) in enumerate(cropped_images):
            st.markdown(f"### 🩻 ROI {i+1}")
            st.image(roi, caption="Cropped Region", width="stretch")

            try:
                class_result = client.run_workflow(
                    workspace_name=WORKSPACE,
                    workflow_id=CLASSIFICATION_WORKFLOW,
                    images={"image": roi_path},
                    use_cache=True
                )

                # ✅ FIXED: proper parsing for classification JSON
                if isinstance(class_result, list) and len(class_result) > 0:
                    predictions_obj = class_result[0].get("predictions", {})
                    class_preds = predictions_obj.get("predictions", [])
                    top_class = predictions_obj.get("top", "N/A")
                    top_conf = predictions_obj.get("confidence", 0)
                else:
                    class_preds = []
                    top_class = "N/A"
                    top_conf = 0

                if class_preds:
                    st.write(f"**🧩 Classified as:** {top_class} ({top_conf:.2f})")

                    # Optionally show detailed scores
                    for p in class_preds:
                        st.text(f"{p['class']}: {p['confidence']:.2f}")
                        st.progress(float(p["confidence"]))
                else:
                    st.warning("⚠️ Could not classify this region.")

            except Exception as e:
                st.error(f"⚠️ Classification error: {e}")

        st.success("Detection of ROI and classification complete!")

        # Clean up temp files
        os.remove(image_path)
        for _, roi_path in cropped_images:
            if os.path.exists(roi_path):
                os.remove(roi_path)

st.markdown("---")
