"""
app.py — Gradio demo for the Brain Tumor MRI Classifier.

Deployed on HuggingFace Spaces. The model weights (best_model.pth) must sit
in the same directory as this file. Upload them via:

    huggingface-cli upload YOUR_USERNAME/brain-tumor-classifier \
        weights/best_model_phase_2.pth best_model.pth
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import gradio as gr

from src.model import BrainTumorEfficientNet


# ======================================================================
#  CONFIG
# ======================================================================

WEIGHTS_PATH = "weights/best_model.pth"
MEAN, STD    = 0.1856, 0.1892
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Human-readable class names (same order as BrainTumorDataset.CLASSES)
CLASS_LABELS = {
    "glioma":      "🔴 Glioma",
    "meningioma":  "🟠 Meningioma",
    "notumor":     "🟢 No Tumor",
    "pituitary":   "🟡 Pituitary",
}
CLASSES = list(CLASS_LABELS.values())   # preserves glioma / meningioma / notumor / pituitary order


# ======================================================================
#  MODEL  (loaded once at startup)
# ======================================================================

model = BrainTumorEfficientNet(num_classes=4)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()


# ======================================================================
#  PREPROCESSING  (identical to test_transform in main.py)
# ======================================================================

preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[MEAN], std=[STD]),
])


# ======================================================================
#  INFERENCE
# ======================================================================

def predict(image: Image.Image) -> dict[str, float]:
    """
    Run inference on a PIL image and return a label → probability dict.
    Gradio's Label component reads this format directly.
    """
    tensor = preprocess(image).unsqueeze(0).to(DEVICE)   # (1, 1, 224, 224)

    with torch.no_grad():
        logits = model(tensor)                            # (1, 4)
        probs  = F.softmax(logits, dim=1).squeeze()       # (4,)

    return {label: float(prob) for label, prob in zip(CLASSES, probs)}


# ======================================================================
#  GRADIO UI
# ======================================================================

with gr.Blocks(title="🧠 Brain Tumor MRI Classifier") as demo:

    gr.Markdown(
        """
        # 🧠 Brain Tumor MRI Classifier
        Upload a brain MRI scan and the model will classify it into one of four categories.

        > ⚠️ **Disclaimer:** This model is for **educational purposes only** and must
        > **not** be used to make any medical or diagnostic decisions.
        > Always consult a qualified medical professional.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="Upload Brain MRI Scan",
                image_mode="RGB",   # Gradio always passes RGB; preprocess converts to grayscale
            )
            submit_btn = gr.Button("Classify", variant="primary")

        with gr.Column(scale=1):
            label_output = gr.Label(
                num_top_classes=4,
                label="Prediction",
            )

    gr.Markdown(
        """
        ### Classes
        | Label | Description |
        |---|---|
        | 🔴 Glioma | Tumor originating in the glial cells of the brain or spine |
        | 🟠 Meningioma | Tumor on the membranes surrounding the brain and spine |
        | 🟡 Pituitary | Tumor in the pituitary gland at the base of the brain |
        | 🟢 No Tumor | Healthy scan — no tumor detected |

        **Model:** EfficientNet-B3 with two-phase transfer learning · **Test accuracy:** ~98 - 99%
        """
    )

    submit_btn.click(fn=predict, inputs=image_input, outputs=label_output)
    image_input.change(fn=predict, inputs=image_input, outputs=label_output)


if __name__ == "__main__":
    demo.launch()
