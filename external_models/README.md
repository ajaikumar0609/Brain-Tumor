# External pretrained models — inventory

Last verified: 2026-10-11 (macOS, Apple Silicon, no torch installed at verification time).

**Status vocabulary** — *Downloaded* = file is on disk. *Verified* = integrity/structure checked. *Inference-tested* = the model was actually run. **Neither model has been inference-tested.** No training was run.

Weights are git-ignored by the starter `.gitignore` for `*.pth`; the two `.zip` files are **not** — add `external_models/` to `.gitignore` before any `git init`/push. Do not commit weights.

---

## 1. Brain tumor classifier (EfficientNet-B3)

| Item | Value |
|---|---|
| Source | https://github.com/Phi-007/brain-tumor-classifier (branch `main`) |
| How obtained | `git clone` failed 3× (HTTP/2 stream cancel / early EOF on the unstable connection). Fell back to `https://codeload.github.com/Phi-007/brain-tumor-classifier/zip/refs/heads/main` (resumable). **No `.git` history in the folder.** |
| Source archive | `external_models/brain_tumor_classifier_source_main.zip` (40,287,506 B) |
| Code | `external_models/brain_tumor_classifier/` (`src/model.py`, `src/dataset.py`, `main.py`, `app.py`) |
| Weights | `external_models/brain_tumor_classifier/weights/best_model.pth` |
| Size | 43,373,711 bytes |
| SHA-256 | `7840ed87964a5ce06c0866b167e5624bb259a58895139c9687163c3f77d77c4a` |
| License | Apache-2.0 (`LICENSE` in repo). Repo disclaimer: educational use only, not for medical/diagnostic decisions. Attribution: GitHub user Phi-007 (repo author; demo linked at HF Space `LeonardoESS02/brain_tumor_classifier`). |
| Architecture | `BrainTumorEfficientNet`: torchvision `efficientnet_b3`, first conv replaced by 1-channel (40,1,3,3) (RGB weights averaged), head `Dropout(0.4) -> Linear(1536, 4)` |
| Input | 1-channel grayscale, **224×224**; `Grayscale(1) -> Resize((224,224)) -> ToTensor -> Normalize(mean=0.1856, std=0.1892)` |
| Output | 4 logits, class order (alphabetical): **0 glioma, 1 meningioma, 2 notumor, 3 pituitary** |
| Dependencies | torch>=2.0, torchvision>=0.15, Pillow, tqdm (gradio/kaggle/huggingface_hub only for the demo app). Note `BrainTumorEfficientNet.__init__` calls `efficientnet_b3(weights=DEFAULT)`, which **downloads ImageNet weights** on construction (unneeded when loading this checkpoint — pass `weights=None` in our own wrapper). |
| Training data (per README) | Kaggle `masoudnickparvar/brain-tumor-mri-dataset` (5,712 train / 1,311 test) — the same dataset we are downloading |

**Verification performed (no code from the repo executed, torch not imported):**
- Real PyTorch zip archive, not a Git LFS pointer; `zipfile.testzip()` clean; 580 entries.
- Parsed with a restricted unpickler: `OrderedDict` state_dict, 574 tensors, 10,789,034 parameters; `backbone.features.0.0.weight` = (40,1,3,3); `backbone.classifier.1.weight` = (4,1536). Shapes match `src/model.py`.

| Downloaded | Structurally verified | Inference-tested |
|---|---|---|
| Yes | Yes (shapes/keys vs. source) | **Yes — 2026-10-11, MPS and CPU, `archive/Testing` 1,600 images: acc 0.9594, macro-F1 0.9587 (see `outputs/pretrained_classifier/`)** |

**Compatibility notes / limitations**
- Compatible in principle with our 4-class Kaggle data (same class names, same folder layout `Training/`, `Testing/`).
- Published accuracy is from a Colab notebook run and is **not** reproduced here. The model's test split is the Kaggle Testing folder, so any use of that data for our own fine-tuning must not leak into evaluation.
- `main.py` / `app.py` use `torch.load(path)` without `weights_only`; on torch>=2.6 the default is safe for state_dicts, on older torch prefer `weights_only=True`.
- Class index order differs from the order in the README prose (glioma/meningioma/pituitary/no tumor); **use the dataset order above**.

---

## 2. Brain MRI Inpaint GAN (00007_INPAINT_BRAIN_MRI)

| Item | Value |
|---|---|
| Source | https://zenodo.org/records/7041738 (medigan model `00007_INPAINT_BRAIN_MRI`) |
| Archive | `external_models/brain_mri_inpaint_gan/00007_INPAINT_BRAIN_MRI.zip` (preserved) |
| Archive size | 207,138,548 bytes (matches Zenodo API) |
| MD5 | `b1b11fad1e6e5a5553b93501b546f8c9` — **matches** published checksum |
| SHA-256 (archive) | `35c5ff63491002aec921f5b7d245e801dd6a7d423ae513916c22cc2d336d9811` |
| Archive integrity | `unzip -t`: no errors; no absolute/`..` paths; extracted to `brain_mri_inpaint_gan/extracted/` |
| License | CC-BY-4.0 (Zenodo record). `metadata.json` has `license: null`. Cite the publication listed in `metadata.json`: *Medical Physics* (doi 10.1002/mp.14701; arXiv 2003.07526) — check the author list there; original repo https://github.com/KSH0660/BrainTumor; packaged for medigan. |

Checkpoints (in `extracted/`):

| File | Bytes | SHA-256 | State dict |
|---|---|---|---|
| `inp_gen.pth` | 39,778,342 | `8b6bdff83610a472a9b96cb7eed70d892656290fa54ad018ddc16366022d439d` | 100 tensors, 9,937,340 params; first conv (32,5,3,3) |
| `tumor_shape.pth` | 17,341,378 | `422a83c54b7460a5d4ec7834170968d5cc731737e887284e4d61150c2fd2ec32` | 40 tensors, 4,333,089 params; first conv (32,2,5,5) |
| `tumor_grade.pth` | 17,341,644 | `d9f26010ae001f5e0b8befbc469f2d696d8c1b658eca9df6a47e5e3cccc3bb45` | 40 tensors, 4,333,089 params; first conv (32,2,5,5) |

Supporting code: `__init__.py` (`generate()`), `model.py` (`Inpaint_generator`, `Tumor_shape`, `Tumor_grade`), `inpaint_model.py`, `train.py`, `dataloader.py`, `losses.py`, `util.py`, `augment.py`, `inference_example.ipynb`, `requirements.txt`, `test.sh`, MATLAB helpers (`nii2png.m`, `Pick_normslice.m`), example images in `Norm_slice/{F,F2,M,T1,T1c,T2}` and `temp_slice/`. (`.DS_Store` and `__pycache__/*.pyc` in the archive are junk and were not run.)

| Item | Value |
|---|---|
| Architecture | Concentric-circle → `Tumor_shape` (binary mask) and `Tumor_grade` (grade mask) networks → partial-conv `Inpaint_generator` that paints a tumor into a normal slice, one head per modality (Flair, T1, T1c, T2) |
| Input | Normal-brain 2D slices (PNG) per modality folder, 256×256, plus tumor placement via circle params `x_center, y_center, radius_1/2/3` (defaults 130,130,10,15,30) |
| Output | 256×256 synthetic tumor-inpainted slices per modality (+ binary and grade masks) |
| Dependencies | matplotlib, albumentations, torch, torchvision, numpy, pillow, opencv-contrib-python. `test.sh` pins `torch==1.10.0+cu113` — **CUDA-specific, do not use on the MacBook**; the code picks `cuda` if available, else CPU (MPS untested). |
| Trained on | BRATS (multi-modal glioma MRI: Flair/T1/T1c/T2), 2020 |

| Downloaded | Checksum-verified | Structurally verified | Inference-tested |
|---|---|---|---|
| Yes | Yes (MD5 matches) | Yes (tensor shapes consistent with `model.py`) | **No** |

**Compatibility limitations — important**
- This is a **tumor-inpainting GAN, not a class-conditional generator.** It takes an existing normal brain slice plus a circle-based mask and synthesizes a (glioma-style, BRATS-grade) tumor inside it. It has **no glioma/meningioma/pituitary/no-tumor class input**, so it cannot directly produce the four Kaggle classes.
- Domain mismatch: trained on BRATS multi-sequence 256×256 slices; Kaggle images are single-sequence (mixed orientations, varied sizes), so it needs resizing/modality choice and the result is a domain shift. Only the Flair/T1/T1c/T2 head structure exists; none corresponds to the Kaggle data.
- It is **not** our proposed hybrid conditional WGAN-GP. At best it can be (a) a baseline/comparison, or (b) a source of tumor-augmented images for the `glioma` class, clearly labelled as such.
- Legacy pickle checkpoint format (protocol 2 `.pth`); load with `weights_only=True` on modern torch.
