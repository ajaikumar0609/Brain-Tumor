"""Reads the saved experiment result files (outputs/deduplicated_experiment/). Values are passed through; the only arithmetic is
the x1000 scaling of KID that the report also uses. Nothing here recomputes or modifies a result."""
import json
from . import config

EXP = config.EXP
GAN_NAMES = {"Real train reps (class-matched, reference)": "Real-vs-real floor", "DCGAN": "DCGAN", "WGAN-GP": "WGAN-GP", "cWGAN-GP": "cWGAN-GP"}
COND_NAMES = {"A_real_only": ("A", "Real only"), "B_real_plus_cwgangp_synthetic": ("B", "Real + cWGAN-GP synthetic"),
              "C_real_only_matched_composition": ("C", "Real only, matched size and class mix")}


def _load(rel: str):
    return json.loads((EXP / rel).read_text())


def metrics() -> dict:
    g = _load("gan_comparison/metrics.json")
    d = _load("downstream/summary.json")
    gan = []
    for key, label in GAN_NAMES.items():
        a = g["aggregate"][key]
        gan.append({"model": label, "is_reference": key.startswith("Real"), "fid": a["fid_mean"], "fid_std": a["fid_std_across_seeds"],
                    "kid_x1000": a["kid_mean"] * 1000, "kid_std_x1000": a["kid_std_across_seeds"] * 1000,
                    "n_generation_seeds": a["n_seeds"], "n_generated_per_seed": a["n_generated_per_seed"], "n_real": a["n_real"]})
    cond = []
    for key, (letter, label) in COND_NAMES.items():
        c = d["conditions"][key]
        cond.append({"id": letter, "name": label, "n_train": c["n_train"], "n_runs": c["n_runs"],
                     "accuracy": c["accuracy"]["mean"], "accuracy_std": c["accuracy"]["std"], "accuracy_values": c["accuracy"]["values"],
                     "macro_f1": c["macro_f1"]["mean"], "macro_f1_std": c["macro_f1"]["std"], "balanced_accuracy": c["balanced_accuracy"]["mean"],
                     "val_accuracy": c["val_accuracy"]["mean"], "per_class_f1": c["per_class_f1_mean"], "per_class_recall": c["per_class_recall_mean"],
                     "mean_confusion_matrix": c["mean_confusion_matrix"]})
    return {"n_test": d["n_test"], "classes": config.CLASSES, "gan": gan, "classifiers": cond, "paired_differences": d.get("paired_differences"),
            "per_class_cwgangp": g.get("per_class_cwgangp"),
            "sources": ["outputs/deduplicated_experiment/gan_comparison/metrics.json", "outputs/deduplicated_experiment/downstream/summary.json"],
            "notes": "FID/KID: 3 generation seeds against the 492-image test partition (mean and std across seeds); classifier values: mean and std over 3 training seeds on the same 492 test images; one GAN training run per model."}


def experiment() -> dict:
    split = _load("dedup_audit_summary.json")
    cfg = _load("split_config.json")
    ver = _load("gan/gan_verification.json")
    cnt = {}
    import csv
    for r in csv.DictReader(open(EXP / "split_manifest.csv")):
        if r["is_representative"] == "True":
            cnt.setdefault(r["partition"], {}).setdefault(r["source_path"].split("/")[1], 0)
            cnt[r["partition"]][r["source_path"].split("/")[1]] += 1
    gan = {}
    for k, v in ver["models"].items():
        gan[k] = {"generator_updates": v.get("generator_updates"), "runtime_seconds": v.get("runtime_seconds")}
    return {
        "title": "Brain Tumor MRI Image Generation and Classification Using a Hybrid Conditional WGAN-GP",
        "status": "Academic research prototype. Not a clinically validated medical diagnostic system.",
        "dataset": {
            "description": "4-class brain-tumour MRI images, 64x64 grayscale for the GAN experiments.",
            "provenance": "Matches Kaggle 'Brain Tumor MRI Dataset' v2 (Masoud Nickparvar, CC BY 4.0) by file count, per-class split and size; the original download event is unverified. The dataset is not redistributed with this application.",
            "split": "Cluster-disjoint, class-stratified 70/15/15 split after near-duplicate clustering; one representative per cluster.",
            "partition_counts": cnt, "split_config_threshold": cfg.get("threshold"), "audit": split,
        },
        "models": {"DCGAN": "BCE GAN, unconditional", "WGAN-GP": "Wasserstein GAN with gradient penalty, unconditional",
                   "cWGAN-GP": "Proposed: conditional WGAN-GP with a projection critic (class label embedding; no classifier guidance)"},
        "training": gan,
        "key_findings": ["DCGAN had the best FID and KID (154.7 / 129.4).",
                         "The matched-size real-only control (C) had the highest mean test accuracy; real + synthetic (B) did not outperform it.",
                         "The proposed cWGAN-GP was second on FID/KID, a small margin ahead of WGAN-GP."],
        "matched_control": "Condition C trains on real images only, resampled to the same size and per-class mix as condition B (real + synthetic). It separates the effect of the synthetic images from the effect of simply having more training samples.",
        "limitations": [
            "The generator did not demonstrate reliable tumour-specific anatomy; class conditioning is weak.",
            "Synthetic labels are requested conditioning labels, not expert-verified tumour diagnoses.",
            "Small, imbalanced, cluster-disjoint test set (492 images, 46 no-tumour); patient-level independence is not proven (no patient IDs, heuristic near-duplicate detection).",
            "Synthetic augmentation did not outperform the matched-size real-only control.",
            "3 training seeds for the classifiers and one training run per GAN; no significance tests; short GAN training (3,000 generator updates), not compute-matched.",
            "FID/KID use ImageNet-Inception features on 64x64 grayscale MRI, a weak proxy for medical-image quality.",
            "The cWGAN-GP run was interrupted at step 2,861 (non-finite loss) and resumed from step 2,500; the cause is unconfirmed.",
            "An earlier experiment on the original train/test folders is preliminary and superseded because of train/test near-duplicates.",
        ],
        "source_files": ["outputs/deduplicated_experiment/FINAL_REPORT.md", "outputs/submission/FINAL_REPORT.pdf"],
    }


# --- gallery: a fixed registry (id -> file); nothing outside it can be served ------------------------------------------------
def _reg():
    f = config.SUBMISSION_FIGS
    items = [
        ("grid-real-vs-generated", "Real test images vs generated samples (all models)", "mixed", f / "sample_grids_real_vs_generated.png", "Panels are labelled in the image: one real-test-image panel and one per GAN."),
        ("cwgangp-by-class", "cWGAN-GP samples by requested class (step 3000)", "synthetic", f / "cwgangp_samples_by_class.png", "Rows = requested class label; columns share the same 8 latent vectors."),
        ("near-duplicate-pairs", "Original-folder near-duplicate pairs (real MRI)", "real", f / "original_near_duplicate_pairs.png", "Real dataset images: Testing images and their near-copies in Training."),
        ("remaining-candidates", "Residual cross-partition candidates (real MRI)", "real", f / "remaining_cross_partition_candidates.png", "Real dataset images from the near-duplicate audit."),
    ]
    for key, label in (("dcgan", "DCGAN"), ("wgangp", "WGAN-GP"), ("cwgangp", "cWGAN-GP")):
        items.append((f"samples-{key}-3000", f"{label} samples, 3,000 updates", "synthetic", config.EXP / "gan" / key / "samples_step_3000.png",
                      "Generated by the saved final checkpoint during training." + (" Rows = glioma, meningioma, no tumour, pituitary." if key == "cwgangp" else " Unconditional: no class labels.")))
    return [{"id": i, "title": t, "kind": k, "path": p, "note": n} for i, t, k, p, n in items]


def gallery() -> list[dict]:
    out = []
    for r in _reg():
        if r["path"].exists():
            out.append({"id": r["id"], "title": r["title"], "kind": r["kind"], "note": r["note"], "url": f"/api/gallery/image/{r['id']}"})
    return out


def gallery_path(item_id: str):
    for r in _reg():
        if r["id"] == item_id and r["path"].exists():
            return r["path"]
    return None
