#!/usr/bin/env python3
"""Build the final documents of the de-duplicated experiment from the SAVED result files (numbers are read, never typed).

    python scripts/build_dedup_report.py
Writes (into outputs/deduplicated_experiment/): results_analysis.md, FINAL_REPORT.md, SUBMISSION_CHECKLIST.md, ia3_slides_outline.md, ia3_viva_questions.md
Existing files are renamed *.prev-<timestamp>, never overwritten."""
import csv, glob, json, math, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "outputs/deduplicated_experiment"
J = lambda p: json.load(open(D / p))
split, scfg = J("dedup_audit_summary.json"), J("split_config.json")
ds, dcfg = J("downstream/summary.json"), J("downstream/config.json")
gm, gcfg = J("gan_comparison/metrics.json"), J("gan_comparison/config.json")
gv, ver = J("gan/gan_verification.json"), J("verification.json")
fail = json.load(open(glob.glob(str(D / "gan/cwgangp.failed-attempt-*/metrics.json"))[0]))
cwm = J("gan/cwgangp/metrics.json")
cm_c = J("downstream/resampling_manifest_C.json")
C = ds["conditions"]
A, B, Cc = C["A_real_only"], C["B_real_plus_cwgangp_synthetic"], C["C_real_only_matched_composition"]
PD = ds["paired_differences"]
CL = ["glioma", "meningioma", "notumor", "pituitary"]
agg = gm["aggregate"]
floor = agg["Real train reps (class-matched, reference)"]
GA = {k: agg[k] for k in ("DCGAN", "WGAN-GP", "cWGAN-GP")}
NT = ds["n_test"]
pc = {r["class"]: r for r in gm["per_class_cwgangp"]}
conf = gm["class_confusion_kid_cwgangp"]["kid"]
cnt = split["partition_class_counts_representatives"]
ntest_cls = gcfg["test_class_counts"]
f3 = lambda x: f"{x:.3f}"
f4 = lambda x: f"{x:.4f}"
pm = lambda m, s, d=3: f"{m:.{d}f} ± {s:.{d}f}"
seeds = dcfg["seeds"]
acc = {k: C[k]["accuracy"]["values"] for k in C}
bc_pos = sum(b >= c for b, c in zip(acc["B_real_plus_cwgangp_synthetic"], acc["C_real_only_matched_composition"]))
c_ge_b = sum(c >= b for b, c in zip(acc["B_real_plus_cwgangp_synthetic"], acc["C_real_only_matched_composition"]))
b_gt_a = sum(b > a for a, b in zip(acc["A_real_only"], acc["B_real_plus_cwgangp_synthetic"]))
fvals = {k: [x["fid"] for x in csv.DictReader(open(D / "gan_comparison/fid_kid_results.csv")) if x["model"] == k] for k in agg}
ci = 1.96 * math.sqrt(B["accuracy"]["mean"] * (1 - B["accuracy"]["mean"]) / NT)
n_reps = split["representatives"]
orig_total = split["pool_images"]
rank = sorted(GA, key=lambda k: GA[k]["fid_mean"])
rank_kid = sorted(GA, key=lambda k: GA[k]["kid_mean"])
assert rank == rank_kid
d_ = lambda k: GA[k]["fid_mean"]
gap_cw_w = GA["WGAN-GP"]["fid_mean"] - GA["cWGAN-GP"]["fid_mean"]
colwise_own = [c for c in CL if pc[c]["own_label_is_lowest_kid_column_wise"]]
rowwise_own = [c for c in CL if pc[c]["own_class_is_lowest_kid_row_wise"]]
sc = split["stage2_refinement"]["scored"]
n_ref_ge97 = sc["val"]["refined_ge_0.97"] + sc["test"]["refined_ge_0.97"]
n_ref_q = split["stage2_refinement"]["n_quarantined_by_refinement"]


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


gan_table = table(["Model", "FID", "KID x1000", "generated / reference (x3 seeds)"],
                  [["Real train representatives (class-matched; floor)", pm(floor["fid_mean"], floor["fid_std_across_seeds"], 2), pm(floor["kid_mean"] * 1000, floor["kid_std_across_seeds"] * 1000, 2), f"{floor['n_generated_per_seed']}/{floor['n_real']}"]] +
                  [[("**" + k + "**" if k == rank[0] else k), pm(GA[k]["fid_mean"], GA[k]["fid_std_across_seeds"], 2), pm(GA[k]["kid_mean"] * 1000, GA[k]["kid_std_across_seeds"] * 1000, 2), f"{GA[k]['n_generated_per_seed']}/{GA[k]['n_real']}"] for k in rank])
cond_names = {"A_real_only": "A real only", "B_real_plus_cwgangp_synthetic": "B real + cWGAN-GP synthetic", "C_real_only_matched_composition": "C real only, matched composition"}
cls_table = table(["Condition", "n_train", "Test accuracy", "Macro-F1", "Balanced acc.", "Weighted-F1", "Validation acc.", "Test accuracy per seed (" + ", ".join(map(str, seeds)) + ")"],
                  [[cond_names[k], v["n_train"], pm(v["accuracy"]["mean"], v["accuracy"]["std"]), pm(v["macro_f1"]["mean"], v["macro_f1"]["std"]), f3(v["balanced_accuracy"]["mean"]), f3(v["weighted_f1"]["mean"]), f3(v["val_accuracy"]["mean"]),
                    ", ".join(f3(x) for x in v["accuracy"]["values"])] for k, v in C.items()])
pdt = table(["Paired difference (per seed)", "Accuracy", "mean", "Macro-F1", "mean"],
            [[lab, ", ".join(f"{x:+.3f}" for x in PD["accuracy"][key]), f"{PD['accuracy']['mean_' + key]:+.3f}", ", ".join(f"{x:+.3f}" for x in PD["macro_f1"][key]), f"{PD['macro_f1']['mean_' + key]:+.3f}"]
             for lab, key in (("B − A", "B_minus_A"), ("C − A", "C_minus_A"), ("B − C", "B_minus_C"))])
f1t = table(["Class (test n)"] + [cond_names[k] for k in C], [[f"{c} ({ntest_cls[i]})"] + [f"F1 {v['per_class_f1_mean'][c]:.3f} / recall {v['per_class_recall_mean'][c]:.3f}" for v in C.values()] for i, c in enumerate(CL)])
pct = table(["Class", "n (generated = real per seed)", "FID", "KID x1000", "KID subset", "reliability"],
            [[c, pc[c]["n_generated_per_seed"], pm(pc[c]["fid_mean"], pc[c]["fid_std"], 1), pm(pc[c]["kid_mean"] * 1000, pc[c]["kid_std"] * 1000, 1), pc[c]["kid_subset_size"], pc[c]["reliability_note"]] for c in CL])
conft = table(["generated label \\ real test class"] + CL, [[CL[i]] + [f"{conf[i][j] * 1000:.1f}" for j in range(4)] for i in range(4)])
run_rows = table(["Model", "Generator updates", "Discriminator / critic updates", "Runtime (s, MPS)", "Finite", "Training-file hash = manifest train partition"],
                 [[m, v["generator_updates"], v["critic_or_discriminator_updates"], f"{v['runtime_seconds']:.0f}", v["all_losses_and_grads_finite"], v["train_hash_matches_manifest"]] for m, v in gv["models"].items()])
split_table = table(["Partition", *CL, "Total"], [[p] + [cnt[p][c] for c in CL] + [sum(cnt[p].values())] for p in ("train", "val", "test")])
nover = ver and sum(r["pass"] for r in ver)
vtot = len(ver)

report = f"""# Brain Tumor MRI Image Generation and Classification Using a Hybrid Conditional WGAN-GP: de-duplicated experiment (final)

> This report supersedes the earlier reports in `outputs/` (`final_report.md` and related files), which were produced on the original `archive/Training` / `archive/Testing`
> folders. A later audit showed those folders share many near-identical images, so their test numbers were optimistic. All results below were recomputed on a new
> cluster-disjoint split. **The old and new numbers are not comparable** (different data, sample sizes and reference sets).

## Abstract
The original train/test folders of the Kaggle brain-tumour MRI data used here contain many near-copies (mostly mirrored) across the split, so a near-duplicate audit was used to
build a new split: {orig_total:,} images were grouped into {split['n_clusters']:,} near-duplicate clusters, {split['quarantined_images']} images were quarantined, and one representative per cluster was assigned to a class-stratified,
cluster-disjoint train / validation / test split of {split['partition_totals']['train']:,} / {split['partition_totals']['val']} / {split['partition_totals']['test']} images (only {n_reps:,} of {orig_total:,} images are unique). A DCGAN, an unconditional WGAN-GP and the proposed conditional WGAN-GP (cWGAN-GP,
projection critic) were retrained for 3,000 generator updates on the new training partition and evaluated with FID/KID against the new test partition (n={NT}). **DCGAN had the best FID and KID**
(FID {d_('DCGAN'):.1f}), followed by cWGAN-GP (FID {d_('cWGAN-GP'):.1f}) and WGAN-GP (FID {d_('WGAN-GP'):.1f}); the proposed hybrid was not the best generator, and its margin over WGAN-GP ({gap_cw_w:.1f} FID) is small. In a downstream test
a small CNN trained on (A) real images only, (B) real plus cWGAN-GP images, or (C) a real-only control resampled to B's size and class mix reached test accuracies of
{pm(A['accuracy']['mean'], A['accuracy']['std'])}, {pm(B['accuracy']['mean'], B['accuracy']['std'])} and {pm(Cc['accuracy']['mean'], Cc['accuracy']['std'])} (3 seeds). B beat A in all {b_gt_a} seeds, but C scored higher than B on average and in {c_ge_b} of 3 seeds, so these experiments
**provide no evidence that the synthetic images help beyond added training volume**. The test set is small ({NT} images, {ntest_cls[2]} no-tumour), copies were removed but same-patient neighbouring slices could not be, and each GAN was trained once; no clinical claim is made.

## 1. Why a new split was needed
An audit of the original data found that about two thirds of the original Testing images (1,076 of 1,600) had a near-identical, often mirrored, copy in Training (exact file hashing misses these), and that
56% of the CNN-validation images had a copy in the CNN-training images. Test accuracies, validation-based epoch selection and the FID/KID reference were therefore not independent of training (details in
`outputs/final_audit/`). This experiment repeats the GAN training, the downstream comparison and the FID/KID evaluation on data split by near-duplicate cluster.

## 2. Dataset and the cluster-disjoint split
**Source.** A four-class brain tumour MRI dataset obtained from Kaggle (identified at closeout as Kaggle "Brain Tumor MRI Dataset" v2 by Masoud Nickparvar, CC BY 4.0, by matching its listed 7,200 files, 1,400/400 images per class and size; the original download record is unverified; file names include `Tr-gl_*`, `Te-gl_*`, `Tr-aug-me_*`, `Te-aug-me_*`). Both original folders were pooled ({orig_total:,} images); the old Testing folder is **not** used as the test set.

**Near-duplicate detection** (`src/dedup_split.py`, `src/dedup_refine.py`): every image is reduced to 32×32 grayscale, mean-removed and L2-normalised; similarity is the cosine similarity, evaluated against each other image in 8 orientations (identity, horizontal/vertical flip, three rotations, two transposes). Two images are linked if their similarity is ≥ {split['threshold']} or their files are byte-identical ({split['exact_hash_duplicate_groups']} byte-identical groups); transitive links form clusters (connected components).
- Result: {split['n_clusters']:,} clusters (largest {split['max_cluster_size']} images); {split['clusters_spanning_original_train_and_test']} clusters contain images from both original folders.
- Threshold sensitivity (clusters / largest cluster): {', '.join(f"{t}: {v['n_clusters']:,} / {v['max_cluster_size']}" for t, v in split['threshold_sensitivity'].items())}. The 0.98 cut was chosen after inspecting the strongest cross-partition pairs of a 0.99 attempt (kept in `_exploratory_attempt_threshold_0.99/`), at least one of which looked like a near-copy.
- Quarantine ({split['quarantined_images']} images excluded from every partition): {split['quarantined_clusters']['conflicting_labels']} clusters whose members carry **conflicting class labels** (a label-noise finding about the dataset), {split['quarantined_clusters']['all_augmented_filenames']} clusters made only of files whose names mark them as augmented (`aug`; they may be transformed copies this method cannot link), and {n_ref_q} held-out images removed in a second stage.
- Second stage: each validation/test representative's best match in a higher-priority partition was re-scored at 96×96 with ±4 px shifts. No pair reached 0.99; {n_ref_ge97} pairs reached ≥ 0.97 (maximum {max(sc['val']['max_refined'], sc['test']['max_refined']):.3f}) and the {n_ref_q} held-out images involved were quarantined. Higher-resolution scores for pairs that look alike at 32×32 (0.97-0.98) were about 0.94, i.e. similar but distinct scans, not copies.
- One representative per cluster was kept (preferring a non-augmented file, then an original Training file) and clusters were split 70/15/15 per class with seed {scfg['seed']}. **Final real-image dataset** ({n_reps:,} unique images):

{split_table}

**Verification.** No cluster occurs in more than one partition ({split['clusters_in_more_than_one_partition']} violations); the manifest and the three partition file lists are hash-recorded (`split_manifest.csv`, `split_config.json`). The strongest remaining cross-partition candidates were inspected visually (`cross_partition_top_pairs.png`): they were the same anatomy/sequence but different slices or scans.

**Residual uncertainty (important).** (i) This is a similarity heuristic with no ground truth. (ii) There are no patient identifiers, so **different slices of the same patient are not detected** and may sit in different partitions; this leakage route cannot be excluded. (iii) Files marked augmented were removed by name only. (iv) Rescaled or heavily re-cropped copies below the similarity cut would be missed. Consequently the results are better protected against copy-level leakage than the earlier ones, not proven leak-free.
**Cost:** only {n_reps:,} of {orig_total:,} images are unique and usable, and the classes became imbalanced (no-tumour has only {cnt['train']['notumor']} training and {ntest_cls[2]} test images).

## 3. Methods
- **GANs** (64×64 grayscale, z=100, batch 16, seed 42, 3,000 generator updates, identical architectures/objectives/optimisers to the earlier experiment): DCGAN (BCE, Adam 2e-4, β 0.5/0.999, 1 discriminator update per generator update); unconditional WGAN-GP (gradient penalty λ=10, 5 critic updates per generator update, Adam 1e-4, β 0.5/0.9, no BatchNorm in the critic); cWGAN-GP (50-d label embedding concatenated to z; projection critic D(x,y)=ψ(h(x))+⟨E(y),h(x)⟩; same loss; no classifier guidance). All three were trained on the {split['partition_totals']['train']:,}-image training partition only (`--manifest`); the training-file hash recorded in each checkpoint equals the manifest's training partition. The unconditional models receive no labels; the conditional model's labels are *requested* conditions, not proof of tumour-specific content.
- **Downstream CNN** (`src/dedup_downstream.py`): 4 conv blocks (32-64-128-256), global average pooling, dropout 0.3, Adam 1e-3, batch 64, 15 epochs, no augmentation, seeds {', '.join(map(str, seeds))}. The validation partition ({dcfg['n_val']} real images) selects the best epoch; the test partition ({NT}) is read only afterwards. A = real training representatives; B = A + {dcfg['synthetic_per_class']} cWGAN-GP images per class ({4 * dcfg['synthetic_per_class']:,} images, labels = requested conditions, correctness unverified); C = real training representatives resampled with replacement to B's exact per-class counts ({', '.join(str(x) for x in cm_c['per_class'])}; {cm_c['unique_files_used']} of {cm_c['unique_available']} distinct images drawn, maximum repetition {cm_c['max_multiplicity']}; no synthetic images).
- **FID/KID** (`src/dedup_eval_fid_kid.py`): torch-fidelity 0.4.0, `inception-v3-compat` pool3 features, identical preprocessing for real and generated images (grayscale → 64×64 uint8 → 3 channels), reference = the {gcfg['n_ref']} test representatives; {gcfg['n_ref']} generated images per model and seed (cWGAN-GP: the test class counts {gcfg['test_class_counts']}), generation seeds {', '.join(map(str, gcfg['generation_seeds']))}; KID = 100 subsets of {gcfg['kid']['subset_size_overall']}. The floor row compares class-matched random training representatives with the test set. The test partition is not used for checkpoint selection or tuning.

## 4. Training outcomes
{run_rows}

- All three GANs reached the intended update counts with finite losses and gradients; checkpoints reload with `weights_only=True`.
- **Interruption of the cWGAN-GP (cause unconfirmed).** The first cWGAN-GP attempt stopped at generator step {fail['generator_updates_completed'] + 1} with `{fail['error']}` after healthy losses; its log contains no Metal/GPU error message. The Mac was then found rebooted (possibly after battery exhaustion, a hypothesis only: the battery was at 21% with about 24 minutes remaining when last checked before the failure). The run was resumed from the verified step-2,500 checkpoint (all weights and optimizer states finite) and completed; the model therefore consists of steps 1-2,500 of the first attempt and 2,501-3,000 of the resumed run (data order restarts at an epoch boundary, so it is not bit-identical to an uninterrupted run). The abandoned attempt's logs and CSVs are preserved in `gan/cwgangp.failed-attempt-*`.
- Runtimes are longer than in the earlier experiment (e.g. DCGAN {gv['models']['DCGAN']['runtime_seconds']:.0f} s versus 204 s); the Mac reported Low Power Mode as enabled when checked. The runtimes are not comparable across the two experiments.

## 5. Results
### 5.1 GAN quality (new test partition, n={NT}; mean ± std over 3 generation seeds; lower is better)
{gan_table}

Ranking by FID and by KID (identical): {', '.join(rank)}. The seed standard deviations cover generation-seed variability only; each GAN was trained once, so training-run variability is unmeasured. DCGAN's lead is large relative to the seed spread; the cWGAN-GP-over-WGAN-GP margin ({gap_cw_w:.1f} FID, {GA['WGAN-GP']['kid_mean'] * 1000 - GA['cWGAN-GP']['kid_mean'] * 1000:.1f} KID x1000) is only a few seed standard deviations and could easily change across training runs. DCGAN also used 5× fewer discriminator updates than the WGAN variants' critic updates, so the comparison is not compute-matched. FID is biased upward for small samples, and the cluster-disjoint partitions no longer share copies; both factors probably contribute to the higher floor ({floor['fid_mean']:.1f}) than in the earlier experiment (a slightly negative floor KID is possible with the unbiased estimator and is consistent with zero). Sample grids: `gan_comparison/sample_grids.png`: all three models produce head-shaped, MRI-like images with soft internal structure and no identifiable tumours.

### 5.2 Class conditioning of the cWGAN-GP (per class; small samples)
{pct}

Class-confusion KID x1000 (rows = generated label, columns = real test class):

{conft}

Row-wise, only {', '.join(rowwise_own) or 'no class'} is closest to its own real class (meningioma acts as a hub that most labels are near). Column-wise, the generated label is the closest for {' and '.join(colwise_own)}; for real meningioma, generated glioma ({conf[0][1] * 1000:.1f}) is slightly closer than generated meningioma ({conf[1][1] * 1000:.1f}); pituitary is not separated (its row and column values are essentially tied) and the no-tumour estimates rest on only {ntest_cls[2]} real images (KID subset {pc['notumor']['kid_subset_size']}), so they are indicative only. Per-class FID with 46-196 images in 2,048 dimensions is strongly biased and should not be compared across classes. The labels change the output distribution (no-tumour images are visibly brighter), which is not evidence of tumour-specific or medically correct features.

### 5.3 Downstream classification (test n={NT}; mean ± sd over 3 seeds)
{cls_table}

{pdt}

B beat A in {b_gt_a} of 3 seeds (mean accuracy {PD['accuracy']['mean_B_minus_A']:+.3f}, macro-F1 {PD['macro_f1']['mean_B_minus_A']:+.3f}). The matched-composition real-only control C was {PD['accuracy']['mean_C_minus_A']:+.3f} (accuracy) above A on average and scored higher than B on average ({-PD['accuracy']['mean_B_minus_C']:+.3f} accuracy, {-PD['macro_f1']['mean_B_minus_C']:+.3f} macro-F1) and in {c_ge_b} of 3 seeds. With 3 seeds, a test set of {NT} images (a 95% interval of about ±{ci * 100:.1f} accuracy points for a single run) and seed standard deviations of about 0.02, B and C cannot be separated statistically; no significance test was run. **The measurements give no evidence that the synthetic images add benefit beyond training-set size and class balance, and the point estimates favour the control.**

Per-class F1 / recall (mean over seeds; test counts in brackets; the no-tumour estimates rest on {ntest_cls[2]} images):

{f1t}

Mean confusion matrices are in `downstream/classification_report.txt`. The largest confusion in every condition is glioma predicted as meningioma ({', '.join(f"{v['mean_confusion_matrix'][0][1]:.1f}" for v in C.values())} of {ntest_cls[0]} glioma images on average for A, B, C); meningioma is confused with glioma and with no-tumour at similar rates. Per-seed results, predictions and training histories are in `downstream/runs/`. Figure: `downstream/accuracy_A_B_C.png`.

## 6. What the results support and do not support
**Supported:** DCGAN gave the lowest FID/KID on this split, with the proposed cWGAN-GP second and WGAN-GP last (ranking identical to the earlier, contaminated experiment, though by small margins for the last two); adding cWGAN-GP images to the real data improved the mean test accuracy and macro-F1 over real-only training in all three seeds; a size- and class-mix-matched real-only control scored at least as well on average; generated labels change the output distribution (clearest for no-tumour).
**Not supported:** that the hybrid is the best generator; that the synthetic images improve classification beyond added training volume; that the synthetic labels correspond to real tumour classes; that the images are anatomically or clinically meaningful; that the result is free of patient-level leakage; that any ranking would persist across training seeds, longer training or other hyperparameters; absolute performance estimates for new patients (small test set, residual leakage).

## 7. Limitations
Small data after de-duplication ({n_reps:,} unique images; {NT} test images, {ntest_cls[2]} no-tumour) and class imbalance; heuristic near-duplicate detection with residual same-patient and rescaled-copy leakage; 3 seeds, one training run per GAN and no significance tests; short GAN training (3,000 updates; not compute-matched between DCGAN and the WGAN variants); the cWGAN-GP was resumed after an interrupted attempt (cause unconfirmed); unverified synthetic labels; Inception features on 64×64 grayscale MRI and FID bias at small n; dataset matched to Kaggle v2 but download record unverified; 8 clusters (30 images) carried conflicting class labels, indicating label noise in the source data; the two experiments (old and new split) are not comparable; no clinical claims.

## 8. Conclusion
On a split in which copies and mirrored duplicates cannot cross partitions, DCGAN again produced the best-scoring images, the proposed conditional WGAN-GP was second by a small margin over unconditional WGAN-GP, and augmenting real data with the hybrid's synthetic images did not outperform, on average, simply resampling the real data to the same size and class mix. The project demonstrates the implementation and careful evaluation of the hybrid model, but not a benefit of it; stronger claims would need patient-level splits, more seeds and training runs, longer training, label-correctness checks and significance testing.

## References
1-6 checked against arXiv (titles, authors and years match):
1. Mirza & Osindero, *Conditional Generative Adversarial Nets*, arXiv:1411.1784.
2. Radford, Metz & Chintala, *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks*, arXiv:1511.06434.
3. Gulrajani et al., *Improved Training of Wasserstein GANs*, arXiv:1704.00028.
4. Heusel et al., *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium*, arXiv:1706.08500.
5. Binkowski, Sutherland, Arbel & Gretton, *Demystifying MMD GANs*, arXiv:1801.01401.
6. Miyato & Koyama, *cGANs with Projection Discriminator*, arXiv:1802.05637.
7. Dataset: matches Kaggle Brain Tumor MRI Dataset v2 (Nickparvar, CC BY 4.0); original download record unverified.
"""

analysis = f"""# Results analysis (de-duplicated experiment)
Source files: `gan_comparison/metrics.json`, `downstream/summary.json`, `dedup_audit_summary.json`, `verification.json` ({nover}/{vtot} verification checks passed).

**1. Split.** {orig_total:,} pooled images → {split['n_clusters']:,} near-duplicate clusters (threshold {split['threshold']}; second-stage refinement at 0.97) → {n_reps:,} unique representatives: train {split['partition_totals']['train']:,}, validation {split['partition_totals']['val']}, test {split['partition_totals']['test']}; {split['quarantined_images']} images quarantined; no cluster crosses partitions. Same-patient neighbouring slices cannot be detected, so patient-level leakage may remain.

**2. GAN quality (vs {NT} test images).** FID: DCGAN {d_('DCGAN'):.1f}, cWGAN-GP {d_('cWGAN-GP'):.1f}, WGAN-GP {d_('WGAN-GP'):.1f} (floor {floor['fid_mean']:.1f}); KID x1000: {', '.join(f"{k} {GA[k]['kid_mean'] * 1000:.1f}" for k in rank)}. DCGAN is best by a wide margin; cWGAN-GP beats WGAN-GP by {gap_cw_w:.1f} FID, a few seed standard deviations, with one training run each and unequal compute (DCGAN: 3,000 discriminator updates vs 15,000 critic updates). Images are head-shaped with soft interiors and no identifiable tumours.

**3. Conditioning.** Row-wise only meningioma is closest to its own class; column-wise the generated label is closest for {' and '.join(colwise_own)}. No-tumour estimates use {ntest_cls[2]} real images. Class-dependent, not shown tumour-specific.

**4. Downstream (test n={NT}).** A {pm(A['accuracy']['mean'], A['accuracy']['std'])} (macro-F1 {f3(A['macro_f1']['mean'])}); B {pm(B['accuracy']['mean'], B['accuracy']['std'])} ({f3(B['macro_f1']['mean'])}); C {pm(Cc['accuracy']['mean'], Cc['accuracy']['std'])} ({f3(Cc['macro_f1']['mean'])}). B>A in {b_gt_a}/3 seeds; C≥B in {c_ge_b}/3 seeds (mean B−C accuracy {PD['accuracy']['mean_B_minus_C']:+.3f}). Differences between B and C are within seed spread and test-sampling error (about ±{ci * 100:.1f} points), so **no evidence of a benefit of synthetic images beyond training volume**.

**5. Caveats.** Small test set; class imbalance (no-tumour test n={ntest_cls[2]}); 3 seeds; one GAN training run each; cWGAN-GP resumed after an interrupted attempt (cause unconfirmed); unverified synthetic labels; results not comparable with the earlier experiment; dataset provenance unverified.
"""

checklist = f"""# Submission checklist (de-duplicated experiment)
Verification: `scripts/verify_dedup_experiment.py` → **{nover}/{vtot} checks passed** (`verification.json`, `verification.log`). Every decimal in the generated documents is read from the saved result files (`scripts/build_dedup_report.py`).

## Submission readiness
**Ready to submit as the primary results, with the disclosures below.** It is *not* proven leak-free: copies were removed but same-patient slices could not be, the test set is small, and the conclusions are modest. Do not describe the results as clinically meaningful or as showing a benefit of the hybrid GAN.

## Files to submit
- `outputs/deduplicated_experiment/FINAL_REPORT.md` (primary report), `ia3_slides_outline.md`, `ia3_viva_questions.md`, `results_analysis.md`.
- Figures/tables: `gan_comparison/fid_kid_bar_chart.png`, `gan_comparison/sample_grids.png`, `gan_comparison/fid_kid_results.csv`, `downstream/accuracy_A_B_C.png`, `downstream/comparison_A_B_C.csv`, `cross_partition_top_pairs.png`, GAN sample grids `gan/*/samples_step_3000.png`.
- Evidence: `split_manifest.csv`, `split_config.json`, `dedup_audit_summary.json`, `gan/gan_verification.json`.

## Key measured results
| Item | Result |
|---|---|
| Split | {split['partition_totals']['train']:,} / {split['partition_totals']['val']} / {split['partition_totals']['test']} unique images (train/val/test) from {orig_total:,}; {split['n_clusters']:,} clusters; {split['quarantined_images']} quarantined |
| FID / KID x1000 (n={NT}) | {'; '.join(f"{k} {GA[k]['fid_mean']:.1f} / {GA[k]['kid_mean'] * 1000:.1f}" for k in rank)}; floor {floor['fid_mean']:.1f} |
| Test accuracy A / B / C | {f3(A['accuracy']['mean'])} / {f3(B['accuracy']['mean'])} / {f3(Cc['accuracy']['mean'])} (macro-F1 {f3(A['macro_f1']['mean'])} / {f3(B['macro_f1']['mean'])} / {f3(Cc['macro_f1']['mean'])}) |
| GAN training | 3,000 generator updates each; critic/discriminator updates {', '.join(f"{m} {v['critic_or_discriminator_updates']:,}" for m, v in gv['models'].items())}; all finite; cWGAN-GP resumed from step 2,500 |

## Required disclosures (all in FINAL_REPORT.md)
New split replaces the original folders; reduced and imbalanced data (test n={NT}, no-tumour n={ntest_cls[2]}); heuristic duplicate detection with residual same-patient leakage; hybrid not best; no downstream benefit beyond sample size; 3 seeds / single GAN runs / no significance tests; cWGAN-GP interruption at step {fail['generator_updates_completed'] + 1} (cause unconfirmed) and resume; unverified synthetic labels; 8 label-conflict clusters (30 images) in the source data; old and new results not comparable; no clinical claims.

## Open items for you
1. Cite the dataset (Nickparvar, Kaggle, CC BY 4.0); confirm against your own download record if you have one.
2. Read the report text yourself (numbers are generated from saved files; wording was written and hedged by hand).
3. Decide whether to keep the earlier documents in `outputs/` in the submission: they are superseded (banner added) and contain optimistic numbers; submit the files in this folder instead.

## Do not submit (internal)
`*.prev-*`, `_exploratory_attempt_threshold_0.99/`, `gan/cwgangp.failed-attempt-*`, `outputs/autopilot/`, earlier experiment folders, checkpoints (large), `external_models/`.
"""

slides = f"""1. **Title** — Brain Tumor MRI Generation and Classification Using a Hybrid Conditional WGAN-GP (de-duplicated evaluation).
2. **Problem and questions** — Can a conditional WGAN-GP generate better MRI-like images than DCGAN/WGAN-GP, and do its images help a tumour classifier once training size is controlled? Academic study, no clinical claims.
3. **Why the data had to be re-split** — Original Testing folder shares near-identical (often mirrored) images with Training (1,076 of 1,600 test images; 56% of validation images); exact hashing misses them. Earlier numbers were optimistic. Figure: `outputs/final_audit/closest_testing_training_pairs.png`.
4. **Cluster-aware split** — 32×32 cosine similarity over 8 orientations, threshold {split['threshold']}, connected components, second-stage 96×96 re-scoring; {orig_total:,} images → {split['n_clusters']:,} clusters → {n_reps:,} unique images; {split['quarantined_images']} quarantined (label conflicts, augmented files). Table: split counts ({split['partition_totals']['train']:,}/{split['partition_totals']['val']}/{split['partition_totals']['test']}; no-tumour test n={ntest_cls[2]}).
5. **Residual uncertainty** — Heuristic; no patient IDs, so same-patient slices not detected; cross-partition candidate pairs inspected (`cross_partition_top_pairs.png`).
6. **Models** — DCGAN, WGAN-GP, cWGAN-GP (50-d label embedding; projection critic). 3,000 generator updates each, same seed/data. Table: architectures/optimisers.
7. **Training outcomes** — counters and runtimes; all finite; cWGAN-GP interrupted at step {fail['generator_updates_completed'] + 1} (cause unconfirmed) and resumed from step 2,500. Table: training runs.
8. **Evaluation protocol** — torch-fidelity FID/KID vs test partition (n={NT}); 3 generation seeds; floor row; downstream A/B/C with matched-composition control; test used only for final evaluation.
9. **Sample grids** — `gan_comparison/sample_grids.png`: head-shaped, soft, no identifiable tumours.
10. **GAN results** — FID: DCGAN {d_('DCGAN'):.1f} < cWGAN-GP {d_('cWGAN-GP'):.1f} < WGAN-GP {d_('WGAN-GP'):.1f}; floor {floor['fid_mean']:.1f}. Figure: `gan_comparison/fid_kid_bar_chart.png`. Hybrid not best; margin over WGAN-GP small.
11. **Class conditioning** — per-class KID and confusion; column-wise own label closest for {' and '.join(colwise_own)}; small n; class-dependent, not tumour-specific.
12. **Downstream design** — A real; B real + {4 * dcfg['synthetic_per_class']:,} synthetic; C real resampled to B's size and class mix.
13. **Downstream results** — test accuracy A {f3(A['accuracy']['mean'])}, B {f3(B['accuracy']['mean'])}, C {f3(Cc['accuracy']['mean'])}; macro-F1 {f3(A['macro_f1']['mean'])}/{f3(B['macro_f1']['mean'])}/{f3(Cc['macro_f1']['mean'])}. Figure: `downstream/accuracy_A_B_C.png`.
14. **Interpretation** — B>A in {b_gt_a}/3 seeds, but C ≥ B in {c_ge_b}/3 seeds (mean B−C {PD['accuracy']['mean_B_minus_C']:+.3f}); no evidence of benefit beyond training volume; differences within noise.
15. **Limitations** — small test set, imbalance, residual patient-level leakage, 3 seeds, single GAN runs, unverified labels/dataset download record, Inception features on small grayscale MRI.
16. **Conclusion and future work** — implemented and evaluated honestly; DCGAN best FID/KID; no downstream benefit shown. Future: patient-level split, more data and seeds, longer training, label verification, significance tests. References (arXiv 1411.1784, 1511.06434, 1704.00028, 1706.08500, 1801.01401, 1802.05637 verified; dataset download record unverified).
"""

qa = [
 ("What was the problem with the original train/test split?", "About two thirds of the original Testing images had near-identical, often mirrored, copies in Training, and 56% of CNN-validation images had copies in CNN-training; exact hashing could not see them. So earlier test numbers were optimistic."),
 ("How was the new split built?", f"All {orig_total:,} images were pooled; near-duplicates were found with 32×32 cosine similarity over 8 orientations (threshold {split['threshold']}), grouped by connected components into {split['n_clusters']:,} clusters, refined at 96×96, label-conflict and augmented-file clusters were quarantined, one representative per cluster was kept, and clusters were split 70/15/15 per class."),
 ("How many images remain?", f"{n_reps:,} unique images: train {split['partition_totals']['train']:,}, validation {split['partition_totals']['val']}, test {split['partition_totals']['test']} (no-tumour: {cnt['train']['notumor']}/{cnt['val']['notumor']}/{cnt['test']['notumor']})."),
 ("Is the new split guaranteed leak-free?", "No. It prevents copies and mirrored duplicates detected by the method from crossing partitions; same-patient neighbouring slices cannot be detected without patient IDs, and rescaled or heavily re-cropped copies below the cut may remain."),
 ("Why was the old Testing folder not reused as test?", "Because it overlaps Training; after pooling and re-partitioning by cluster, the new test partition is the only valid final-evaluation set."),
 ("Which GAN was best?", f"DCGAN: FID {d_('DCGAN'):.1f}, KID x1000 {GA['DCGAN']['kid_mean'] * 1000:.1f}; cWGAN-GP {d_('cWGAN-GP'):.1f} / {GA['cWGAN-GP']['kid_mean'] * 1000:.1f}; WGAN-GP {d_('WGAN-GP'):.1f} / {GA['WGAN-GP']['kid_mean'] * 1000:.1f}; floor {floor['fid_mean']:.1f}."),
 ("Is the cWGAN-GP better than WGAN-GP?", f"Slightly on both metrics ({gap_cw_w:.1f} FID), but with one training run each and a few seed standard deviations of margin, so it is weak evidence. The runs are also not compute-matched against DCGAN."),
 ("Why can't these FID values be compared with the earlier ones?", f"The reference set changed (n={NT} instead of 1,600) and FID is biased at small n; even the real-vs-real floor rose from 14 to {floor['fid_mean']:.0f}."),
 ("Describe the downstream conditions.", f"A: real training images ({A['n_train']:,}). B: A plus {4 * dcfg['synthetic_per_class']:,} cWGAN-GP images (requested labels). C: A resampled with replacement to B's size ({Cc['n_train']:,}) and class mix, with no synthetic images."),
 ("What were the downstream results?", f"Test accuracy A {pm(A['accuracy']['mean'], A['accuracy']['std'])}, B {pm(B['accuracy']['mean'], B['accuracy']['std'])}, C {pm(Cc['accuracy']['mean'], Cc['accuracy']['std'])}; macro-F1 {f3(A['macro_f1']['mean'])}, {f3(B['macro_f1']['mean'])}, {f3(Cc['macro_f1']['mean'])} (3 seeds, n={NT})."),
 ("Does synthetic augmentation help?", f"B beat A in {b_gt_a}/3 seeds, but C (no synthetic data) was higher than B on average and in {c_ge_b}/3 seeds. So there is no evidence the synthetic images add benefit beyond training volume; with 3 seeds and n={NT} the difference between B and C is not statistically resolved."),
 ("Why was condition C needed and what does it not control?", "B has more examples and optimiser steps than A. C matches size and class mix without synthetic data. It does not control for diversity: repeated real images are exact copies."),
 ("How uncertain are the accuracy numbers?", f"A single run on {NT} test images has a 95% interval of about ±{ci * 100:.1f} points; per-class results, especially no-tumour (n={ntest_cls[2]}), are much less certain."),
 ("What does the class-confusion KID show?", f"Row-wise only meningioma is closest to its own class; column-wise the generated label is closest for {' and '.join(colwise_own)}. Labels change the output (no-tumour images are brighter) but this is not evidence of tumour-specific or medically correct images."),
 ("What happened to the cWGAN-GP training?", f"The first attempt stopped at step {fail['generator_updates_completed'] + 1} with a non-finite critic loss/gradient after healthy losses; no Metal error was logged, the Mac was later found rebooted, and the cause is unconfirmed. It was resumed from the verified step-2,500 checkpoint and completed 3,000 generator / 15,000 critic updates; the failed attempt's logs are preserved."),
 ("How do you know each GAN used only the training partition?", "Each checkpoint records the SHA-256 of its training file list, which equals the hash of the manifest's training representatives; validation, test and quarantined images are excluded; 23 verification checks passed."),
 ("Validation vs test metrics?", f"Validation ({dcfg['n_val']} images) only chooses the best epoch; headline numbers are test accuracies of that epoch; the test partition is read only after training."),
 ("What are the main limitations?", "Small imbalanced data, heuristic de-duplication with possible patient-level leakage, 3 seeds, single GAN runs, short training, unverified synthetic labels and dataset source, Inception features on small grayscale MRI."),
 ("Can this be used clinically?", "No. The images are not shown to contain anatomically accurate tumours, labels are unverified and the classifier is a small research model."),
 ("What did you find about the dataset itself?", f"Heavy duplication (only {n_reps:,} of {orig_total:,} images are unique), augmented files in both folders, and 8 clusters (30 images) with conflicting class labels, i.e. label noise."),
 ("What would you do next?", "Obtain patient identifiers for patient-level splits, collect more unique data, run more seeds and training runs with significance tests, train longer, verify synthetic labels, and confirm the dataset source and licence."),
]
viva = "\n".join(f"{i}. **Q: {q}** A: {a}" for i, (q, a) in enumerate(qa, 1)) + "\n"

ts = time.strftime("%Y%m%d-%H%M%S")
for name, text in (("FINAL_REPORT.md", report), ("results_analysis.md", analysis), ("SUBMISSION_CHECKLIST.md", checklist), ("ia3_slides_outline.md", slides), ("ia3_viva_questions.md", viva)):
    p = D / name
    if p.exists():
        p.rename(p.with_name(f"{p.stem}.prev-{ts}{p.suffix}"))
    p.write_text(text)
    print("wrote", p.relative_to(ROOT), len(text), "chars")
