#!/usr/bin/env python3
"""Build outputs/submission/{FINAL_REPORT.pdf, FINAL_REPORT.docx, IA3_Presentation.pptx} + figures from the VERIFIED de-duplicated experiment.

    python scripts/build_submission_docs.py
All numbers are read from saved result files (outputs/deduplicated_experiment/**). Report text = outputs/deduplicated_experiment/FINAL_REPORT.md plus
clearly marked additions (title block with placeholders, provenance check, hardware, figures). Existing outputs are renamed *.prev-<ts>, never overwritten."""
import csv, json, math, re, shutil, time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "outputs/deduplicated_experiment"
OUT = ROOT / "outputs/submission"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
TS = time.strftime("%Y%m%d-%H%M%S")
J = lambda p: json.load(open(D / p))
CL = ["glioma", "meningioma", "notumor", "pituitary"]


def versioned(p: Path):
    if p.exists():
        p.rename(p.with_name(f"{p.stem}.prev-{TS}{p.suffix}"))


# ------------------------------------------------------------------ data (read, never typed)
split, scfg, ds, dcfg = J("dedup_audit_summary.json"), J("split_config.json"), J("downstream/summary.json"), J("downstream/config.json")
gm, gcfg, gv = J("gan_comparison/metrics.json"), J("gan_comparison/config.json"), J("gan/gan_verification.json")
C = ds["conditions"]
A, B, Cc = C["A_real_only"], C["B_real_plus_cwgangp_synthetic"], C["C_real_only_matched_composition"]
agg = gm["aggregate"]
floor = agg["Real train reps (class-matched, reference)"]
GA = {k: agg[k] for k in ("DCGAN", "WGAN-GP", "cWGAN-GP")}
rank = sorted(GA, key=lambda k: GA[k]["fid_mean"])
cnt = split["partition_class_counts_representatives"]
NT = ds["n_test"]
pc = {r["class"]: r for r in gm["per_class_cwgangp"]}
PD = ds["paired_differences"]
f3 = lambda x: f"{x:.3f}"
pm = lambda m, s, d=3: f"{m:.{d}f} ± {s:.{d}f}"
acc = {k: C[k]["accuracy"]["values"] for k in C}
c_ge_b = sum(c >= b for b, c in zip(acc["B_real_plus_cwgangp_synthetic"], acc["C_real_only_matched_composition"]))
b_gt_a = sum(b > a for a, b in zip(acc["A_real_only"], acc["B_real_plus_cwgangp_synthetic"]))

# ------------------------------------------------------------------ figures
for src, name in ((D / "gan_comparison/sample_grids.png", "sample_grids_real_vs_generated.png"), (D / "gan_comparison/fid_kid_bar_chart.png", "fid_kid_bar_chart.png"),
                  (D / "downstream/accuracy_A_B_C.png", "accuracy_A_B_C.png"), (D / "gan/cwgangp/samples_step_3000.png", "cwgangp_samples_by_class.png"),
                  (ROOT / "outputs/final_audit/closest_testing_training_pairs.png", "original_near_duplicate_pairs.png"), (D / "cross_partition_top_pairs.png", "remaining_cross_partition_candidates.png")):
    shutil.copy2(src, FIG / name)
fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
for ax, (k, v) in zip(axes, C.items()):
    cm = np.array(v["mean_confusion_matrix"])
    ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(4), CL, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(4), CL, fontsize=8)
    ax.set_title({"A_real_only": "A real only", "B_real_plus_cwgangp_synthetic": "B real + synthetic", "C_real_only_matched_composition": "C real, matched"}[k] + f"\nmacro-F1 {v['macro_f1']['mean']:.3f}", fontsize=9)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i, j]:.0f}", ha="center", va="center", fontsize=8, color="white" if cm[i, j] / cm[i].sum() > 0.5 else "black")
    ax.set_xlabel("predicted", fontsize=8)
axes[0].set_ylabel("true", fontsize=8)
fig.suptitle(f"Mean confusion matrices over 3 seeds (counts; shading = row fraction); test n={NT}", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "confusion_matrices_A_B_C.png", dpi=150)
plt.close(fig)
fig, ax = plt.subplots(figsize=(6.4, 3.4))
bottom = np.zeros(3)
cols = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]
for c, col in zip(CL, cols):
    vals = np.array([cnt[p][c] for p in ("train", "val", "test")])
    ax.bar(["train", "validation", "test"], vals, bottom=bottom, color=col, label=c)
    bottom += vals
for i, t in enumerate(bottom):
    ax.text(i, t + 20, f"{int(t)}", ha="center", fontsize=9)
ax.legend(fontsize=8, frameon=False)
ax.set_title(f"Unique images per partition after de-duplication (total {int(bottom.sum()):,} of {split['pool_images']:,})", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "split_counts.png", dpi=150)
plt.close(fig)

# ------------------------------------------------------------------ report source (verified text + marked additions)
md = (D / "FINAL_REPORT.md").read_text()
FIGS = {
    "orig": ("Figure 1. Examples of original Testing images (top row) and their near-identical, often mirrored, copies in Training (bottom row), found by the near-duplicate audit.", "figures/original_near_duplicate_pairs.png"),
    "split": ("Figure 2. Unique images per partition and class after de-duplication.", "figures/split_counts.png"),
    "remain": ("Figure 3. Strongest remaining cross-partition candidate pairs after the final split: similar anatomy, different slices or scans (inspected visually).", "figures/remaining_cross_partition_candidates.png"),
    "samples": ("Figure 4. Real test images and samples from DCGAN, WGAN-GP and cWGAN-GP (rows of the conditional model: glioma, meningioma, notumor, pituitary).", "figures/sample_grids_real_vs_generated.png"),
    "fid": ("Figure 5. FID and KID (x1000) against the new test partition; error bars = standard deviation over 3 generation seeds.", "figures/fid_kid_bar_chart.png"),
    "cls": ("Figure 6. cWGAN-GP samples at step 3,000: rows = requested class label, columns = the same eight latent vectors.", "figures/cwgangp_samples_by_class.png"),
    "acc": ("Figure 7. Downstream test accuracy, macro-F1 and validation accuracy for conditions A, B and C (dots = seeds).", "figures/accuracy_A_B_C.png"),
    "cm": ("Figure 8. Mean confusion matrices for A, B and C.", "figures/confusion_matrices_A_B_C.png"),
}
fig_md = lambda k: f"![{FIGS[k][0]}]({FIGS[k][1]})\n"
prov = ("\n\n" + "**Provenance check (final closeout).** The original download record could not be recovered: the extracted `archive/` folder carries a macOS download-quarantine attribute showing a Chrome download, but no URL is recorded, and no ZIP or Kaggle metadata remains on the machine. The dataset was therefore identified by comparison with the public Kaggle page (viewed in a browser during closeout): *Brain Tumor MRI Dataset*, owner Masoud Nickparvar, `https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset`, Version 2, licence **CC BY 4.0**. The page lists 7,200 images (Training 1,400 and Testing 400 per class; classes glioma, meningioma, pituitary, no tumour), a size of 167.85 MB, and sources figshare, SARTAJ and Br35H. The local data match on every one of these points (7,200 files, identical per-class split, 167.85 MB, file prefixes Tr-/Te-; 203 files carry an `aug` marker, which the page does not mention). This is a strong match but not proof that the local files are byte-identical to that version, and the original download event is **unverified**. The page also says Version 2 removed duplicates and eliminated train/test overlap, whereas this project's own near-duplicate audit (heuristic; see Section on de-duplication) found near-identical copies across the two folders. Cite the dataset as: Nickparvar, M., Brain Tumor MRI Dataset, Kaggle (CC BY 4.0), together with its listed sources. Neither the dataset nor the pretrained weights are redistributed with this submission.")
hw = ("**Hardware and software.** Apple MacBook Air (M4 chip, 16 GB unified memory), macOS (Darwin 25.6.0), Python 3.14.3, PyTorch 2.14.0 on the Metal Performance Shaders (MPS) GPU backend "
      "with no CPU fallback, torchvision 0.29.1, torch-fidelity 0.4.0. Low Power Mode was enabled when checked during the later runs, which made them slower than the earlier experiment.\n\n")


def ins(text, anchor, new, before=True):
    assert text.count(anchor) == 1, anchor[:60]
    return text.replace(anchor, (new + anchor) if before else (anchor + new))


md = ins(md, "\n## 2. Dataset and the cluster-disjoint split", "\n" + fig_md("orig"))
md = md.replace("Both original folders were pooled (7,200 images); the old Testing folder is **not** used as the test set.",
                "Both original folders were pooled (7,200 images); the old Testing folder is **not** used as the test set." + prov)
md = ins(md, "**Verification.** No cluster occurs", fig_md("split") + "\n" + fig_md("remain") + "\n")
md = ins(md, "\n## 4. Training outcomes", hw.rstrip("\n") + "\n")
md = ins(md, "\n### 5.2 Class conditioning", "\n" + fig_md("samples") + "\n" + fig_md("fid"))
md = ins(md, "\n### 5.3 Downstream classification", "\n" + fig_md("cls"))
md = ins(md, "\n## 6. What the results support", "\n" + fig_md("acc") + "\n" + fig_md("cm"))
(OUT / "FINAL_REPORT_source.md").write_text(md)

# ------------------------------------------------------------------ markdown subset parser
INL = re.compile(r"(\*\*[^*]+\*\*|`[^`]+`|(?<![\w*])\*[^*\n]+\*(?![\w*]))")


def inline(text):
    out = []
    for part in INL.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            out.append((part[2:-2], "b"))
        elif part.startswith("`") and part.endswith("`") and len(part) > 2:
            out.append((part[1:-1], "c"))
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            out.append((part[1:-1], "i"))
        else:
            out.append((part, ""))
    return out


def parse(md_text):
    lines, blocks, i = md_text.split("\n"), [], 0
    while i < len(lines):
        l = lines[i]
        if not l.strip():
            i += 1
        elif l.startswith("# "):
            blocks.append(("h1", l[2:])); i += 1
        elif l.startswith("## "):
            blocks.append(("h2", l[3:])); i += 1
        elif l.startswith("### "):
            blocks.append(("h3", l[4:])); i += 1
        elif l.startswith("!["):
            m = re.match(r"!\[(.*)\]\((.*)\)", l)
            blocks.append(("img", (m.group(1), m.group(2)))); i += 1
        elif l.startswith(">"):
            buf = []
            while i < len(lines) and lines[i].startswith(">"):
                buf.append(lines[i].lstrip("> ").strip()); i += 1
            blocks.append(("quote", " ".join(buf)))
        elif l.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"-+", c) for c in cells):
                    rows.append(cells)
                i += 1
            blocks.append(("table", rows))
        elif re.match(r"^\d+\. ", l):
            items = []
            while i < len(lines) and re.match(r"^\d+\. ", lines[i]):
                items.append(re.sub(r"^\d+\. ", "", lines[i])); i += 1
            blocks.append(("ol", items))
        elif l.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(lines[i][2:]); i += 1
            blocks.append(("ul", items))
        else:
            buf = [l.strip()]
            i += 1
            while i < len(lines) and lines[i].strip() and not re.match(r"^(#|>|\||!\[|- |\d+\. )", lines[i]):
                buf.append(lines[i].strip()); i += 1
            blocks.append(("p", " ".join(buf)))
    return blocks


blocks = parse(md)
TITLE = blocks[0][1]
blocks = blocks[1:]
PLACE = [("Student name", "Ajai Kumar N"), ("Register number", "URK23CS5064"), ("Programme", "B.Tech in Computer Science and Engineering (specialization: Artificial Intelligence)"), ("Institution", "Karunya Institute of Technology and Sciences"),
         ("Guide / faculty", "[to be filled in]"), ("Submission date", "[to be filled in]")]

# ------------------------------------------------------------------ DOCX
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


def shade(cell, color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), color)
    tcPr.append(shd)


def add_runs(par, text, size=None, color=None):
    for t, st in inline(text):
        r = par.add_run(t)
        r.bold = "b" in st
        r.italic = "i" in st
        if "c" in st:
            r.font.name = "Menlo"
            r.font.size = Pt((size or 10.5) - 1)
        elif size:
            r.font.size = Pt(size)
        if color:
            r.font.color.rgb = color


doc = Document()
sec = doc.sections[0]
sec.left_margin = sec.right_margin = Inches(0.9)
sec.top_margin = sec.bottom_margin = Inches(0.85)
doc.styles["Normal"].font.name = "Calibri"
doc.styles["Normal"].font.size = Pt(10.5)
for h, sz in (("Heading 1", 16), ("Heading 2", 13.5), ("Heading 3", 11.5)):
    doc.styles[h].font.name = "Calibri"
    doc.styles[h].font.size = Pt(sz)
    doc.styles[h].font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
t = doc.add_paragraph()
t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run(TITLE.split(":")[0]); r.bold = True; r.font.size = Pt(20); r.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
t2 = doc.add_paragraph(); t2.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t2.add_run("IA3 project report: final, de-duplicated evaluation"); r.font.size = Pt(13)
for k, v in PLACE:
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    rr = p.add_run(f"{k}: "); rr.bold = True
    p.add_run(v).font.color.rgb = RGBColor(0x99, 0x55, 0x00)
    p.paragraph_format.space_after = Pt(2)
doc.add_page_break()
for kind, val in blocks:
    if kind in ("h1", "h2", "h3"):
        doc.add_heading(val, level={"h1": 1, "h2": 2, "h3": 3}[kind])
    elif kind == "p":
        add_runs(doc.add_paragraph(), val)
    elif kind == "quote":
        p = doc.add_paragraph(); p.paragraph_format.left_indent = Inches(0.3)
        add_runs(p, val, color=RGBColor(0x55, 0x55, 0x55))
    elif kind in ("ul", "ol"):
        for it in val:
            add_runs(doc.add_paragraph(style="List Bullet" if kind == "ul" else "List Number"), it)
    elif kind == "img":
        cap, path = val
        doc.add_picture(str(OUT / path), width=Inches(6.4))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cp = doc.add_paragraph(); cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        rr = cp.add_run(cap); rr.italic = True; rr.font.size = Pt(9)
    elif kind == "table":
        tb = doc.add_table(rows=len(val), cols=len(val[0]))
        tb.style = "Table Grid"
        tb.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, row in enumerate(val):
            for j, cell in enumerate(row):
                c = tb.cell(i, j)
                c.text = ""
                add_runs(c.paragraphs[0], cell, size=8.5 if len(val[0]) > 5 else 9.5)
                if i == 0:
                    shade(c, "DCE6F1")
                    for rn in c.paragraphs[0].runs:
                        rn.bold = True
        doc.add_paragraph()
foot = sec.footer.paragraphs[0]
foot.alignment = WD_ALIGN_PARAGRAPH.CENTER
fr = foot.add_run("Brain Tumor MRI Generation and Classification (IA3): page ")
for tag, txt in (("begin", None), (None, "PAGE"), ("end", None)):
    el = OxmlElement("w:fldChar" if tag else "w:instrText")
    if tag:
        el.set(qn("w:fldCharType"), tag)
    else:
        el.set(qn("xml:space"), "preserve"); el.text = txt
    foot.add_run()._r.append(el)
versioned(OUT / "FINAL_REPORT.docx")
doc.save(OUT / "FINAL_REPORT.docx")

# ------------------------------------------------------------------ PDF (reportlab)
import matplotlib as _mpl
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image as RLImage, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

fdir = Path(_mpl.get_data_path()) / "fonts/ttf"
for nm, fn in (("DV", "DejaVuSans.ttf"), ("DV-B", "DejaVuSans-Bold.ttf"), ("DV-I", "DejaVuSans-Oblique.ttf"), ("DV-BI", "DejaVuSans-BoldOblique.ttf"), ("DVM", "DejaVuSansMono.ttf")):
    pdfmetrics.registerFont(TTFont(nm, str(fdir / fn)))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")
NAVY = colors.HexColor("#1F3A5F")
S = {
    "p": ParagraphStyle("p", fontName="DV", fontSize=9, leading=12.6, spaceAfter=5),
    "q": ParagraphStyle("q", fontName="DV-I", fontSize=8.6, leading=12, leftIndent=12, textColor=colors.HexColor("#555555"), spaceAfter=6, borderPadding=3),
    "h1": ParagraphStyle("h1", fontName="DV-B", fontSize=15, leading=19, textColor=NAVY, spaceBefore=12, spaceAfter=6),
    "h2": ParagraphStyle("h2", fontName="DV-B", fontSize=12, leading=16, textColor=NAVY, spaceBefore=10, spaceAfter=4),
    "h3": ParagraphStyle("h3", fontName="DV-B", fontSize=10, leading=14, textColor=NAVY, spaceBefore=8, spaceAfter=3),
    "li": ParagraphStyle("li", fontName="DV", fontSize=9, leading=12.6, leftIndent=14, bulletIndent=3, spaceAfter=2),
    "cap": ParagraphStyle("cap", fontName="DV-I", fontSize=8, leading=10.5, alignment=TA_CENTER, spaceAfter=8),
    "t": ParagraphStyle("t", fontName="DV", fontSize=7.6, leading=9.6),
    "th": ParagraphStyle("th", fontName="DV-B", fontSize=7.6, leading=9.6),
    "title": ParagraphStyle("title", fontName="DV-B", fontSize=20, leading=26, alignment=TA_CENTER, textColor=NAVY, spaceAfter=10),
    "sub": ParagraphStyle("sub", fontName="DV", fontSize=12, leading=16, alignment=TA_CENTER, spaceAfter=24),
    "ph": ParagraphStyle("ph", fontName="DV", fontSize=10.5, leading=17, alignment=TA_CENTER),
}


def rl(text):
    out = ""
    for t_, st in inline(text):
        t_ = t_.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if "b" in st:
            t_ = f"<b>{t_}</b>"
        if "i" in st:
            t_ = f"<i>{t_}</i>"
        if "c" in st:
            t_ = f'<font name="DVM" size="7.6">{t_}</font>'
        out += t_
    return out


W = A4[0] - 4 * cm
story = [Spacer(1, 4 * cm), Paragraph(rl(TITLE.split(":")[0]), S["title"]), Paragraph("IA3 project report: final, de-duplicated evaluation", S["sub"])]
for k, v in PLACE:
    story.append(Paragraph(f'<b>{k}:</b> <font color="#995500">{v}</font>', S["ph"]))
story.append(PageBreak())
for kind, val in blocks:
    if kind in ("h1", "h2", "h3"):
        story.append(Paragraph(rl(val), S[kind]))
    elif kind == "p":
        story.append(Paragraph(rl(val), S["p"]))
    elif kind == "quote":
        story.append(Paragraph(rl(val), S["q"]))
    elif kind in ("ul", "ol"):
        for n, it in enumerate(val, 1):
            story.append(Paragraph(rl(it), S["li"], bulletText="•" if kind == "ul" else f"{n}."))
    elif kind == "img":
        cap, path = val
        w0, h0 = Image.open(OUT / path).size
        w = min(W, 15.5 * cm)
        h = w * h0 / w0
        if h > 11.5 * cm:
            h, w = 11.5 * cm, 11.5 * cm * w0 / h0
        story += [RLImage(str(OUT / path), width=w, height=h), Paragraph(rl(cap), S["cap"])]
    elif kind == "table":
        ncol = len(val[0])
        plain = lambda c: re.sub(r"[*`]", "", c)
        CH = 4.7                                           # approx. points per character at 7.6 pt
        minw, des = [], []
        for j in range(ncol):
            toks = []
            for r_ in val:
                c_ = plain(r_[j])
                toks += [c_] if len(c_) <= 24 else c_.split()
            minw.append(max(len(t_) for t_ in toks) * 5.6 + 9)
            des.append(min(max(len(plain(r_[j])) for r_ in val), 48) * CH + 9)
        if sum(des) <= W:
            widths = [d_ * W / sum(des) for d_ in des]
        elif sum(minw) >= W:
            widths = [m_ * W / sum(minw) for m_ in minw]
        else:
            extra = [max(d_ - m_, 0) for d_, m_ in zip(des, minw)]
            share = (W - sum(minw)) / max(sum(extra), 1e-9)
            widths = [m_ + e_ * share for m_, e_ in zip(minw, extra)]
        data = [[Paragraph(rl(c), S["th"] if i == 0 else S["t"]) for c in row] for i, row in enumerate(val)]
        tb = Table(data, colWidths=widths, repeatRows=1)
        tb.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE6F1")),
                                ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3), ("TOPPADDING", (0, 0), (-1, -1), 2),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
        story += [tb, Spacer(1, 7)]


def footer(canvas, doc_):
    canvas.saveState()
    canvas.setFont("DV", 7.5)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawCentredString(A4[0] / 2, 1.1 * cm, f"Brain Tumor MRI Generation and Classification (IA3)  |  page {doc_.page}")
    canvas.restoreState()


versioned(OUT / "FINAL_REPORT.pdf")
SimpleDocTemplate(str(OUT / "FINAL_REPORT.pdf"), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=1.8 * cm,
                  title="Brain Tumor MRI Generation and Classification: IA3 report", author="Ajai Kumar N").build(story, onFirstPage=footer, onLaterPages=footer)

# ------------------------------------------------------------------ PPTX
from pptx import Presentation
from pptx.dml.color import RGBColor as PC
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Emu, Inches as PI, Pt as PP

NAVY_P, TEAL, AMBER, GREY, LIGHT = PC(0x1F, 0x3A, 0x5F), PC(0x2A, 0x9D, 0x8F), PC(0xE9, 0xA2, 0x3B), PC(0x33, 0x33, 0x33), PC(0xEE, 0xF3, 0xF8)
prs = Presentation()
prs.slide_width, prs.slide_height = PI(13.333), PI(7.5)
BL = prs.slide_layouts[6]
overflow = []
SLIDES = []


def est_lines(text, width_in, size):
    cpl = max(1, int(width_in * 72 / (size * 0.52)))
    return sum(max(1, math.ceil(len(par) / cpl)) for par in text.split("\n"))


def textbox(slide, x, y, w, h, paras, size=16, color=GREY, bold_first=False, bullet=True, fill=None, line=None, anchor=MSO_ANCHOR.TOP):
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, PI(x), PI(y), PI(w), PI(h)) if (fill or line) else slide.shapes.add_textbox(PI(x), PI(y), PI(w), PI(h))
    if fill or line:
        shp.fill.solid(); shp.fill.fore_color.rgb = fill or PC(255, 255, 255)
        if line:
            shp.line.color.rgb = line; shp.line.width = PP(1.5)
        else:
            shp.line.fill.background()
        shp.shadow.inherit = False
    tf = shp.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = PI(0.1)
    tf.margin_top = tf.margin_bottom = PI(0.05)
    tot = 0.0
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        txt = para if isinstance(para, str) else para[0]
        sz = size if isinstance(para, str) else para[1]
        r = p.add_run()
        r.text = ("• " if bullet and not (bold_first and i == 0) else "") + txt
        r.font.size = PP(sz)
        r.font.color.rgb = color
        r.font.bold = bool(bold_first and i == 0)
        r.font.name = "Calibri"
        p.space_after = PP(4)
        if (fill or line) and bullet and not bold_first:
            p.alignment = PP_ALIGN.LEFT
        tot += est_lines(r.text, w - 0.25, sz) * sz * 1.2 / 72 + 4 / 72
    if tot > h + 0.05:
        overflow.append((len(SLIDES), round(tot, 2), h, paras[0] if isinstance(paras[0], str) else paras[0][0]))
    return shp


def new_slide(title, subtitle=None):
    s = prs.slides.add_slide(BL)
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, PI(1.0))
    bar.fill.solid(); bar.fill.fore_color.rgb = NAVY_P; bar.line.fill.background()
    tb = s.shapes.add_textbox(PI(0.5), PI(0.12), PI(12.3), PI(0.8))
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    r = tf.paragraphs[0].add_run(); r.text = title; r.font.size = PP(28); r.font.bold = True; r.font.color.rgb = PC(255, 255, 255); r.font.name = "Calibri"
    ft = s.shapes.add_textbox(PI(0.5), PI(7.05), PI(12.3), PI(0.35))
    fr_ = ft.text_frame.paragraphs[0].add_run()
    fr_.text = f"Brain Tumor MRI Generation and Classification (IA3)   |   {len(SLIDES) + 1}"; fr_.font.size = PP(10); fr_.font.color.rgb = PC(0x88, 0x88, 0x88)
    SLIDES.append(s)
    return s


def picture(slide, path, x, y, w, h):
    iw, ih = Image.open(path).size
    sc = min(w / iw, h / ih)
    pw, ph = iw * sc, ih * sc
    slide.shapes.add_picture(str(path), PI(x + (w - pw) / 2), PI(y + (h - ph) / 2), PI(pw), PI(ph))


def table(slide, rows, x, y, w, h, size=12, colw=None):
    shp = slide.shapes.add_table(len(rows), len(rows[0]), PI(x), PI(y), PI(w), PI(h))
    tbl = shp.table
    if colw:
        for j, cw in enumerate(colw):
            tbl.columns[j].width = PI(cw)
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            c = tbl.cell(i, j)
            c.text = str(cell)
            for p in c.text_frame.paragraphs:
                for r in p.runs:
                    r.font.size = PP(size); r.font.name = "Calibri"; r.font.bold = (i == 0)
                    r.font.color.rgb = PC(255, 255, 255) if i == 0 else GREY
            c.fill.solid(); c.fill.fore_color.rgb = NAVY_P if i == 0 else (LIGHT if i % 2 else PC(255, 255, 255))
            c.margin_left = c.margin_right = PI(0.06); c.margin_top = c.margin_bottom = PI(0.03)
    return shp


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


# 1 title
s = prs.slides.add_slide(BL); SLIDES.append(s)
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height); bg.fill.solid(); bg.fill.fore_color.rgb = NAVY_P; bg.line.fill.background()
textbox(s, 0.8, 1.5, 11.7, 2.2, ["Brain Tumor MRI Image Generation and Classification Using a Hybrid Conditional WGAN-GP"], size=38, color=PC(255, 255, 255), bullet=False, bold_first=True)
textbox(s, 0.8, 3.7, 11.7, 0.6, ["IA3 project presentation: final, de-duplicated evaluation"], size=20, color=PC(0xCF, 0xDD, 0xEE), bullet=False)
textbox(s, 0.8, 4.6, 11.7, 2.2, ["Student name: Ajai Kumar N", "Register number: URK23CS5064", "Programme: B.Tech in Computer Science and Engineering (specialization: Artificial Intelligence)", "Institution: Karunya Institute of Technology and Sciences", "Guide / faculty: [to be filled in]   |   Date: [to be filled in]"],
        size=16, color=PC(0xF2, 0xC9, 0x7A), bullet=False)
notes(s, "Team details are placeholders to be completed by the student; none were invented.")

# 2 problem
s = new_slide("Healthcare problem and research questions")
textbox(s, 0.6, 1.3, 12.1, 3.65, ["Brain tumour MRI classification needs many labelled images, but medical data are scarce, imbalanced and hard to share.",
                                "GANs are often proposed to create synthetic training images; we ask whether a conditional WGAN-GP can do this for four classes: glioma, meningioma, pituitary tumour and no tumour.",
                                "Q1: Does the proposed hybrid conditional WGAN-GP generate MRI-like images of better distributional quality than DCGAN and WGAN-GP (FID / KID)?",
                                "Q2: Does adding its images to the training set help a CNN classifier once training size is controlled for?",
                                "Academic comparison on 2-D slices (64×64 grayscale), not a clinical tool."], size=19)
textbox(s, 0.6, 5.2, 12.1, 1.4, ["Honest framing: the results below answer these questions as measured; the proposed model is evaluated, not assumed to win."], size=18, bullet=False, fill=LIGHT, line=TEAL)
notes(s, "Both questions are answered with measured results; the outcome is that DCGAN had the best FID and that a matched-size real-only control matched or beat synthetic augmentation.")

# 3 dataset
s = new_slide("Dataset, class counts and de-duplication")
textbox(s, 0.5, 1.2, 7.1, 5.6, [f"Source: Kaggle brain-tumour MRI data (4 classes). Matches Kaggle 'Brain Tumor MRI Dataset' v2 (M. Nickparvar, CC BY 4.0) by counts and size; original download record UNVERIFIED.",
                                f"Original folders: 7,200 images. Audit: 1,076 of 1,600 original test images had a near-identical (often mirrored) copy in training; exact hashing missed them.",
                                f"Fix: pool all images, link near-duplicates (32×32 cosine similarity over 8 orientations, ≥ {split['threshold']}), connected components → {split['n_clusters']:,} clusters.",
                                f"Quarantined {split['quarantined_images']} images (label-conflict clusters, augmented files, 4 residual near-copies).",
                                f"Result: {split['representatives']:,} unique images, split by cluster (no cluster crosses partitions)."], size=15)
table(s, [["Partition", "glioma", "mening.", "no tumour", "pituit.", "total"]] + [[p] + [cnt[p][c] for c in CL] + [sum(cnt[p].values())] for p in ("train", "val", "test")], 7.9, 1.5, 5.0, 1.9, size=13)
textbox(s, 7.9, 3.7, 5.0, 2.9, [f"Only {split['representatives']:,} of 7,200 images are unique and usable.", f"No-tumour has just {cnt['train']['notumor']} training and {cnt['test']['notumor']} test images, so its metrics are very uncertain.",
                                "Same-patient slices cannot be detected (no patient IDs)."], size=14, fill=PC(0xFD, 0xF1, 0xDC), line=AMBER)
notes(s, "Provenance: no download URL was recorded locally. The data match the Kaggle page kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset (v2, CC BY 4.0): 7,200 files, 1,400/400 per class, 167.85 MB. Strong match, not proof of identical files; original download event unverified.")

# 4 methodology
s = new_slide("Proposed methodology")
steps = ["Pool 7,200 images", f"Near-duplicate clusters\n{split['n_clusters']:,}", f"Cluster-disjoint split\n{split['partition_totals']['train']:,} / {split['partition_totals']['val']} / {split['partition_totals']['test']}", "Train 3 GANs\n3,000 updates each", "Evaluate\nFID / KID  +  CNN A/B/C"]
for i, st in enumerate(steps):
    textbox(s, 0.45 + i * 2.55, 1.6, 2.25, 1.5, st.split("\n"), size=15, bullet=False, fill=TEAL if i < 4 else NAVY_P, color=PC(255, 255, 255), anchor=MSO_ANCHOR.MIDDLE)
    if i < 4:
        a = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, PI(0.45 + i * 2.55 + 2.27), PI(2.15), PI(0.26), PI(0.4)); a.fill.solid(); a.fill.fore_color.rgb = GREY; a.line.fill.background()
textbox(s, 0.6, 3.5, 12.1, 3.3, ["All three GANs train only on the training partition; validation images select the CNN epoch; the test partition is read only for final evaluation.",
                                "Quality: torch-fidelity FID and KID (Inception-v3 pool3) against the test partition; 3 generation seeds; real-vs-real floor.",
                                "Usefulness: the same small CNN trained on A real only, B real + cWGAN-GP images, C real resampled to B's size and class mix (controls for sample count).",
                                "Integrity: manifest and file-list hashes, per-run predictions saved, 23 automated verification checks passed."], size=18)

# 5 architectures
s = new_slide("DCGAN, WGAN-GP and the conditional WGAN-GP")
table(s, [["", "DCGAN", "WGAN-GP", "cWGAN-GP (proposed)"],
          ["Generator", "z(100) → 4 transposed-conv blocks, BatchNorm, Tanh → 64×64×1", "same generator", "label embedding (50-d) concatenated to z; same conv stack"],
          ["Critic / discriminator", "conv + BatchNorm discriminator (logit)", "conv critic, no BatchNorm (scalar score)", "projection critic: D(x,y) = ψ(h(x)) + E(y)·h(x), no BatchNorm"],
          ["Objective", "non-saturating BCE", "Wasserstein + gradient penalty (λ = 10)", "same loss; label shared by real and fake in each critic step"],
          ["Optimiser", "Adam 2e-4, β (0.5, 0.999)", "Adam 1e-4, β (0.5, 0.9)", "Adam 1e-4, β (0.5, 0.9)"],
          ["Updates (3,000 generator)", f"{gv['models']['DCGAN']['critic_or_discriminator_updates']:,} discriminator", f"{gv['models']['WGAN-GP']['critic_or_discriminator_updates']:,} critic (5 per generator)", f"{gv['models']['cWGAN-GP']['critic_or_discriminator_updates']:,} critic (5 per generator)"],
          ["Conditioning", "none (unconditional)", "none (unconditional)", "requested class label; no classifier guidance"]], 0.5, 1.3, 12.3, 4.6, size=13, colw=[2.2, 3.4, 3.2, 3.5])
textbox(s, 0.5, 6.1, 12.3, 0.8, ["Same data, seed (42), batch size (16) and preprocessing for all three. DCGAN has 5× fewer discriminator updates than the WGAN variants' critic updates, so the comparison is not compute-matched."], size=14, bullet=False)

# 6 setup
s = new_slide("Experimental setup and hardware")
textbox(s, 0.5, 1.2, 6.2, 5.6, ["Hardware: Apple MacBook Air, M4 chip, 16 GB unified memory.", "Software: PyTorch 2.14.0 on the MPS GPU backend (no CPU fallback), torchvision 0.29.1, torch-fidelity 0.4.0, Python 3.14.3.",
                                "GAN runtimes (MPS): " + ", ".join(f"{m} {v['runtime_seconds']:.0f} s" for m, v in gv["models"].items()) + " (Low Power Mode was on).",
                                "All GANs: 64×64 grayscale, z = 100, batch 16, seed 42, 3,000 generator updates; checkpoints and hashes verified.",
                                "cWGAN-GP was resumed from step 2,500 after an interrupted attempt (non-finite loss at step 2,861; cause unconfirmed)."], size=15)
textbox(s, 6.9, 1.2, 6.0, 5.6, [f"Downstream CNN: 4 conv blocks (32-64-128-256), global pooling, dropout 0.3, Adam 1e-3, batch 64, {dcfg['epochs']} epochs, no augmentation, seeds 42-44.",
                                f"B adds {4 * dcfg['synthetic_per_class']:,} synthetic images ({dcfg['synthetic_per_class']} per class); C resamples real images to B's per-class counts.",
                                f"Validation ({dcfg['n_val']} images) picks the best epoch; test ({NT}) is evaluated once afterwards.",
                                f"FID/KID: {gcfg['n_ref']} generated vs {gcfg['n_ref']} real test images per seed, 3 generation seeds, KID 100 subsets of {gcfg['kid']['subset_size_overall']}.",
                                "Uncertainty shown = std over seeds only; single GAN training run per model."], size=15)

# 7 samples
s = new_slide("Generated samples versus real test images")
picture(s, FIG / "sample_grids_real_vs_generated.png", 0.4, 1.15, 8.6, 5.8)
textbox(s, 9.2, 1.3, 3.8, 5.4, ["All three GANs produce head-shaped, MRI-like images with soft interiors.", "None shows an identifiable tumour; real images show clear lesions and fine detail.", "cWGAN-GP no-tumour row is visibly brighter; other labels look alike.",
                                "Samples are not shown to be anatomically or clinically valid."], size=14)

# 8 FID/KID
s = new_slide("FID / KID comparison (lower is better)")
table(s, [["Model", "FID", "KID ×1000"], ["Real-vs-real floor", pm(floor["fid_mean"], floor["fid_std_across_seeds"], 1), pm(floor["kid_mean"] * 1000, floor["kid_std_across_seeds"] * 1000, 1)]] +
      [[k + (" (best)" if k == rank[0] else ""), pm(GA[k]["fid_mean"], GA[k]["fid_std_across_seeds"], 1), pm(GA[k]["kid_mean"] * 1000, GA[k]["kid_std_across_seeds"] * 1000, 1)] for k in rank], 0.5, 1.3, 6.0, 2.6, size=14)
picture(s, FIG / "fid_kid_bar_chart.png", 6.7, 1.2, 6.3, 3.4)
textbox(s, 0.5, 4.1, 6.0, 2.8, [f"DCGAN achieved the best FID and KID of the three GANs.", f"The proposed cWGAN-GP is second; its margin over WGAN-GP is small ({GA['WGAN-GP']['fid_mean'] - GA['cWGAN-GP']['fid_mean']:.1f} FID).",
                                f"Reference: {gcfg['n_ref']} test images; values are not comparable with the earlier 1,600-image experiment."], size=15)
textbox(s, 6.7, 4.8, 6.3, 2.0, ["Std = generation seeds only; one training run per GAN; DCGAN used fewer updates (not compute-matched).", "FID is biased at small n; ImageNet features on 64×64 grayscale MRI are a weak proxy."], size=13, fill=PC(0xFD, 0xF1, 0xDC), line=AMBER)

# 9 conditioning
s = new_slide("Does the conditional GAN generate class-specific images?")
picture(s, FIG / "cwgangp_samples_by_class.png", 0.4, 1.2, 6.9, 4.4)
table(s, [["Class", "n", "FID", "KID ×1000"]] + [[c, pc[c]["n_generated_per_seed"], f"{pc[c]['fid_mean']:.0f}", f"{pc[c]['kid_mean'] * 1000:.0f}"] for c in CL], 7.5, 1.3, 5.4, 2.4, size=13)
textbox(s, 7.5, 3.9, 5.4, 3.0, ["Labels change the output (no-tumour images brighter) but rows for glioma, meningioma and pituitary look alike.",
                                "KID: row-wise only meningioma is closest to its own class; column-wise glioma and no-tumour are.", "No-tumour rests on only 46 real images. Class-dependent, not shown tumour-specific."], size=13)
textbox(s, 0.4, 5.8, 6.9, 1.1, ["Per-class FID/KID use 46-196 images: indicative only."], size=13, bullet=False, fill=PC(0xFD, 0xF1, 0xDC), line=AMBER)

# 10 downstream
s = new_slide("Downstream classification: conditions A, B and C")
table(s, [["Condition", "Train n", "Test accuracy", "Macro-F1", "Val. acc."]] + [[n, C[k]["n_train"], pm(C[k]["accuracy"]["mean"], C[k]["accuracy"]["std"]), pm(C[k]["macro_f1"]["mean"], C[k]["macro_f1"]["std"]), f3(C[k]["val_accuracy"]["mean"])]
                                                                                for n, k in (("A real only", "A_real_only"), ("B real + synthetic", "B_real_plus_cwgangp_synthetic"), ("C real, matched size", "C_real_only_matched_composition"))],
      0.4, 1.25, 7.4, 2.0, size=13, colw=[2.4, 0.9, 1.7, 1.4, 1.0])
picture(s, FIG / "accuracy_A_B_C.png", 0.3, 3.4, 7.7, 3.5)
textbox(s, 8.3, 1.25, 4.7, 5.6, [f"Test set: {NT} images (3 seeds).", "C (matched-size real-only) had the highest mean accuracy and macro-F1.", f"B beat A in {b_gt_a}/3 seeds; C ≥ B in {c_ge_b}/3 seeds.", f"Mean B−C accuracy {PD['accuracy']['mean_B_minus_C']:+.3f}.",
                                f"A single run has about ±3 points of test-sampling error; B and C cannot be separated.", "No evidence that synthetic images help beyond added training volume.", "Validation accuracy (selection only) is shown separately."], size=14)

# 11 confusion
s = new_slide("Confusion matrices and class-wise findings")
picture(s, FIG / "confusion_matrices_A_B_C.png", 0.3, 1.15, 12.7, 3.3)
table(s, [["Class (test n)", "A F1 / recall", "B F1 / recall", "C F1 / recall"]] + [[f"{c} ({gcfg['test_class_counts'][i]})"] + [f"{C[k]['per_class_f1_mean'][c]:.2f} / {C[k]['per_class_recall_mean'][c]:.2f}" for k in C] for i, c in enumerate(CL)], 0.4, 4.6, 7.2, 2.3, size=12)
textbox(s, 7.8, 4.6, 5.2, 2.3, [f"Largest confusion in every condition: glioma predicted as meningioma ({', '.join(f'{v['mean_confusion_matrix'][0][1]:.0f}' for v in C.values())} of {gcfg['test_class_counts'][0]} for A, B, C).", "No-tumour estimates (n = 46) are very uncertain."], size=13)

# 12 limitations
s = new_slide("Limitations and data-integrity risks")
textbox(s, 0.5, 1.2, 12.3, 5.7, ["De-duplication is heuristic; without patient IDs, same-patient slices may still straddle partitions (patient-level independence not guaranteed).",
                                f"Small, imbalanced data: {split['representatives']:,} unique images; test n = {NT}, no-tumour test n = {cnt['test']['notumor']}.",
                                "Only 3 seeds and one training run per GAN; no significance tests; short training (3,000 updates).",
                                "Synthetic labels are requested conditions, not verified tumour classes; conditioning is weak.",
                                "The earlier experiment on the original folders (67% of test images had near-copies) is PRELIMINARY and SUPERSEDED; its numbers are not comparable.",
                                "cWGAN-GP was interrupted at step 2,861 (non-finite loss, cause unconfirmed) and resumed from step 2,500.",
                                "Dataset matched to Kaggle v2 (CC BY 4.0) but download record unverified; 8 clusters (30 images) carried conflicting class labels.", "Inception features on 64×64 grayscale MRI and small-sample FID bias; no clinical claims."], size=16, fill=PC(0xFD, 0xF1, 0xDC), line=AMBER)

# 13 conclusion
s = new_slide("Conclusion and future work")
textbox(s, 0.5, 1.2, 6.4, 5.6, ["The hybrid conditional WGAN-GP was implemented and evaluated on a de-duplicated split.", f"DCGAN achieved the best FID ({GA['DCGAN']['fid_mean']:.1f}) and KID; the hybrid was second, only slightly ahead of WGAN-GP.",
                                f"The matched-size real-only control achieved the highest mean test accuracy ({f3(Cc['accuracy']['mean'])}) versus {f3(B['accuracy']['mean'])} for real + synthetic.",
                                "The experiments show no benefit of the synthetic images beyond added training volume, and no clinically valid tumour generation."], size=17)
textbox(s, 7.1, 1.2, 5.8, 5.6, ["Future work", "Patient-level splits with patient IDs", "More unique data, more seeds and training runs, significance tests", "Longer GAN training and compute-matched baselines", "Verify synthetic label correctness (expert review)", "Confirm dataset version against your download record"], size=16, fill=LIGHT, line=TEAL, bold_first=True)

# 14 appendix
s = new_slide("Appendix: how the data-leakage problem was found")
picture(s, FIG / "original_near_duplicate_pairs.png", 0.3, 1.15, 8.4, 3.3)
textbox(s, 8.9, 1.2, 4.1, 3.4, ["Top: original Testing images; bottom: their copies in Training.", "Exact SHA-256 found none (mirrored/re-encoded), a similarity audit found 1,076 of 1,600.", "56% of CNN-validation images also had copies in CNN-training."], size=13)
textbox(s, 0.4, 4.7, 12.5, 2.2, ["Earlier results (leaky) kept as preliminary; superseded by the cluster-disjoint evaluation.",
                                "References checked on arXiv: Mirza & Osindero 1411.1784; Radford et al. 1511.06434 (DCGAN); Gulrajani et al. 1704.00028 (WGAN-GP); Heusel et al. 1706.08500 (FID); Binkowski et al. 1801.01401 (KID); Miyato & Koyama 1802.05637 (projection critic).",
                                "Dataset: Nickparvar, Brain Tumor MRI Dataset, Kaggle, CC BY 4.0 (download record unverified; slide 3)."], size=13)

assert 12 <= len(prs.slides) <= 16
versioned(OUT / "IA3_Presentation.pptx")
prs.save(OUT / "IA3_Presentation.pptx")
print(f"slides: {len(prs.slides)} | possible text overflow: {overflow}")
print("wrote:", [p.name for p in sorted(OUT.glob('FINAL_REPORT.*')) + [OUT / 'IA3_Presentation.pptx']])
