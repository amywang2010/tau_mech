"""Create the mentor-facing top-results contact sheet.

This script only composites existing PNG artifacts; it does not recompute,
rescale scientifically, or edit any underlying result. The SPH sweep panel is
retained as an explicitly flagged open/blocked result because the report
identifies zero-shear drift in that sweep.
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "top_18_results.png"

PANELS = [
    ("EDA · Radius of gyration", "outputs/figures/fig_rg_distribution.png"),
    ("EDA · End-to-end distance", "outputs/figures/fig_e2e_distribution.png"),
    ("EDA · Total SASA", "outputs/figures/fig_total_sasa_distribution.png"),
    ("EDA · APR relative SASA", "outputs/figures/fig_rsa_profile.png"),
    ("EDA · Contact maps", "outputs/figures/fig_contact_maps.png"),
    ("EDA · Graph degree", "outputs/figures/fig_degree_distribution.png"),
    ("EDA · Graph density", "outputs/figures/fig_graph_density.png"),
    ("EDA · Residue flexibility", "outputs/figures/fig_flexibility.png"),
    ("EDA · Ensemble sizes", "outputs/figures/fig_ensemble_sizes.png"),
    ("EDA · Residue frequency", "outputs/figures/fig_residue_frequency.png"),
    ("EDA · Charge/hydropathy", "outputs/figures/fig_charge_hydropathy.png"),
    ("EDA · PCA structural embedding", "outputs/figures/fig_embedding_pca.png"),
    ("EDA · t-SNE structural embedding", "outputs/figures/fig_embedding_tsne.png"),
    ("GNN · PR/ROC curves", "outputs/gnn/pr_roc_curves.png"),
    ("GNN · Training curves", "outputs/gnn/training_curves.png"),
    ("GNN · Transfer matrix", "outputs/gnn/transfer_matrix.png"),
    ("GNN · Supplementary ablations", "outputs/gnn/supplement_ablations.png"),
    ("SPH · deformation sweep [OPEN: drift-affected]", "outputs/sph/sph_deformation_vs_Ca.png"),
]

BG = (250, 251, 253)
INK = (24, 31, 42)
MUTED = (88, 99, 115)
ACCENT = (175, 44, 44)
COLS, CELL_W, CELL_H = 3, 660, 410
HEADER = 110
ROWS = (len(PANELS) + COLS - 1) // COLS
canvas = Image.new("RGB", (COLS * CELL_W, HEADER + ROWS * CELL_H), BG)
draw = ImageDraw.Draw(canvas)
try:
    title_font = ImageFont.truetype("arial.ttf", 30)
    subtitle_font = ImageFont.truetype("arial.ttf", 16)
    label_font = ImageFont.truetype("arial.ttf", 15)
except OSError:
    title_font = subtitle_font = label_font = ImageFont.load_default()

draw.text((24, 18), "Tau mechanobiology — top 18 existing results", fill=INK, font=title_font)
draw.text((24, 57), "Composite of repository PNG artifacts; no underlying result was modified. Red flag marks the documented open SPH sweep issue.", fill=MUTED, font=subtitle_font)

for i, (label, rel) in enumerate(PANELS):
    x = (i % COLS) * CELL_W
    y = HEADER + (i // COLS) * CELL_H
    path = ROOT / rel
    im = Image.open(path).convert("RGB")
    im.thumbnail((CELL_W - 24, CELL_H - 58), Image.Resampling.LANCZOS)
    px = x + (CELL_W - im.width) // 2
    py = y + 30 + (CELL_H - 42 - im.height) // 2
    canvas.paste(im, (px, py))
    draw.text((x + 12, y + 8), f"{i + 1:02d}  {label}", fill=ACCENT if "OPEN" in label else INK, font=label_font)
    draw.rectangle((x, y, x + CELL_W - 1, y + CELL_H - 1), outline=(215, 220, 228), width=1)

canvas.save(OUT, format="PNG", optimize=True)
print(OUT)
print(canvas.size)
