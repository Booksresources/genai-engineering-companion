"""Figure style for 'Generative AI Engineering Step by Step':
large black labels, light fills, and automatic wrapping of text inside boxes."""
import os, textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

BLACK = "#000000"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 14.5, "text.color": BLACK,
    "axes.labelcolor": BLACK, "xtick.color": BLACK, "ytick.color": BLACK, "axes.edgecolor": BLACK,
    "axes.titlecolor": BLACK, "axes.labelsize": 15.5, "xtick.labelsize": 14, "ytick.labelsize": 14,
    "legend.fontsize": 13.5, "axes.titlesize": 15.5, "axes.titleweight": "bold", "text.parse_math": False,
})
C_BLUE, C_TEAL, C_AMBER, C_GREY, C_ROSE, C_LAV, C_GREEN = "#C9DAEE", "#BFE6E0", "#FBE2B6", "#E3E7EC", "#F6CFCF", "#DCD4F0", "#D5EBC8"
EDGE = "#1F3A5F"
L_BLUE, L_TEAL, L_AMBER, L_RED, L_PURPLE, L_GREEN = "#1F5FA8", "#13867A", "#C77D0A", "#B22A2A", "#6A3FA0", "#2E7D32"
OUT = None
def set_out(p):
    global OUT
    OUT = p; os.makedirs(p, exist_ok=True)

def canvas(w=13, h=6, xlim=120, ylim=60):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
    ax.set_xlim(0, xlim); ax.set_ylim(0, ylim); ax.axis("off")
    return fig, ax

def _units_to_pt(ax, w):
    fig = ax.figure
    width_in = ax.get_position().width * fig.get_figwidth()
    return w * width_in / (ax.get_xlim()[1] - ax.get_xlim()[0]) * 72
def _hunits_to_pt(ax, h):
    fig = ax.figure
    height_in = ax.get_position().height * fig.get_figheight()
    return h * height_in / (ax.get_ylim()[1] - ax.get_ylim()[0]) * 72

def wrap_text(ax, text, w, h, fs, pad=0.9, min_fs=11.5):
    """Wrap text to fit a box of w x h data units; shrink font if needed (never below min_fs)."""
    while True:
        max_chars = max(4, int((_units_to_pt(ax, w) - 2 * pad * 6) / (fs * 0.60)))
        lines = []
        for para in text.split("\n"):
            lines += textwrap.wrap(para, max_chars, break_long_words=len(para) > 0 and max(len(w) for w in para.split() or [""]) > max_chars + 4) or [""]
        need = len(lines) * fs * 1.28
        if need <= _hunits_to_pt(ax, h) - 6 or fs <= min_fs:
            return "\n".join(lines), fs
        fs -= 0.5

def box(ax, x, y, w, h, text, fc=C_BLUE, fs=14.5, bold=True, ec=EDGE, lw=1.5, r=0.8, align="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw))
    t, f = wrap_text(ax, text, w, h, fs)
    ax.text(x + w / 2 if align == "center" else x + 1.2, y + h / 2, t, ha=align, va="center", fontsize=f,
            color=BLACK, fontweight="bold" if bold else "normal", linespacing=1.2)

def note(ax, x, y, w, h, text, fc=C_GREY, fs=12, bold=False):
    box(ax, x, y, w, h, text, fc=fc, fs=fs, bold=bold)

def arrow(ax, p1, p2, cs="arc3,rad=0", lw=1.7, color=BLACK, ls="-", ms=16):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=ms, color=color, connectionstyle=cs, lw=lw, linestyle=ls))

def label(ax, x, y, text, fs=14, bold=False, ha="center", va="center", width=None, style="normal"):
    if width:
        text = "\n".join(textwrap.wrap(text, width))
    ax.text(x, y, text, fontsize=fs, ha=ha, va=va, color=BLACK, fontweight="bold" if bold else "normal", style=style, linespacing=1.2)

def clean(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.15)
    plt.close(fig)

# ---- Print-size scaling: figures are placed at ~6.5 in wide, so render them smaller
# (fonts therefore appear larger on the printed page).
SCALE = 0.78
_orig_subplots, _orig_figure = plt.subplots, plt.figure
def _scaled(fs):
    return (fs[0] * SCALE, fs[1] * SCALE) if fs else fs
def subplots(*a, figsize=None, **k): return _orig_subplots(*a, figsize=_scaled(figsize), **k)
def figure(*a, figsize=None, **k): return _orig_figure(*a, figsize=_scaled(figsize), **k)
plt.subplots, plt.figure = subplots, figure
