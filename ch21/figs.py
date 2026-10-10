import sys, json
sys.path.insert(0, "/home/claude/gb")
from fs2 import *
import fs2
fs2.set_out("/mnt/user-data/outputs/gb_figures_ch21")

fig, ax = plt.subplots(figsize=(13, 5.6)); fig.subplots_adjust(left=0.02, right=0.98)
ax.set_xlim(2024.3, 2028.9); ax.set_ylim(-6, 6); ax.axis("off"); ax.plot([2024.5, 2028.7], [0, 0], color=BLACK, lw=2.2)
ev = [(2024.58, "Aug 2024\nAI Act in force", 2.6), (2025.09, "Feb 2025\nprohibitions,\nAI literacy", -3.0), (2025.58, "Aug 2025\ngeneral-purpose\nmodel obligations", 3.2),
      (2026.58, "Aug 2026\nArticle 50\ntransparency", -3.0), (2026.92, "Dec 2026\nmarking grace period\nends (existing systems)", 3.4),
      (2027.92, "Dec 2027\nhigh-risk (Annex III)", -2.6), (2028.58, "Aug 2028\nhigh-risk\n(Annex I products)", 2.6)]
for x, t, y in ev:
    ax.plot([x, x], [0, y * 0.75], color="#666", lw=1); ax.scatter(x, 0, s=70, color=L_BLUE, zorder=5)
    ax.text(x, y, t, ha="center", va="center", fontsize=12.5, fontweight="bold", color=BLACK, bbox=dict(fc="white", ec=EDGE, boxstyle="round,pad=0.35"))
for y in range(2025, 2029): ax.text(y, -0.45, str(y), ha="center", va="top", fontsize=12)
ax.text(2024.5, -5.6, "EU AI Act timeline as amended by the Digital Omnibus (reported mid-2026). Verify current dates with counsel.", fontsize=12, style="italic")
save(fig, "fig_21_1_timeline")

fig, ax = canvas(13, 6.4, 124, 58)
label(ax, 62, 55, "Transparency for AI-generated content: who must do what", fs=15, bold=True)
rows = [("Tell people they are talking to AI", "chat assistants, voice agents", C_BLUE), ("Mark synthetic content in machine-readable form", "providers of image, audio, video, text generators", C_TEAL),
        ("Disclose deepfakes", "deployers publishing realistic synthetic people, places, events", C_ROSE), ("Disclose AI text on matters of public interest", "deployers publishing unreviewed AI text", C_AMBER)]
for i, (t, w, fc) in enumerate(rows):
    y = 41 - i * 11; box(ax, 2, y, 52, 9, t, fc, fs=13.5); box(ax, 58, y, 64, 9, w, "white", fs=13, bold=False)
label(ax, 62, 2, "Muse: AI labels on drafts, signed manifests + watermarks on media, named human reviewer before publishing", fs=12.5, style="italic")
save(fig, "fig_21_2_transparency")

fig, ax = canvas(13, 6.4, 124, 58)
st = [("user prompts and uploads", C_GREY), ("provider API (retention? training use?)", C_AMBER), ("logs and traces", C_LAV), ("RAG index and caches", C_TEAL), ("feedback and fine-tuning data", C_ROSE)]
for i, (t, fc) in enumerate(st):
    x = 1 + i * 24.6; box(ax, x, 28, 21.5, 14, t, fc, fs=13)
    if i < 4: arrow(ax, (x + 21.7, 35), (x + 24.6, 35))
label(ax, 62, 54, "Where personal data can travel in a generative AI system", fs=15, bold=True)
ctrl = ["minimize; redact PII before sending", "zero-retention terms; regional endpoints", "redact; retention limits; access control", "permission filters; deletion propagates", "consent; de-identify; honor deletion"]
for i, t in enumerate(ctrl): box(ax, 1 + i * 24.6, 6, 21.5, 14, t, "white", fs=12, bold=False); arrow(ax, (11.75 + i * 24.6, 28), (11.75 + i * 24.6, 20.2))
save(fig, "fig_21_3_privacy")

fig, ax = canvas(13, 6.6, 124, 60)
label(ax, 62, 57, "Model cards and system cards", fs=15, bold=True)
box(ax, 2, 6, 56, 44, "MODEL CARD (per model)\n\nintended use and out-of-scope use\ntraining data summary\nevaluation results by slice\nknown limitations and biases\nlicense", C_BLUE, fs=13.5, bold=False)
box(ax, 66, 6, 56, 44, "SYSTEM CARD (per product)\n\npurpose and users\ncomponents, models, licenses\nrelease-gate results\nsafety tests and mitigations\ntransparency measures\nlimitations, contacts", C_TEAL, fs=13.5, bold=False)
save(fig, "fig_21_4_cards")

fig, ax = canvas(13, 6.2, 124, 56)
label(ax, 62, 53, "Lightweight governance: review effort proportional to risk", fs=15, bold=True)
tiers = [("Low", "internal tools, no external effect", "standard engineering review", C_GREEN), ("Medium", "public content, synthetic media, external actions (Muse)", "safety evals, labels, owner sign-off, system card", C_AMBER),
         ("High", "decisions about people: hiring, credit, health, education", "full review, legal sign-off, impact assessment, monitoring", C_ROSE)]
for i, (t, w, r, fc) in enumerate(tiers):
    y = 36 - i * 14; box(ax, 2, y, 16, 11, t, fc, fs=15); box(ax, 21, y, 48, 11, w, "white", fs=13, bold=False); box(ax, 72, y, 50, 11, r, "white", fs=13, bold=False)
save(fig, "fig_21_5_governance")
print("ok")
