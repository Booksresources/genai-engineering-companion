import sys, json
sys.path.insert(0, "/home/claude/gb")
from fs2 import *
import fs2
fs2.set_out("/mnt/user-data/outputs/gb_figures_ch14")
R = json.load(open("rag_results.json"))

fig, ax = canvas(13, 7.0, 124, 64)
label(ax, 2, 61, "Indexing (offline)", fs=15, bold=True, ha="left")
st = [("documents: PDFs, pages, sheets", C_GREY), ("parse + clean", C_AMBER), ("chunk with metadata", C_AMBER), ("embed + keyword index", C_LAV), ("vector + BM25 index", C_TEAL)]
for i, (t, fc) in enumerate(st):
    x = 1 + i * 24.6; box(ax, x, 44, 21.5, 12, t, fc, fs=13)
    if i < 4: arrow(ax, (x + 21.7, 50), (x + 24.6, 50))
label(ax, 2, 35, "Answering (online)", fs=15, bold=True, ha="left")
st2 = [("question", C_GREY), ("rewrite query", C_AMBER), ("hybrid search", C_TEAL), ("rerank top 20 → top 3", C_LAV), ("LLM answers with citations, or abstains", C_GREEN)]
for i, (t, fc) in enumerate(st2):
    x = 1 + i * 24.6; box(ax, x, 16, 21.5, 14, t, fc, fs=13)
    if i < 4: arrow(ax, (x + 21.7, 23), (x + 24.6, 23))
arrow(ax, (110, 44), (61, 30.2), cs="arc3,rad=0.1", ls="--"); label(ax, 92, 37.5, "index", fs=12.5, style="italic")
label(ax, 62, 6, "Evaluate retrieval (recall@k) and generation (correctness, citations, abstention) separately", fs=13, style="italic")
save(fig, "fig_14_1_architecture")

ch = R["chunking"]; names = list(ch.keys())
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.0))
a1.bar(range(5), [100 * ch[n]["recall@3"] for n in names], color=[C_GREY, C_TEAL, C_ROSE, C_BLUE, C_LAV], edgecolor=EDGE)
for i, n in enumerate(names): a1.text(i, 100 * ch[n]["recall@3"] + 0.4, f"{100*ch[n]['recall@3']:.1f}", ha="center", fontsize=13, fontweight="bold")
a1.set_ylim(90, 101); a1.set_xticks(range(5)); a1.set_xticklabels(["whole\ndoc", "heading", "sentence", "fixed", "fixed +\noverlap"], fontsize=12.5); a1.set_ylabel("Recall@3 (%)"); a1.set_title("(a) Retrieval"); clean(a1)
a2.bar(range(5), [ch[n]["chunks"] for n in names], color=[C_GREY, C_TEAL, C_ROSE, C_BLUE, C_LAV], edgecolor=EDGE)
for i, n in enumerate(names): a2.text(i, ch[n]["chunks"] + 3, f"{ch[n]['chunks']}", ha="center", fontsize=13, fontweight="bold")
a2.set_xticks(range(5)); a2.set_xticklabels(["whole\ndoc", "heading", "sentence", "fixed", "fixed +\noverlap"], fontsize=12.5); a2.set_ylabel("Number of chunks"); a2.set_title("(b) Index size"); a2.set_ylim(0, 215); clean(a2)
fig.tight_layout(w_pad=3); save(fig, "fig_14_2_chunking")

rt = R["retrievers"]; kinds = ["brand", "brand-paraphrase", "product", "all test"]; modes = ["bm25", "dense", "hybrid", "hybrid + rewrite", "hybrid + rewrite + rerank"]
fig, ax = plt.subplots(figsize=(12.5, 5.8)); x = np.arange(len(kinds)); w = 0.16; cols = [C_AMBER, C_LAV, C_TEAL, C_BLUE, C_GREEN]
for j, (m, col) in enumerate(zip(modes, cols)):
    ax.bar(x + (j - 2) * w, [100 * rt[k][m] for k in kinds], w, color=col, edgecolor=EDGE, label=m)
ax.set_xticks(x); ax.set_xticklabels([f"{k}\n(n = {rt[k]['n']})" for k in kinds], fontsize=13); ax.set_ylabel("Recall@1 (%)"); ax.set_ylim(50, 105)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=12); clean(ax)
save(fig, "fig_14_3_retrievers")

A = R["answers"]; th = [a["threshold"] for a in A]
fig, ax = plt.subplots(figsize=(11, 5.4))
for key, col, mk, lab in [("correct", L_BLUE, "o", "answered correctly"), ("citation_correct", L_TEAL, "s", "cited the right document"),
                          ("wrongly_abstained", L_RED, "^", "abstained on answerable question"), ("abstained_on_unanswerable", L_PURPLE, "D", "abstained on unanswerable question")]:
    ax.plot(th, [100 * a[key] for a in A], marker=mk, ms=9, lw=2.6, color=col, label=lab)
ax.set_xlabel("Reranker confidence threshold for answering"); ax.set_ylabel("Share of questions (%)"); ax.set_ylim(-3, 105)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=12); ax.grid(True, color="#DDD"); clean(ax)
save(fig, "fig_14_4_abstention")

fig, ax = canvas(13, 5.8, 124, 52)
label(ax, 62, 49, "Three ways to retrieve over images and PDFs", fs=15, bold=True)
opts = [("Describe, then index text", "VLM captions or extracts each image or page region; index the text; return the original image", C_AMBER),
        ("Multimodal embeddings", "embed images and text into one space (CLIP-style); search images with text queries", C_LAV),
        ("Page-image retrieval", "embed whole page images (late-interaction models); a VLM reads the retrieved pages", C_TEAL)]
for i, (t, d, fc) in enumerate(opts):
    y = 32 - i * 14; box(ax, 2, y, 32, 11, t, fc, fs=14); box(ax, 38, y, 84, 11, d, "white", fs=13.5, bold=False)
save(fig, "fig_14_5_multimodal")

fig, ax = canvas(13, 6.4, 124, 58)
box(ax, 44, 46, 36, 9, "wrong or unsupported answer", C_ROSE, fs=14)
box(ax, 6, 26, 34, 12, "Was the evidence in the top-k chunks?", C_AMBER, fs=13.5)
box(ax, 84, 26, 34, 12, "Did the answer follow the evidence?", C_AMBER, fs=13.5)
arrow(ax, (55, 46), (23, 38.2)); arrow(ax, (69, 46), (101, 38.2))
label(ax, 30, 43.5, "step 1: retrieval?", fs=13, bold=True); label(ax, 94, 43.5, "step 2: generation?", fs=13, bold=True)
box(ax, 1, 4, 44, 16, "fix: parsing, chunking, query rewriting, hybrid search, reranking, metadata filters, more documents", C_TEAL, fs=12.5, bold=False)
box(ax, 79, 4, 44, 16, "fix: prompt (answer only from context), citations, abstention threshold, model, context order", C_TEAL, fs=12.5, bold=False)
arrow(ax, (23, 26), (23, 20.2)); arrow(ax, (101, 26), (101, 20.2))
save(fig, "fig_14_6_debugging")
print("ok")
