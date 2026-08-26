import sys
import os
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import numpy as np
import subprocess

chunk_id   = int(sys.argv[1])
data_path  = sys.argv[2]
output_dir = sys.argv[3]

with open(data_path, "rb") as f:
    payload = pickle.load(f)

df_slice       = payload["df_slice"]
frame_indices  = payload["frame_indices"]
age_colors     = payload["age_colors"]
age_list       = payload["age_list"]
xpixels        = payload["xpixels"]
DAY            = payload["DAY"]
comb           = payload["comb"]
FPS            = payload["FPS"]
chunk_uid_map  = payload["chunk_uid_map"]
score_lookup   = payload["score_lookup"]
FRAME_START    = payload["FRAME_START"]
CHUNK_SIZE     = payload["CHUNK_SIZE"]
BOX_X          = payload["BOX_X"]
BOX_Y          = payload["BOX_Y"]

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import displayfunctions as bp
import definitions_2019 as bd

# ── Tier colors ───────────────────────────────────────────────────────────────
TIER_COLORS = {
    0: "#FFD700",   # gold   — top 10
    1: "#00FFFF",   # cyan   — next 10 (rank 11-20)
    2: "#FF69B4",   # pink   — last 10 (rank 21-30)
}
TIER_LABELS = {
    0: "Top 10",
    1: "Rank 11–20",
    2: "Rank 21–30",
}
TOP_N = 30

# Build per-chunk uid->tier lookup (0, 1, or 2)
# top_per_chunk is sorted by festoon_score ascending, so head(30) gives ranks 1-30
# We need rank within chunk to assign tier
tier_lookup = {}   # (chunk_idx, uid) -> tier index
for chunk_idx, grp in payload.get("top_per_chunk_full", {}).items():
    pass  # handled below via chunk_uid_map ordering

# chunk_uid_map already stores uids in score order (sorted ascending in notebook)
# so index 0-9 = top 10, 10-19 = next 10, 20-29 = last 10
def get_tier(chunk_idx, uid):
    uids = chunk_uid_map.get(chunk_idx, [])
    try:
        rank = uids.index(uid)
        return rank // 10
    except ValueError:
        return None

# ── Pre-process frames ────────────────────────────────────────────────────────
print(f"[Chunk {chunk_id}] Pre-processing frames...", flush=True)

frames_data = {}
for frame_number, group in df_slice.groupby("framenum"):
    uids  = group["uid"].values
    xy    = group[["x", "y"]].values.astype(np.float32)
    theta = group["theta"].values.astype(np.float32)
    frames_data[frame_number] = (uids, xy, theta)

print(f"[Chunk {chunk_id}] Pre-processing done, starting render...", flush=True)

# ── Setup figure ──────────────────────────────────────────────────────────────
ax  = bp.showcomb(comb)
fig = ax.figure
fig.set_dpi(72)

# ── Tier colorbar at top (replaces age colorbar) ──────────────────────────────
cbar_ax = fig.add_axes([0.05, 0.96, 0.90, 0.025])
tier_entries = [
    ("#888888", "Non-festooning"),
    (TIER_COLORS[0], "Top 10 festooners"),
    (TIER_COLORS[1], "Rank 11–20"),
    (TIER_COLORS[2], "Rank 21–30"),
]
n_entries = len(tier_entries)
for k, (color, label) in enumerate(tier_entries):
    cbar_ax.add_patch(mpatches.Rectangle((k, 0), 1, 1, color=color))
    cbar_ax.text(k + 0.5, 0.5, label, ha="center", va="center",
                 fontsize=7, color="black", fontweight="bold")
cbar_ax.set_xlim(0, n_entries)
cbar_ax.set_ylim(0, 1)
cbar_ax.axis("off")
cbar_ax.set_title("Festoon rank", fontsize=8, pad=2)

# ── Bounding box (baked into background) ──────────────────────────────────────
ax.add_patch(Rectangle(
    (BOX_X[0], BOX_Y[0]), BOX_X[1]-BOX_X[0], BOX_Y[1]-BOX_Y[0],
    linewidth=2, edgecolor="green", facecolor="none", zorder=5
))

# ── ffmpeg pipe ───────────────────────────────────────────────────────────────
width  = int(fig.get_figwidth() * fig.dpi)
height = int(fig.get_figheight() * fig.dpi)
chunk_output = os.path.join(output_dir, f"chunk_{chunk_id:04d}.mp4")

ffmpeg_cmd = [
    "ffmpeg", "-y",
    "-f", "rawvideo", "-vcodec", "rawvideo",
    "-s", f"{width}x{height}",
    "-pix_fmt", "rgb24", "-r", str(FPS),
    "-i", "pipe:0",
    "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
    chunk_output
]

proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

fig.canvas.draw()
bg = fig.canvas.copy_from_bbox(fig.bbox)

# ── Render loop ───────────────────────────────────────────────────────────────
active_artists = []

for i, frame_number in enumerate(frame_indices):
    for artist in active_artists:
        artist.remove()
    active_artists = []

    fig.canvas.restore_region(bg)

    if frame_number in frames_data:
        uids, xy, theta = frames_data[frame_number]

        chunk_idx   = (frame_number - FRAME_START) // CHUNK_SIZE
        ranked_uids = chunk_uid_map.get(chunk_idx, [])[:TOP_N]

        # Build uid -> tier dict for this chunk
        uid_to_tier = {}
        for rank, uid in enumerate(ranked_uids):
            uid_to_tier[uid] = rank // 10   # 0, 1, or 2

        all_cand_uids = set(ranked_uids)
        finite_mask   = np.isfinite(theta)

        # ── Background bees (grey, full opacity) ─────────────────────────────
        bg_mask = ~np.isin(uids, list(all_cand_uids)) & finite_mask
        if bg_mask.any():
            q_bg = ax.quiver(
                xy[bg_mask, 0], xy[bg_mask, 1],
                np.cos(theta[bg_mask]), np.sin(theta[bg_mask]),
                color="#888888",
                angles="xy", scale_units="xy",
                scale=1.0/70, width=0.003,
                headwidth=4, headlength=5, headaxislength=4.5,
                alpha=1.0
            )
            ax.draw_artist(q_bg)
            active_artists.append(q_bg)

        # ── Festoon candidates per tier ───────────────────────────────────────
        for tier in range(3):
            tier_uids = [uid for uid, t in uid_to_tier.items() if t == tier]
            if not tier_uids:
                continue
            tmask = np.isin(uids, tier_uids) & finite_mask
            if not tmask.any():
                continue

            color = TIER_COLORS[tier]

            # Black outline
            q_out = ax.quiver(
                xy[tmask, 0], xy[tmask, 1],
                np.cos(theta[tmask]), np.sin(theta[tmask]),
                color="black",
                angles="xy", scale_units="xy",
                scale=1.0/90, width=0.006,
                headwidth=5, headlength=6, headaxislength=5.5,
                alpha=1
            )
            ax.draw_artist(q_out)
            active_artists.append(q_out)

            # Tier-colored arrow on top
            q_tier = ax.quiver(
                xy[tmask, 0], xy[tmask, 1],
                np.cos(theta[tmask]), np.sin(theta[tmask]),
                color=color,
                angles="xy", scale_units="xy",
                scale=1.0/70, width=0.003,
                headwidth=4, headlength=5, headaxislength=4.5,
                alpha=1, zorder=6 + tier
            )
            ax.draw_artist(q_tier)
            active_artists.append(q_tier)

    fig.canvas.blit(fig.bbox)

    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    proc.stdin.write(buf.tobytes())

    if i % 100 == 0:
        print(f"[Chunk {chunk_id}] Frame {i+1}/{len(frame_indices)}", flush=True)

proc.stdin.close()
proc.wait()
plt.close(fig)
print(f"[Chunk {chunk_id}] Done → {chunk_output}", flush=True)