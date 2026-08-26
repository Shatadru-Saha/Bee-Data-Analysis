import sys
import os
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import subprocess

chunk_id   = int(sys.argv[1])
data_path  = sys.argv[2]
output_dir = sys.argv[3]

with open(data_path, "rb") as f:
    payload = pickle.load(f)

df_slice      = payload["df_slice"]
frame_indices = payload["frame_indices"]
age_colors    = payload["age_colors"]
age_list      = payload["age_list"]
xpixels       = payload["xpixels"]
DAY           = payload["DAY"]
comb          = payload["comb"]
FPS           = payload["FPS"]

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import displayfunctions as bp
import definitions_2019 as bd

# ── Pre-convert grouped data to dict of numpy arrays ────────────────────────
print(f"[Chunk {chunk_id}] Pre-processing frames...", flush=True)

frames_data = {}
for frame_number, group in df_slice.groupby("framenum"):
    ages  = group["age(days)"].values
    xy    = group[["x", "y"]].values.astype(np.float32)
    theta = group["theta"].values.astype(np.float32)
    frames_data[frame_number] = (xy, ages, theta)

print(f"[Chunk {chunk_id}] Pre-processing done, starting render...", flush=True)

# ── Setup figure ─────────────────────────────────────────────────────────────
ax  = bp.showcomb(comb)
fig = ax.figure
fig.set_dpi(72)

# ── Age colorbar strip at the top ────────────────────────────────────────────
cbar_ax = fig.add_axes([0.05, 0.96, 0.90, 0.025])
n_ages = len(age_list)
for k, a in enumerate(age_list):
    color = age_colors.get(a, "grey")
    cbar_ax.add_patch(mpatches.Rectangle((k, 0), 1, 1, color=color))
    cbar_ax.text(k + 0.5, 0.5, f"{int(a)}d", ha="center", va="center",
                 fontsize=7, color="black", fontweight="bold")
cbar_ax.set_xlim(0, n_ages)
cbar_ax.set_ylim(0, 1)
cbar_ax.axis("off")
cbar_ax.set_title("Age (days)", fontsize=8, pad=2)

age_groups = sorted(df_slice["age(days)"].dropna().unique())

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

# ── One-time background capture (includes the static colorbar) ───────────────
fig.canvas.draw()
bg = fig.canvas.copy_from_bbox(fig.bbox)

# ── Render loop ───────────────────────────────────────────────────────────────
active_quivers = []

for i, frame_number in enumerate(frame_indices):
    for q in active_quivers:
        q.remove()
    active_quivers = []

    fig.canvas.restore_region(bg)

    if frame_number in frames_data:
        xy, ages, theta = frames_data[frame_number]
        for a in age_groups:
            mask = (ages == a) & np.isfinite(theta)
            if mask.any():
                color = age_colors.get(a, "grey")

                # Black outline
                q_bg = ax.quiver(
                    xy[mask, 0], xy[mask, 1],
                    np.cos(theta[mask]), np.sin(theta[mask]),
                    color="black",
                    angles="xy", scale_units="xy",
                    scale=1.0 / 90,
                    width=0.006,
                    headwidth=5, headlength=6, headaxislength=5.5,
                    alpha=1
                )
                ax.draw_artist(q_bg)
                active_quivers.append(q_bg)

                # Colored arrow on top
                q = ax.quiver(
                    xy[mask, 0], xy[mask, 1],
                    np.cos(theta[mask]), np.sin(theta[mask]),
                    color=color,
                    angles="xy", scale_units="xy",
                    scale=1.0 / 70,
                    width=0.003,
                    headwidth=4, headlength=5, headaxislength=4.5,
                    alpha=1
                )
                ax.draw_artist(q)
                active_quivers.append(q)

    fig.canvas.blit(fig.bbox)

    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    proc.stdin.write(buf.tobytes())

    if i % 500 == 0:
        print(f"[Chunk {chunk_id}] Frame {i+1}/{len(frame_indices)}", flush=True)

proc.stdin.close()
proc.wait()
plt.close(fig)
print(f"[Chunk {chunk_id}] Done → {chunk_output}", flush=True)