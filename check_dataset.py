import numpy as np
import matplotlib.pyplot as plt
import json, os, glob

dataset_dir = "dataset"
images_dir  = os.path.join(dataset_dir, "images")
labels_dir  = os.path.join(dataset_dir, "labels")

# 随机挑4张（选不同场点）
samples = ["sample_00000", "sample_00200", "sample_00500", "sample_00900"]

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
for ax, name in zip(axes.flatten(), samples):
    img  = np.load(os.path.join(images_dir, f"{name}.npy")).astype(np.float32)
    with open(os.path.join(labels_dir, f"{name}.json")) as f:
        lab = json.load(f)

    ax.imshow(img, cmap="gray", origin="lower",
              vmin=np.percentile(img, 1), vmax=np.percentile(img, 99))
    ax.set_title(f"{name}\nr={lab['r_norm']:.1f} snr={lab['snr_level']} "
                 f"n_fiber={lab['n_fibers']}", fontsize=9)
    # 画出基准光纤（红）和待测光纤（蓝）
    for fib in lab["fibers"]:
        c = "red" if fib["is_calib"] else "cyan"
        ax.plot(fib["true_x_px"], fib["true_y_px"], ".", color=c,
                markersize=2, alpha=0.6)

plt.tight_layout()
plt.savefig("check_dataset.png", dpi=150)
plt.show()
print("已保存 check_dataset.png")