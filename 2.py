import numpy as np

path = r"D:/pythoncode/oneformer3d-main/work_dirs/full_vis/1_gt_sem.npy"

pred = np.load(path)

print("shape:", pred.shape)
print("dtype:", pred.dtype)

labels, counts = np.unique(pred, return_counts=True)

for label, count in zip(labels, counts):
    print(
        f"类别 {label}: {count} 点, "
        f"占比 {count / len(pred) * 100:.2f}%"
    )