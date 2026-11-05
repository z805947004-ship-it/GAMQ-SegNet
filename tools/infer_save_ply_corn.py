import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import open3d as o3d

# =========================================================
# 保证可以 import oneformer3d
# 要从项目根目录运行：
# cd /d D:\pythoncode\oneformer3d-main
# =========================================================
PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import oneformer3d  # noqa: F401
except Exception as e:
    print("[警告] import oneformer3d 失败，请确认你在项目根目录运行。")
    print("当前工作目录:", PROJECT_ROOT)
    print("错误:", repr(e))


# =========================================================
# 屏蔽非致命 warning
# =========================================================
warnings.filterwarnings("ignore", message=".*FileClient.*")
warnings.filterwarnings("ignore", message=".*HardDiskBackend.*")
warnings.filterwarnings("ignore", message="Mean of empty slice.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide.*", category=RuntimeWarning)

try:
    from numba.core.errors import NumbaDeprecationWarning
    warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
except Exception:
    pass


# =========================================================
# 工具函数
# =========================================================
def to_numpy(x):
    """把 torch.Tensor / BasePoints / numpy 转成 numpy；遇到复杂 list/tuple 时返回 None。"""
    if x is None:
        return None

    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()

    if hasattr(x, "tensor"):
        t = x.tensor
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().numpy()
        try:
            return np.asarray(t)
        except Exception:
            return None

    # 关键修改：不要直接 np.asarray(list/tuple)，否则会遇到异构结构报错
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return None

        if len(x) == 1:
            return to_numpy(x[0])

        arrs = []
        for item in x:
            arr = to_numpy(item)
            if arr is None:
                return None
            arrs.append(arr)

        shapes = [a.shape for a in arrs]
        if len(set(map(str, shapes))) == 1:
            try:
                return np.stack(arrs, axis=0)
            except Exception:
                return None

        return None

    try:
        return np.asarray(x)
    except Exception:
        return None

def safe_name(name):
    """把 sample_id 转成安全文件名。"""
    name = str(name)
    keep = []
    for c in name:
        if c.isalnum() or c in ["-", "_", "."]:
            keep.append(c)
        else:
            keep.append("_")
    return "".join(keep)


def get_field(container, names):
    """从 PointData / InstanceData / dict 中安全取字段。"""
    if container is None:
        return None

    for name in names:
        try:
            if hasattr(container, name):
                v = getattr(container, name)
                if v is not None:
                    return v
        except Exception:
            pass

        try:
            if hasattr(container, "get"):
                v = container.get(name, None)
                if v is not None:
                    return v
        except Exception:
            pass

    return None


def field_names(container):
    if container is None:
        return []
    try:
        return list(container.keys())
    except Exception:
        pass
    try:
        return [k for k in dir(container) if not k.startswith("_")]
    except Exception:
        return []


def get_sample_name(data_sample, index):
    """尽量从 data_sample.metainfo 里拿 sample_id。"""
    meta = {}

    try:
        meta = data_sample.metainfo
    except Exception:
        meta = {}

    candidate_keys = [
        "sample_id",
        "sample_idx",
        "lidar_idx",
        "pcd_name",
        "pts_filename",
        "point_cloud_path",
        "file_name",
    ]

    for k in candidate_keys:
        if k in meta:
            value = meta[k]
            value = str(value)
            value = os.path.basename(value)
            value = value.replace("_pts.npy", "")
            value = value.replace(".npy", "")
            value = value.replace(".bin", "")
            return value

    return f"sample_{index:04d}"


def get_points_from_batch(data_batch, batch_index=0):
    """从 dataloader batch 里拿原始点坐标。"""
    inputs = data_batch.get("inputs", None)
    if inputs is None:
        return None

    points = inputs.get("points", None)
    if points is None:
        return None

    if isinstance(points, (list, tuple)):
        pts = points[batch_index]
    else:
        pts = points

    pts = to_numpy(pts)
    if pts is None:
        return None

    return pts[:, :3].astype(np.float64)


def color_by_semantic(labels):
    """
    当前训练配置：
        0 = stem
        1 = leaf
    """
    labels = np.asarray(labels).astype(np.int64).reshape(-1)

    palette = {
        0: [0.2, 0.8, 0.2],   # stem: 绿色
        1: [1.0, 0.6, 0.1],   # leaf: 橙色
        2: [0.5, 0.5, 0.5],   # unlabeled: 灰色，如果存在
        -1: [0.5, 0.5, 0.5],
    }

    colors = np.zeros((len(labels), 3), dtype=np.float64)
    for i, lab in enumerate(labels):
        colors[i] = palette.get(int(lab), [1.0, 0.0, 1.0])  # 未知类别：紫色

    return colors


def color_by_instance(labels):
    """不同实例随机颜色，-1/0 作为背景时显示灰色。"""
    labels = np.asarray(labels).astype(np.int64).reshape(-1)

    colors = np.ones((len(labels), 3), dtype=np.float64) * 0.6
    unique_ids = np.unique(labels)

    rng = np.random.RandomState(42)

    for uid in unique_ids:
        if uid < 0:
            continue

        # 如果你的实例 id 里 0 是有效实例，也会被上色
        # 如果你希望 0 当背景，可以把下面两行取消注释
        # if uid == 0:
        #     continue

        color = rng.rand(3)
        colors[labels == uid] = color

    return colors


def save_ply(points, colors, out_path):
    points = np.asarray(points, dtype=np.float64)
    colors = np.asarray(colors, dtype=np.float64)

    assert points.shape[0] == colors.shape[0], (
        f"点数和颜色数不一致: points={points.shape}, colors={colors.shape}"
    )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out_path), pcd, write_ascii=False)


def logits_to_label(arr, n_points):
    """
    OneFormer3D语义预测提取。

    pts_semantic_mask:
      [0] = 独立语义分割
      [1] = 全景分割的语义结果

    论文语义分割可视化使用 [0]。
    """
    arr = to_numpy(arr)

    if arr is None:
        return None

    arr = np.asarray(arr)

    # OneFormer3D: 两组逐点整数语义标签
    if (
        arr.ndim == 2
        and arr.shape == (2, n_points)
        and np.issubdtype(arr.dtype, np.integer)
    ):
        return arr[0].astype(np.int64)

    # 已经是逐点标签
    if arr.ndim == 1 and arr.shape[0] == n_points:
        return arr.astype(np.int64)

    # 其他网络的类别 logits
    if arr.ndim == 2:
        if arr.shape[1] == n_points:
            return arr.argmax(axis=0).astype(np.int64)

        if arr.shape[0] == n_points:
            return arr.argmax(axis=1).astype(np.int64)

    raise ValueError(
        f"无法识别的语义输出形状: {arr.shape}"
    )


def instance_masks_to_point_labels(masks, scores, n_points):
    """
    把实例 mask [num_inst, N] 转成每点 instance id。
    如果多个实例覆盖同一点，用分数高的覆盖。
    """
    masks = to_numpy(masks)
    scores = to_numpy(scores)

    if masks is None:
        return None

    masks = np.asarray(masks)

    if masks.ndim != 2:
        return None

    # 兼容 [N, num_inst]
    if masks.shape[0] == n_points and masks.shape[1] != n_points:
        masks = masks.T

    if masks.shape[1] != n_points:
        print("[警告] instance masks 形状和点数不匹配:", masks.shape, "n_points=", n_points)
        return None

    num_inst = masks.shape[0]
    point_inst = np.full(n_points, -1, dtype=np.int64)

    if scores is None:
        order = np.arange(num_inst)
    else:
        scores = np.asarray(scores).reshape(-1)
        if len(scores) != num_inst:
            order = np.arange(num_inst)
        else:
            order = np.argsort(scores)

    for new_id, inst_idx in enumerate(order, start=1):
        m = masks[inst_idx]
        if m.dtype != np.bool_:
            m = m > 0.5
        point_inst[m] = new_id

    return point_inst


def extract_gt(data_sample, n_points):
    """提取 GT semantic / instance。"""
    gt_pts_seg = get_field(data_sample, ["gt_pts_seg"])

    sem = get_field(gt_pts_seg, [
        "pts_semantic_mask",
        "semantic_mask",
        "sem_mask",
        "pts_semantic_seg",
    ])

    ins = get_field(gt_pts_seg, [
        "pts_instance_mask",
        "instance_mask",
        "ins_mask",
    ])

    sem = to_numpy(sem)
    ins = to_numpy(ins)

    if sem is not None:
        sem = sem.reshape(-1).astype(np.int64)
        if len(sem) != n_points:
            print("[警告] GT sem 长度和点数不一致:", len(sem), n_points)
            sem = None

    if ins is not None:
        ins = ins.reshape(-1).astype(np.int64)
        if len(ins) != n_points:
            print("[警告] GT ins 长度和点数不一致:", len(ins), n_points)
            ins = None

    return sem, ins

def try_parse_instance_array(arr, scores, n_points):
    """从 ndarray 中解析每点实例 ID。"""
    if arr is None:
        return None

    arr = np.asarray(arr)

    # 情况 1：已经是 [N]，每个点一个 instance id
    if arr.ndim == 1 and arr.shape[0] == n_points:
        return arr.astype(np.int64)

    # 情况 2：是 [num_inst, N] 或 [N, num_inst] 的 mask
    if arr.ndim == 2:
        if arr.shape[1] == n_points:
            return instance_masks_to_point_labels(arr, scores, n_points)

        if arr.shape[0] == n_points:
            return instance_masks_to_point_labels(arr.T, scores, n_points)

    # 情况 3：多了 batch 维度，例如 [1, num_inst, N]
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            return try_parse_instance_array(arr[0], scores, n_points)

    return None


def extract_instance_from_any(x, scores, n_points):
    """
    递归解析 pts_instance_mask。
    兼容 Tensor / numpy / list / tuple。
    """
    if x is None:
        return None

    # 如果是 list / tuple，逐个元素尝试
    if isinstance(x, (list, tuple)):
        for item in x:
            pred = extract_instance_from_any(item, scores, n_points)
            if pred is not None:
                return pred
        return None

    arr = to_numpy(x)
    if arr is None:
        return None

    return try_parse_instance_array(arr, scores, n_points)
def extract_prediction(data_sample, n_points):
    """提取预测 semantic / instance。"""
    pred_sem = None
    pred_inst = None

    pred_pts_seg = get_field(data_sample, ["pred_pts_seg"])

    # =====================================================
    # 1. 提取每点语义预测
    # =====================================================
    sem_candidate_names = [
        "pts_semantic_mask",
        "semantic_mask",
        "sem_mask",
        "pts_semantic_seg",
        "seg_logits",
    ]

    for name in sem_candidate_names:
        sem_raw = get_field(pred_pts_seg, [name])
        if name == "pts_semantic_mask":
            arr = to_numpy(sem_raw)

            print("\n========== 两行语义标签统计 ==========")

            for row_idx in range(arr.shape[0]):
                labels, counts = np.unique(
                    arr[row_idx], return_counts=True
                )

                print(f"第 {row_idx} 行:")

                for label, count in zip(labels, counts):
                    print(
                        f"  类别 {label}: {count} 点，"
                        f"占比 {count / arr.shape[1] * 100:.2f}%"
                    )
        sem_label = logits_to_label(sem_raw, n_points)

        if sem_label is not None and len(sem_label) == n_points:
            pred_sem = sem_label.astype(np.int64)
            break

    # =====================================================
    # 2. 提取实例预测
    # OneFormer3D 这里实际输出在 pred_pts_seg 里：
    # instance_scores
    # instance_labels
    # pts_instance_mask
    # =====================================================
    scores = get_field(pred_pts_seg, [
        "instance_scores",
        "scores",
        "scores_3d",
    ])

    labels = get_field(pred_pts_seg, [
        "instance_labels",
        "labels",
        "labels_3d",
    ])

    masks = get_field(pred_pts_seg, [
        "pts_instance_mask",
        "instance_mask",
        "ins_mask",
        "pts_instance_seg",
        "panoptic_mask",
        "pts_panoptic_mask",
    ])

    scores_np = to_numpy(scores)
    labels_np = to_numpy(labels)

    pred_inst = extract_instance_from_any(masks, scores_np, n_points)

    # =====================================================
    # 3. 如果 pred_pts_seg 里没解析成功，再尝试 pred_instances_3d
    # =====================================================
    if pred_inst is None:
        pred_instances_3d = get_field(data_sample, ["pred_instances_3d"])

        masks = get_field(pred_instances_3d, [
            "masks",
            "masks_3d",
            "pts_instance_mask",
            "instance_masks",
        ])

        scores = get_field(pred_instances_3d, [
            "scores_3d",
            "scores",
        ])

        scores_np = to_numpy(scores)
        pred_inst = extract_instance_from_any(masks, scores_np, n_points)

    # =====================================================
    # 4. 打印调试信息，方便确认实例结构
    # =====================================================
    print("instance_scores type:", type(scores))
    if scores_np is not None:
        print("instance_scores shape:", scores_np.shape)

    print("instance_labels type:", type(labels))
    if labels_np is not None:
        print("instance_labels shape:", labels_np.shape)

    print("pts_instance_mask type:", type(masks))
    masks_np = to_numpy(masks)
    if masks_np is not None:
        print("pts_instance_mask shape:", masks_np.shape)
    else:
        if isinstance(masks, (list, tuple)):
            print("pts_instance_mask 是 list/tuple，长度:", len(masks))
            for idx, item in enumerate(masks):
                item_np = to_numpy(item)
                print(
                    f"  item {idx}: type={type(item)}, "
                    f"shape={None if item_np is None else item_np.shape}"
                )

    return pred_sem, pred_inst

# =========================================================
# 主函数
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description="OneFormer3D corn inference and save PLY"
    )

    parser.add_argument(
        "--config",
        default="configs/oneformer3d_corn.py",
        help="配置文件路径"
    )
    parser.add_argument(
        "--checkpoint",
        default=r"work_dirs/oneformer3d_corn_win/best_all_ap_50%_epoch_800.pth",
        help="训练好的 pth。注意：当前你的环境不要用 fixed.pth"
    )
    parser.add_argument(
        "--out-dir",
        default=r"work_dirs/corn_pred_ply",
        help="输出目录"
    )
    parser.add_argument(
        "--sample",
        default=None,
        help="只导出指定 sample_id。不填则导出 test_dataloader 里的全部样本"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="最多导出多少个样本，-1 表示全部"
    )
    parser.add_argument(
        "--save-gt",
        action="store_true",
        help="同时保存 GT 语义和实例 PLY"
    )

    args = parser.parse_args()

    from mmengine.config import Config
    from mmengine.runner import Runner

    cfg = Config.fromfile(args.config)

    cfg.load_from = args.checkpoint
    cfg.work_dir = args.out_dir

    # 强制单样本，便于保存
    cfg.test_dataloader.batch_size = 1
    cfg.test_dataloader.num_workers = 0
    cfg.test_dataloader.persistent_workers = False

    # 不使用 OpenMMLab 自带可视化 hook，避免 npy 被当成 bin 读取
    if "default_hooks" in cfg and "visualization" in cfg.default_hooks:
        cfg.default_hooks.visualization.draw = False

    print("配置文件:", args.config)
    print("权重文件:", args.checkpoint)
    print("输出目录:", args.out_dir)
    print("注意：不要使用 fixed checkpoint。")

    runner = Runner.from_cfg(cfg)

    # 加载 cfg.load_from 指定的 checkpoint
    runner.load_or_resume()

    model = runner.model
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0

    with torch.no_grad():
        for data_idx, data_batch in enumerate(runner.test_dataloader):
            outputs = model.test_step(data_batch)

            for i, data_sample in enumerate(outputs):
                sample_name = get_sample_name(data_sample, data_idx)

                if args.sample is not None and sample_name != args.sample:
                    continue

                points = get_points_from_batch(data_batch, i)
                if points is None:
                    print("[跳过] 无pred_sem, pred_inst = extract_prediction(data_sample, n_points)法从 batch 中读取 points:", sample_name)
                    continue

                n_points = points.shape[0]

                print("\n==============================")
                print("样本:", sample_name)
                print("点数:", n_points)

                pred_sem, pred_inst = extract_prediction(data_sample, n_points)
                gt_sem, gt_inst = extract_gt(data_sample, n_points)
                # _, pred_inst = extract_prediction(data_sample, n_points)
                # _, gt_inst = extract_gt(data_sample, n_points)
                print("\n========== 语义预测字段检查 ==========")

                pred_pts_seg = get_field(data_sample, ["pred_pts_seg"])

                print("pred_pts_seg 所有字段:")
                print(field_names(pred_pts_seg))

                for name in [
                    "pts_semantic_mask",
                    "semantic_mask",
                    "sem_mask",
                    "pts_semantic_seg",
                    "seg_logits",
                ]:
                    raw = get_field(pred_pts_seg, [name])

                    if raw is None:
                        print(f"{name}: 不存在")
                        continue

                    arr = to_numpy(raw)

                    if arr is None:
                        print(f"{name}: 无法转换")
                        continue

                    print(f"\n字段: {name}")
                    print("shape:", arr.shape)
                    print("dtype:", arr.dtype)

                    if arr.ndim == 1:
                        labels, counts = np.unique(
                            arr, return_counts=True
                        )
                        print("标签分布:", dict(zip(labels.tolist(), counts.tolist())))

                    elif arr.ndim == 2:
                        print("最小值:", np.min(arr))
                        print("最大值:", np.max(arr))

                print("pred_pts_seg 字段:", field_names(get_field(data_sample, ["pred_pts_seg"])))
                print("pred_instances_3d 字段:", field_names(get_field(data_sample, ["pred_instances_3d"])))

                name = safe_name(sample_name)

                # ---------- 保存预测语义 ----------
                if pred_sem is not None:
                    print("预测语义类别:", np.unique(pred_sem))
                    np.save(out_dir / f"{name}_pred_sem.npy", pred_sem)
                    save_ply(
                        points,
                        color_by_semantic(pred_sem),
                        out_dir / f"{name}_pred_sem.ply"
                    )
                    print("已保存:", out_dir / f"{name}_pred_sem.ply")
                else:
                    print("[警告] 没有提取到 pred_sem，未保存语义预测 PLY。")

                # ---------- 保存预测实例 ----------
                if pred_inst is not None:
                    print("预测实例 ID 数:", len(np.unique(pred_inst)))
                    np.save(out_dir / f"{name}_pred_inst.npy", pred_inst)
                    save_ply(
                        points,
                        color_by_instance(pred_inst),
                        out_dir / f"{name}_pred_inst.ply"
                    )
                    print("已保存:", out_dir / f"{name}_pred_inst.ply")
                else:
                    print("[警告] 没有提取到 pred_inst，未保存实例预测 PLY。")

                # ---------- 保存 GT ----------
                if args.save_gt:
                    if gt_sem is not None:
                        np.save(out_dir / f"{name}_gt_sem.npy", gt_sem)
                        save_ply(
                            points,
                            color_by_semantic(gt_sem),
                            out_dir / f"{name}_gt_sem.ply"
                        )
                        print("已保存:", out_dir / f"{name}_gt_sem.ply")

                    if gt_inst is not None:
                        np.save(out_dir / f"{name}_gt_inst.npy", gt_inst)
                        save_ply(
                            points,
                            color_by_instance(gt_inst),
                            out_dir / f"{name}_gt_inst.ply"
                        )
                        print("已保存:", out_dir / f"{name}_gt_inst.ply")

                saved_count += 1

                if args.max_samples > 0 and saved_count >= args.max_samples:
                    print("\n达到 max-samples，结束。")
                    return

    print("\n完成。保存样本数:", saved_count)
    print("输出目录:", out_dir.resolve())


if __name__ == "__main__":
    main()