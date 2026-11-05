from pathlib import Path
import pickle
import random
import shutil
import numpy as np


# =========================================================
# 1. 路径
# =========================================================
TXT_DIR = Path(r"D:\onef\2time\标注")

# 新输出目录，不覆盖旧的 XYZ 数据
OUT_ROOT = Path(r"D:\onef\2time\oneformer_rgb")

TXT_PATTERN = "*.txt"

# 原 txt 格式：
# x y z r g b label nx ny nz
XYZ_COLS = [0, 1, 2]
RGB_COLS = [3, 4, 5]
LABEL_COL = 6

STEM_LABEL = 0
LEAF_START_LABEL = 11

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 20260707

CLEAR_OLD_OUTPUT = True


def safe_sample_id(path: Path):
    name = path.stem
    keep = []
    for c in name:
        if c.isalnum() or c in ["-", "_", ".", "(", ")"]:
            keep.append(c)
        else:
            keep.append("_")
    return "".join(keep)


def read_txt(txt_path: Path):
    try:
        arr = np.loadtxt(str(txt_path), dtype=np.float64)
    except Exception:
        arr = np.loadtxt(str(txt_path), dtype=np.float64, delimiter=",")

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    return arr


def normalize_rgb(rgb):
    rgb = rgb.astype(np.float32)

    # 兼容 0~255 或 0~1
    if np.nanmax(rgb) > 1.5:
        rgb = rgb / 255.0

    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb


def check_raw_labels(raw_label, sample_id):
    unique_labels = sorted(np.unique(raw_label).astype(int).tolist())
    errors = []

    if STEM_LABEL not in unique_labels:
        errors.append(f"缺少 stem 标签 {STEM_LABEL}")

    if 1 in unique_labels:
        errors.append("发现 label=1。当前正式规则里 stem 应该是 0，不应该再出现 1")

    invalid_middle = [x for x in unique_labels if 1 <= x < LEAF_START_LABEL]
    if invalid_middle:
        errors.append(f"发现非法中间标签 {invalid_middle}")

    leaf_labels = sorted([x for x in unique_labels if x >= LEAF_START_LABEL])

    if len(leaf_labels) == 0:
        errors.append("没有叶片标签，至少应该有 11")

    if len(leaf_labels) > 0:
        if leaf_labels[0] != LEAF_START_LABEL:
            errors.append(
                f"叶片标签必须从 {LEAF_START_LABEL} 开始，但当前最小叶片标签是 {leaf_labels[0]}"
            )

        expected_leaf = list(range(LEAF_START_LABEL, max(leaf_labels) + 1))
        missing = [x for x in expected_leaf if x not in leaf_labels]

        if missing:
            errors.append(
                f"叶片标签不连续。当前 leaf={leaf_labels}，缺失 {missing}"
            )

    if errors:
        msg = "\n".join([f"  - {e}" for e in errors])
        raise ValueError(f"{sample_id} 标签检查失败:\n{msg}")

    return unique_labels, leaf_labels


def check_converted_arrays(sample_id, pts, sem, ins):
    if pts.ndim != 2 or pts.shape[1] != 6:
        raise ValueError(f"{sample_id} pts 应为 N×6，当前 {pts.shape}")

    n = pts.shape[0]

    if sem.shape[0] != n:
        raise ValueError(f"{sample_id} sem 点数不一致: pts={n}, sem={sem.shape[0]}")

    if ins.shape[0] != n:
        raise ValueError(f"{sample_id} ins 点数不一致: pts={n}, ins={ins.shape[0]}")

    sem_unique = sorted(np.unique(sem).astype(int).tolist())
    ins_unique = sorted(np.unique(ins).astype(int).tolist())

    if sem_unique != [0, 1]:
        raise ValueError(f"{sample_id} sem 应为 [0,1]，当前 {sem_unique}")

    if 0 not in ins_unique:
        raise ValueError(f"{sample_id} ins 缺少 0")

    expected_ins = list(range(0, max(ins_unique) + 1))
    missing_ins = [x for x in expected_ins if x not in ins_unique]
    if missing_ins:
        raise ValueError(f"{sample_id} ins 不连续。当前 {ins_unique}，缺失 {missing_ins}")

    stem_ins = sorted(np.unique(ins[sem == 0]).astype(int).tolist())
    leaf_ins = sorted(np.unique(ins[sem == 1]).astype(int).tolist())

    if stem_ins != [0]:
        raise ValueError(f"{sample_id} stem 对应 ins 应为 [0]，当前 {stem_ins}")

    if len(leaf_ins) == 0 or leaf_ins[0] != 1:
        raise ValueError(f"{sample_id} leaf ins 应从 1 开始，当前 {leaf_ins}")

    for iid in ins_unique:
        sems = sorted(np.unique(sem[ins == iid]).astype(int).tolist())
        if iid == 0 and sems != [0]:
            raise ValueError(f"{sample_id} ins=0 应该是 stem，当前 sem={sems}")
        if iid >= 1 and sems != [1]:
            raise ValueError(f"{sample_id} ins={iid} 应该是 leaf，当前 sem={sems}")


def convert_one_txt(txt_path: Path, used_sample_ids):
    sample_id = safe_sample_id(txt_path)

    base_id = sample_id
    k = 1
    while sample_id in used_sample_ids:
        sample_id = f"{base_id}_{k}"
        k += 1
    used_sample_ids.add(sample_id)

    arr = read_txt(txt_path)

    if arr.shape[1] <= LABEL_COL:
        raise ValueError(f"{txt_path.name} 列数不足，当前 {arr.shape[1]} 列")

    finite_mask = np.isfinite(arr).all(axis=1)
    if not finite_mask.all():
        bad_num = int((~finite_mask).sum())
        print(f"[警告] {txt_path.name} 存在 {bad_num} 行 NaN/Inf，已删除。")
        arr = arr[finite_mask]

    if arr.shape[0] == 0:
        raise ValueError(f"{txt_path.name} 有效点数为 0")

    xyz = arr[:, XYZ_COLS].astype(np.float32)
    rgb = normalize_rgb(arr[:, RGB_COLS])

    pts = np.concatenate([xyz, rgb], axis=1).astype(np.float32)

    raw_label_float = arr[:, LABEL_COL]
    rounded = np.rint(raw_label_float)

    not_integer_mask = ~np.isclose(raw_label_float, rounded, atol=1e-6)
    if np.any(not_integer_mask):
        bad_values = np.unique(raw_label_float[not_integer_mask])
        raise ValueError(
            f"{txt_path.name} 存在非整数标签: {bad_values[:20].tolist()}"
        )

    raw_label = rounded.astype(np.int64)

    unique_labels, leaf_labels = check_raw_labels(raw_label, sample_id)

    sem = np.zeros(raw_label.shape[0], dtype=np.int64)
    sem[raw_label == STEM_LABEL] = 0
    sem[raw_label >= LEAF_START_LABEL] = 1

    ins = np.zeros(raw_label.shape[0], dtype=np.int64)
    ins[raw_label == STEM_LABEL] = 0

    for leaf_lab in leaf_labels:
        ins_id = leaf_lab - LEAF_START_LABEL + 1
        ins[raw_label == leaf_lab] = ins_id

    check_converted_arrays(sample_id, pts, sem, ins)

    np.save(OUT_ROOT / f"{sample_id}_pts.npy", pts)
    np.save(OUT_ROOT / f"{sample_id}_sem.npy", sem)
    np.save(OUT_ROOT / f"{sample_id}_ins.npy", ins)

    info = {
        "sample_id": sample_id,
        "txt_file": str(txt_path),
        "num_points": int(pts.shape[0]),
        "raw_labels": unique_labels,
        "leaf_labels": leaf_labels,
        "sem_labels": sorted(np.unique(sem).astype(int).tolist()),
        "ins_labels": sorted(np.unique(ins).astype(int).tolist()),
        "stem_points": int((sem == 0).sum()),
        "leaf_points": int((sem == 1).sum()),
        "rgb_min": rgb.min(axis=0).tolist(),
        "rgb_max": rgb.max(axis=0).tolist(),
    }

    return info


def split_dataset(sample_ids):
    ids = list(sample_ids)
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(round(n * TRAIN_RATIO))
    n_val = int(round(n * VAL_RATIO))

    if n >= 10:
        n_val = max(n_val, 1)
        n_test = n - n_train - n_val
        if n_test <= 0:
            n_test = 1
            n_train = n - n_val - n_test
    else:
        n_train = max(1, n - 2)
        n_val = 1
        n_test = n - n_train - n_val

    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]

    return train_ids, val_ids, test_ids


def make_data_info(sample_id):
    pts_path = OUT_ROOT / f"{sample_id}_pts.npy"
    sem_path = OUT_ROOT / f"{sample_id}_sem.npy"
    ins_path = OUT_ROOT / f"{sample_id}_ins.npy"

    pts = np.load(pts_path)

    return {
        "sample_id": sample_id,
        "pts_path": str(pts_path),
        "pts_semantic_mask_path": str(sem_path),
        "pts_instance_mask_path": str(ins_path),
        "num_points": int(pts.shape[0]),
        "lidar_points": {
            "lidar_path": str(pts_path),
            "num_pts_feats": 6,
        },
        "instances": [],
    }


def make_infos(sample_ids):
    return {
        "data_list": [make_data_info(sid) for sid in sample_ids],
        "metainfo": {
            "classes": ("stem", "leaf"),
            "dataset_name": "S3DIS",
        },
    }


def save_pkl(sample_ids, out_name):
    infos = make_infos(sample_ids)
    out_path = OUT_ROOT / out_name

    with open(out_path, "wb") as f:
        pickle.dump(infos, f)

    print(f"已保存 {out_name}: {len(sample_ids)} 个样本")
    return out_path


def compute_train_class_weight(train_ids):
    stem_count = 0
    leaf_count = 0

    for sid in train_ids:
        sem = np.load(OUT_ROOT / f"{sid}_sem.npy").reshape(-1).astype(np.int64)
        stem_count += int((sem == 0).sum())
        leaf_count += int((sem == 1).sum())

    total = stem_count + leaf_count

    stem_weight = total / (2.0 * stem_count)
    leaf_weight = total / (2.0 * leaf_count)

    return stem_count, leaf_count, stem_weight, leaf_weight


def save_report(all_infos, train_ids, val_ids, test_ids):
    report_path = OUT_ROOT / "prepare_report.txt"

    stem_count, leaf_count, stem_weight, leaf_weight = compute_train_class_weight(train_ids)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"TXT_DIR: {TXT_DIR}\n")
        f.write(f"OUT_ROOT: {OUT_ROOT}\n")
        f.write(f"样本总数: {len(all_infos)}\n\n")

        f.write(f"train: {len(train_ids)}\n")
        for sid in train_ids:
            f.write(f"  {sid}\n")

        f.write(f"\nval: {len(val_ids)}\n")
        for sid in val_ids:
            f.write(f"  {sid}\n")

        f.write(f"\ntest: {len(test_ids)}\n")
        for sid in test_ids:
            f.write(f"  {sid}\n")

        f.write("\ntrain semantic points:\n")
        f.write(f"stem: {stem_count}\n")
        f.write(f"leaf: {leaf_count}\n")
        f.write(f"class_weight=[{stem_weight:.6f}, {leaf_weight:.6f}]\n")

        f.write("\n\n样本详情:\n")
        for info in all_infos:
            f.write("-" * 80 + "\n")
            for k, v in info.items():
                f.write(f"{k}: {v}\n")

    for name, ids in [
        ("train_ids.txt", train_ids),
        ("val_ids.txt", val_ids),
        ("test_ids.txt", test_ids),
    ]:
        with open(OUT_ROOT / name, "w", encoding="utf-8") as f:
            for sid in ids:
                f.write(sid + "\n")

    print("报告已保存:", report_path)
    print("train stem points:", stem_count)
    print("train leaf  points:", leaf_count)
    print(f"建议 class_weight=[{stem_weight:.6f}, {leaf_weight:.6f}]")


def main():
    print("TXT_DIR:", TXT_DIR)
    print("OUT_ROOT:", OUT_ROOT)

    txt_files = sorted(TXT_DIR.glob(TXT_PATTERN))

    if len(txt_files) == 0:
        raise FileNotFoundError(f"没有找到 txt 文件: {TXT_DIR}\\{TXT_PATTERN}")

    print("检测到 txt 数量:", len(txt_files))

    if OUT_ROOT.exists() and CLEAR_OLD_OUTPUT:
        print("清空旧输出目录:", OUT_ROOT)
        shutil.rmtree(OUT_ROOT)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    used_sample_ids = set()
    all_infos = []
    failed = []

    print("\n========== 1. 检查并转换 txt 为 RGB 版 ==========")

    for idx, txt_path in enumerate(txt_files, start=1):
        print(f"\n[{idx}/{len(txt_files)}] {txt_path.name}")

        try:
            info = convert_one_txt(txt_path, used_sample_ids)
            all_infos.append(info)

            print("sample_id:", info["sample_id"])
            print("点数:", info["num_points"])
            print("raw labels:", info["raw_labels"])
            print("sem labels:", info["sem_labels"])
            print("ins labels:", info["ins_labels"])
            print("stem 点数:", info["stem_points"])
            print("leaf 点数:", info["leaf_points"])
            print("rgb min:", info["rgb_min"])
            print("rgb max:", info["rgb_max"])

        except Exception as e:
            print("[失败]", txt_path.name)
            print("错误:", repr(e))
            failed.append((txt_path, repr(e)))

    if failed:
        fail_report = OUT_ROOT / "failed_files.txt"
        with open(fail_report, "w", encoding="utf-8") as f:
            for p, e in failed:
                f.write(f"{p}\n{e}\n\n")

        print("\n有文件转换失败。失败数量:", len(failed))
        print("失败列表已保存:", fail_report)
        raise RuntimeError("存在转换失败文件，停止生成 pkl。")

    sample_ids = [x["sample_id"] for x in all_infos]

    print("\n========== 2. 划分 train / val / test ==========")
    train_ids, val_ids, test_ids = split_dataset(sample_ids)

    print("train:", len(train_ids))
    print("val:", len(val_ids))
    print("test:", len(test_ids))

    print("\n========== 3. 生成 pkl ==========")
    save_pkl(train_ids, "corn_infos_train.pkl")
    save_pkl(val_ids, "corn_infos_val.pkl")
    save_pkl(test_ids, "corn_infos_test.pkl")

    print("\n========== 4. 保存报告 ==========")
    save_report(all_infos, train_ids, val_ids, test_ids)

    print("\n全部完成。输出目录:")
    print(OUT_ROOT)


if __name__ == "__main__":
    main()