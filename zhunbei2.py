from pathlib import Path
import pickle
import random
import shutil
import numpy as np


# =========================================================
# 1. 输入 / 输出路径
# =========================================================
TXT_DIR = Path(r"D:\onef\2time\标注")

OUT_ROOT = Path(r"D:\onef\2time\oneformer")

# txt 文件匹配规则
TXT_PATTERN = "*.txt"

# =========================================================
# 2. txt 列定义
# 当前格式：
# x y z r g b label nx ny nz
# label 是第 7 列，Python 下标为 6
# =========================================================
LABEL_COL = 6

# =========================================================
# 3. 当前正式标注规则
# =========================================================
STEM_LABEL = 0
LEAF_START_LABEL = 11

# =========================================================
# 4. 数据集划分
# 100 多个点云建议先按 8:1:1
# =========================================================
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

RANDOM_SEED = 20260707

# 是否清空旧输出
CLEAR_OLD_OUTPUT = True

# 是否中心化点云。先不要中心化，保持原始坐标。
CENTER_POINTS = False


def safe_sample_id(path: Path):
    """把文件名转成安全 sample_id。"""
    name = path.stem
    keep = []
    for c in name:
        if c.isalnum() or c in ["-", "_", ".", "(", ")"]:
            keep.append(c)
        else:
            keep.append("_")
    return "".join(keep)


def read_txt(txt_path: Path):
    """读取 txt，兼容空格、Tab、逗号分隔。"""
    try:
        arr = np.loadtxt(str(txt_path), dtype=np.float64)
    except Exception:
        arr = np.loadtxt(str(txt_path), dtype=np.float64, delimiter=",")

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    return arr


def check_raw_labels(raw_label, sample_id):
    """
    检查原始标签：
        必须有 0
        叶片必须从 11 开始连续
        不允许出现 1
        不允许出现 2~10
        不允许跳号，例如 11,13 缺 12
    """
    unique_labels = sorted(np.unique(raw_label).astype(int).tolist())

    errors = []

    if STEM_LABEL not in unique_labels:
        errors.append(f"缺少 stem 标签 {STEM_LABEL}")

    if 1 in unique_labels:
        errors.append("发现 label=1。当前正式规则里茎应该是 0，不应该再出现 1")

    invalid_middle = [x for x in unique_labels if 1 <= x < LEAF_START_LABEL]
    if invalid_middle:
        errors.append(f"发现非法中间标签 {invalid_middle}。合法标签只能是 0 和 11,12,13...")

    leaf_labels = sorted([x for x in unique_labels if x >= LEAF_START_LABEL])

    if len(leaf_labels) == 0:
        errors.append("没有叶片标签。至少应该有 11")

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

    valid_labels = {STEM_LABEL}
    if len(leaf_labels) > 0:
        valid_labels.update(range(LEAF_START_LABEL, max(leaf_labels) + 1))

    invalid = sorted([x for x in unique_labels if x not in valid_labels])
    if invalid:
        errors.append(f"存在非法标签 {invalid}")

    if errors:
        msg = "\n".join([f"  - {e}" for e in errors])
        raise ValueError(f"{sample_id} 标签检查失败:\n{msg}")

    return unique_labels, leaf_labels


def convert_one_txt(txt_path: Path, used_sample_ids):
    sample_id = safe_sample_id(txt_path)

    # 防止文件名清洗后重复
    base_id = sample_id
    k = 1
    while sample_id in used_sample_ids:
        sample_id = f"{base_id}_{k}"
        k += 1

    used_sample_ids.add(sample_id)

    arr = read_txt(txt_path)

    if arr.shape[1] <= LABEL_COL:
        raise ValueError(
            f"{txt_path.name} 列数不足。当前 {arr.shape[1]} 列，至少需要 7 列。"
        )

    finite_mask = np.isfinite(arr).all(axis=1)
    if not finite_mask.all():
        bad_num = int((~finite_mask).sum())
        print(f"[警告] {txt_path.name} 存在 {bad_num} 行 NaN/Inf，已删除。")
        arr = arr[finite_mask]

    if arr.shape[0] == 0:
        raise ValueError(f"{txt_path.name} 有效点数为 0")

    pts = arr[:, 0:3].astype(np.float32)

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

    if CENTER_POINTS:
        pts = pts - pts.mean(axis=0, keepdims=True)

    # =====================================================
    # 生成 sem
    # sem:
    #   0 -> stem
    #   11/12/13/... -> leaf
    # =====================================================
    sem = np.zeros(raw_label.shape[0], dtype=np.int64)
    sem[raw_label == STEM_LABEL] = 0
    sem[raw_label >= LEAF_START_LABEL] = 1

    # =====================================================
    # 生成 ins
    # ins:
    #   0  -> 0
    #   11 -> 1
    #   12 -> 2
    #   13 -> 3
    # =====================================================
    ins = np.zeros(raw_label.shape[0], dtype=np.int64)
    ins[raw_label == STEM_LABEL] = 0

    for leaf_lab in leaf_labels:
        ins_id = leaf_lab - LEAF_START_LABEL + 1
        ins[raw_label == leaf_lab] = ins_id

    # 再检查一次转换后的 sem / ins
    check_converted_arrays(sample_id, pts, sem, ins)

    pts_path = OUT_ROOT / f"{sample_id}_pts.npy"
    sem_path = OUT_ROOT / f"{sample_id}_sem.npy"
    ins_path = OUT_ROOT / f"{sample_id}_ins.npy"

    np.save(pts_path, pts)
    np.save(sem_path, sem)
    np.save(ins_path, ins)

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
    }

    return info


def check_converted_arrays(sample_id, pts, sem, ins):
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"{sample_id} pts 应为 N×3，当前 {pts.shape}")

    n = pts.shape[0]

    if sem.shape[0] != n:
        raise ValueError(f"{sample_id} sem 点数不一致: pts={n}, sem={sem.shape[0]}")

    if ins.shape[0] != n:
        raise ValueError(f"{sample_id} ins 点数不一致: pts={n}, ins={ins.shape[0]}")

    sem_unique = sorted(np.unique(sem).astype(int).tolist())
    ins_unique = sorted(np.unique(ins).astype(int).tolist())

    if sem_unique != [0, 1]:
        raise ValueError(f"{sample_id} sem 应该只有 [0,1]，当前 {sem_unique}")

    if 0 not in ins_unique:
        raise ValueError(f"{sample_id} ins 缺少 0，也就是缺少 stem 实例")

    expected_ins = list(range(0, max(ins_unique) + 1))
    missing_ins = [x for x in expected_ins if x not in ins_unique]
    if missing_ins:
        raise ValueError(
            f"{sample_id} ins 不连续。当前 {ins_unique}，缺失 {missing_ins}"
        )

    stem_ins = sorted(np.unique(ins[sem == 0]).astype(int).tolist())
    leaf_ins = sorted(np.unique(ins[sem == 1]).astype(int).tolist())

    if stem_ins != [0]:
        raise ValueError(
            f"{sample_id} stem 对应 ins 应该只有 [0]，当前 {stem_ins}"
        )

    if len(leaf_ins) == 0:
        raise ValueError(f"{sample_id} leaf 没有实例")

    expected_leaf_ins = list(range(1, max(leaf_ins) + 1))
    missing_leaf = [x for x in expected_leaf_ins if x not in leaf_ins]

    if leaf_ins[0] != 1 or missing_leaf:
        raise ValueError(
            f"{sample_id} leaf ins 应从 1 连续开始。当前 {leaf_ins}，缺失 {missing_leaf}"
        )

    # 一个实例内部不能同时有 stem 和 leaf
    for iid in ins_unique:
        sems = sorted(np.unique(sem[ins == iid]).astype(int).tolist())
        if len(sems) != 1:
            raise ValueError(
                f"{sample_id} 实例 {iid} 内部混合多个语义标签: {sems}"
            )

        if iid == 0 and sems != [0]:
            raise ValueError(
                f"{sample_id} ins=0 应该是 stem sem=[0]，当前 {sems}"
            )

        if iid >= 1 and sems != [1]:
            raise ValueError(
                f"{sample_id} ins={iid} 应该是 leaf sem=[1]，当前 {sems}"
            )


def split_dataset(sample_ids):
    ids = list(sample_ids)
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(round(n * TRAIN_RATIO))
    n_val = int(round(n * VAL_RATIO))

    # 防止 val/test 为空
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

    data_info = {
        "sample_id": sample_id,

        # 完全对齐你同学的 pkl：绝对路径
        "pts_path": str(pts_path),
        "pts_semantic_mask_path": str(sem_path),
        "pts_instance_mask_path": str(ins_path),

        "num_points": int(pts.shape[0]),

        "lidar_points": {
            "lidar_path": str(pts_path),
            "num_pts_feats": 3,
        },

        # 这个保持空列表即可。你同学数据也是空列表。
        "instances": [],
    }

    return data_info


def make_infos(sample_ids):
    data_list = [make_data_info(sid) for sid in sample_ids]

    infos = {
        "data_list": data_list,
        "metainfo": {
            "classes": ("stem", "leaf"),
            "dataset_name": "S3DIS",
        },
    }

    return infos


def save_pkl(sample_ids, out_name):
    infos = make_infos(sample_ids)
    out_path = OUT_ROOT / out_name

    with open(out_path, "wb") as f:
        pickle.dump(infos, f)

    print(f"已保存 {out_name}: {len(sample_ids)} 个样本")
    return out_path


def save_report(all_infos, train_ids, val_ids, test_ids):
    report_path = OUT_ROOT / "prepare_report.txt"

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

        f.write("\n\n样本详情:\n")
        for info in all_infos:
            f.write("-" * 80 + "\n")
            for k, v in info.items():
                f.write(f"{k}: {v}\n")

    # 单独保存 split，后面复现实验方便
    for name, ids in [
        ("train_ids.txt", train_ids),
        ("val_ids.txt", val_ids),
        ("test_ids.txt", test_ids),
    ]:
        with open(OUT_ROOT / name, "w", encoding="utf-8") as f:
            for sid in ids:
                f.write(sid + "\n")

    print("报告已保存:", report_path)


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

    print("\n========== 1. 检查并转换 txt ==========")

    failed = []

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
        print("请先修正这些 txt 标注，再重新运行。")
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