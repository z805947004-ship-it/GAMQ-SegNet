from pathlib import Path
import pickle
import numpy as np


DATA_ROOT = Path(r"D:\onef\2time\oneformer")

PKL_FILES = {
    "train": DATA_ROOT / "corn_infos_train.pkl",
    "val": DATA_ROOT / "corn_infos_val.pkl",
    "test": DATA_ROOT / "corn_infos_test.pkl",
}


def check_one_split(split_name, pkl_path):
    print("\n" + "=" * 90)
    print(split_name, pkl_path)
    print("=" * 90)

    if not pkl_path.exists():
        raise FileNotFoundError(pkl_path)

    with open(pkl_path, "rb") as f:
        infos = pickle.load(f)

    data_list = infos["data_list"]
    print("样本数:", len(data_list))
    print("metainfo:", infos.get("metainfo", None))

    num_points_list = []
    total_stem = 0
    total_leaf = 0

    for idx, item in enumerate(data_list):
        sample_id = item["sample_id"]

        pts_path = Path(item["pts_path"])
        sem_path = Path(item["pts_semantic_mask_path"])
        ins_path = Path(item["pts_instance_mask_path"])

        if not pts_path.exists():
            raise FileNotFoundError(f"{sample_id} 缺少 pts: {pts_path}")
        if not sem_path.exists():
            raise FileNotFoundError(f"{sample_id} 缺少 sem: {sem_path}")
        if not ins_path.exists():
            raise FileNotFoundError(f"{sample_id} 缺少 ins: {ins_path}")

        pts = np.load(pts_path)
        sem = np.load(sem_path).reshape(-1).astype(np.int64)
        ins = np.load(ins_path).reshape(-1).astype(np.int64)

        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"{sample_id} pts 形状错误: {pts.shape}")

        if len(sem) != pts.shape[0]:
            raise ValueError(f"{sample_id} sem 点数不一致: pts={pts.shape[0]}, sem={len(sem)}")

        if len(ins) != pts.shape[0]:
            raise ValueError(f"{sample_id} ins 点数不一致: pts={pts.shape[0]}, ins={len(ins)}")

        sem_unique = sorted(np.unique(sem).tolist())
        ins_unique = sorted(np.unique(ins).tolist())

        if sem_unique != [0, 1]:
            raise ValueError(f"{sample_id} sem 应为 [0,1]，当前 {sem_unique}")

        if 0 not in ins_unique:
            raise ValueError(f"{sample_id} ins 缺少 0")

        expected_ins = list(range(0, max(ins_unique) + 1))
        missing = [x for x in expected_ins if x not in ins_unique]
        if missing:
            raise ValueError(f"{sample_id} ins 不连续: {ins_unique}, 缺失 {missing}")

        stem_ins = sorted(np.unique(ins[sem == 0]).tolist())
        leaf_ins = sorted(np.unique(ins[sem == 1]).tolist())

        if stem_ins != [0]:
            raise ValueError(f"{sample_id} stem 对应 ins 应为 [0]，当前 {stem_ins}")

        if len(leaf_ins) == 0 or leaf_ins[0] != 1:
            raise ValueError(f"{sample_id} leaf ins 应从 1 开始，当前 {leaf_ins}")

        for iid in ins_unique:
            sems = sorted(np.unique(sem[ins == iid]).tolist())
            if iid == 0 and sems != [0]:
                raise ValueError(f"{sample_id} ins=0 应该是 stem，当前 sem={sems}")
            if iid >= 1 and sems != [1]:
                raise ValueError(f"{sample_id} ins={iid} 应该是 leaf，当前 sem={sems}")

        n = pts.shape[0]
        num_points_list.append(n)

        total_stem += int((sem == 0).sum())
        total_leaf += int((sem == 1).sum())

        if idx < 3:
            print(f"[示例] {sample_id}: points={n}, sem={sem_unique}, ins={ins_unique}")

    num_points_arr = np.array(num_points_list)

    print("\n点数统计:")
    print("min:", int(num_points_arr.min()))
    print("max:", int(num_points_arr.max()))
    print("mean:", float(num_points_arr.mean()))
    print("median:", float(np.median(num_points_arr)))

    print("\n语义点数:")
    print("stem:", total_stem)
    print("leaf :", total_leaf)

    total = total_stem + total_leaf
    stem_ratio = total_stem / total
    leaf_ratio = total_leaf / total

    print("stem ratio:", stem_ratio)
    print("leaf  ratio:", leaf_ratio)

    # balanced CE weight: total / (num_classes * class_count)
    stem_weight = total / (2 * total_stem)
    leaf_weight = total / (2 * total_leaf)

    print("\n建议 class_weight:")
    print(f"class_weight=[{stem_weight:.4f}, {leaf_weight:.4f}]")

    return {
        "num_samples": len(data_list),
        "min_points": int(num_points_arr.min()),
        "max_points": int(num_points_arr.max()),
        "stem_weight": stem_weight,
        "leaf_weight": leaf_weight,
    }


def main():
    results = {}

    for split_name, pkl_path in PKL_FILES.items():
        results[split_name] = check_one_split(split_name, pkl_path)

    print("\n" + "=" * 90)
    print("总检查结果")
    print("=" * 90)
    print(results)
    print("\nPASS：数据集结构、路径、sem/ins 映射均正确。")


if __name__ == "__main__":
    main()