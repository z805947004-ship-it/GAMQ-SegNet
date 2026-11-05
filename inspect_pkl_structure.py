from pathlib import Path
import pickle
import pprint
import numpy as np


# =========================================================
# 这里改成你同学那个可以正常训练的数据集 pkl
# 可以一次检查 train / val / test
# =========================================================
PKL_FILES = [
    Path(r"D:\pythoncode\oneformer3d-main\data\corn_dataset\corn_infos_train.pkl"),
    Path(r"D:\pythoncode\oneformer3d-main\data\corn_dataset\corn_infos_val.pkl"),
    # Path(r"D:\pythoncode\oneformer3d-main\data\corn_dataset\corn_infos_test.pkl"),
]

# 报告输出位置
REPORT_PATH = Path(r"D:\pythoncode\oneformer3d-main\work_dirs\inspect_pkl_structure_report.txt")

# 每个 pkl 打印前几个样本
NUM_SHOW_ITEMS = 5

# 递归打印最大深度
MAX_DEPTH = 6


def short_value(v, max_len=160):
    """把复杂对象压缩成可读字符串。"""
    try:
        if isinstance(v, np.ndarray):
            return f"np.ndarray(shape={v.shape}, dtype={v.dtype}, min={np.nanmin(v) if v.size > 0 else 'empty'}, max={np.nanmax(v) if v.size > 0 else 'empty'})"

        if isinstance(v, (list, tuple)):
            return f"{type(v).__name__}(len={len(v)})"

        if isinstance(v, dict):
            return f"dict(keys={list(v.keys())})"

        s = repr(v)
        if len(s) > max_len:
            s = s[:max_len] + " ..."

        return s
    except Exception as e:
        return f"<无法显示: {repr(e)}>"


def recursive_print(obj, indent=0, depth=0, max_depth=MAX_DEPTH, lines=None):
    """递归打印 dict/list/tuple 结构。"""
    if lines is None:
        lines = []

    prefix = "  " * indent

    if depth > max_depth:
        lines.append(prefix + "... 超过最大递归深度 ...")
        return lines

    if isinstance(obj, dict):
        lines.append(prefix + f"dict, keys={list(obj.keys())}")
        for k, v in obj.items():
            if isinstance(v, dict):
                lines.append(prefix + f"- {k}: dict")
                recursive_print(v, indent + 1, depth + 1, max_depth, lines)
            elif isinstance(v, (list, tuple)):
                lines.append(prefix + f"- {k}: {type(v).__name__}, len={len(v)}")
                if len(v) > 0:
                    lines.append(prefix + f"  第 0 个元素:")
                    recursive_print(v[0], indent + 2, depth + 1, max_depth, lines)
            else:
                lines.append(prefix + f"- {k}: {short_value(v)}")

    elif isinstance(obj, (list, tuple)):
        lines.append(prefix + f"{type(obj).__name__}, len={len(obj)}")
        for i, item in enumerate(obj[:3]):
            lines.append(prefix + f"[{i}]:")
            recursive_print(item, indent + 1, depth + 1, max_depth, lines)
        if len(obj) > 3:
            lines.append(prefix + f"... 其余 {len(obj) - 3} 个省略 ...")

    else:
        lines.append(prefix + short_value(obj))

    return lines


def collect_all_keys(obj, prefix="", keys=None):
    """收集所有嵌套 key 路径。"""
    if keys is None:
        keys = set()

    if isinstance(obj, dict):
        for k, v in obj.items():
            key_path = f"{prefix}.{k}" if prefix else str(k)
            keys.add(key_path)
            collect_all_keys(v, key_path, keys)

    elif isinstance(obj, (list, tuple)):
        if len(obj) > 0:
            collect_all_keys(obj[0], prefix + "[0]", keys)

    return keys


def find_path_like_fields(obj, prefix="", found=None):
    """查找所有可能是路径的字段。"""
    if found is None:
        found = []

    path_keywords = [
        "path", "file", "filename", "pts", "mask", "lidar", "seg", "semantic", "instance"
    ]

    if isinstance(obj, dict):
        for k, v in obj.items():
            key_path = f"{prefix}.{k}" if prefix else str(k)
            lower_k = str(k).lower()

            if any(word in lower_k for word in path_keywords):
                found.append((key_path, short_value(v)))

            find_path_like_fields(v, key_path, found)

    elif isinstance(obj, (list, tuple)):
        if len(obj) > 0:
            find_path_like_fields(obj[0], prefix + "[0]", found)

    return found


def load_pkl(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def get_data_list(data):
    """兼容不同 pkl 格式。"""
    if isinstance(data, dict):
        if "data_list" in data:
            return data["data_list"]
        if "infos" in data:
            return data["infos"]
        if "data_infos" in data:
            return data["data_infos"]
    elif isinstance(data, list):
        return data

    return None


def analyze_one_pkl(pkl_path: Path):
    lines = []

    lines.append("=" * 100)
    lines.append(f"PKL 文件: {pkl_path}")
    lines.append("=" * 100)

    if not pkl_path.exists():
        lines.append(f"[ERROR] 文件不存在: {pkl_path}")
        return lines

    data = load_pkl(pkl_path)

    lines.append(f"顶层类型: {type(data)}")

    if isinstance(data, dict):
        lines.append(f"顶层 keys: {list(data.keys())}")

        if "metainfo" in data:
            lines.append("\n[metainfo]")
            lines.extend(recursive_print(data["metainfo"], indent=1))

        if "metadata" in data:
            lines.append("\n[metadata]")
            lines.extend(recursive_print(data["metadata"], indent=1))

    data_list = get_data_list(data)

    if data_list is None:
        lines.append("\n[ERROR] 没有找到 data_list / infos / data_infos")
        lines.append("\n完整顶层结构:")
        lines.extend(recursive_print(data, indent=1))
        return lines

    lines.append(f"\ndata_list 类型: {type(data_list)}")
    lines.append(f"data_list 长度: {len(data_list)}")

    if len(data_list) == 0:
        lines.append("[ERROR] data_list 为空")
        return lines

    # =====================================================
    # 打印 data_list[0] 完整结构
    # =====================================================
    lines.append("\n" + "-" * 100)
    lines.append("data_list[0] 完整递归结构:")
    lines.append("-" * 100)
    lines.extend(recursive_print(data_list[0], indent=1))

    # =====================================================
    # 打印前几个样本的原始 pprint
    # =====================================================
    lines.append("\n" + "-" * 100)
    lines.append(f"前 {min(NUM_SHOW_ITEMS, len(data_list))} 个 data_info 原始内容:")
    lines.append("-" * 100)

    for i, item in enumerate(data_list[:NUM_SHOW_ITEMS]):
        lines.append(f"\n[data_list[{i}]]")
        pretty = pprint.pformat(item, width=140, compact=False)
        lines.append(pretty)

    # =====================================================
    # 收集所有 key 路径
    # =====================================================
    all_keys = sorted(collect_all_keys(data_list[0]))

    lines.append("\n" + "-" * 100)
    lines.append("data_list[0] 所有嵌套 key 路径:")
    lines.append("-" * 100)

    for k in all_keys:
        lines.append(k)

    # =====================================================
    # 路径相关字段
    # =====================================================
    path_fields = find_path_like_fields(data_list[0])

    lines.append("\n" + "-" * 100)
    lines.append("data_list[0] 可能是路径/标签相关的字段:")
    lines.append("-" * 100)

    for k, v in path_fields:
        lines.append(f"{k}: {v}")

    # =====================================================
    # 对所有样本统计 key 是否一致
    # =====================================================
    lines.append("\n" + "-" * 100)
    lines.append("检查所有 data_info 的顶层 key 是否一致:")
    lines.append("-" * 100)

    first_keys = set(data_list[0].keys()) if isinstance(data_list[0], dict) else set()
    all_same = True

    for i, item in enumerate(data_list):
        if not isinstance(item, dict):
            lines.append(f"[ERROR] data_list[{i}] 不是 dict，而是 {type(item)}")
            all_same = False
            continue

        keys = set(item.keys())
        if keys != first_keys:
            all_same = False
            lines.append(f"[WARNING] data_list[{i}] keys 不一致")
            lines.append(f"  缺少: {sorted(first_keys - keys)}")
            lines.append(f"  多出: {sorted(keys - first_keys)}")

    if all_same:
        lines.append("所有样本顶层 key 一致。")
        lines.append(f"顶层 keys: {sorted(first_keys)}")

    # =====================================================
    # 统计 sample_id / 路径字段
    # =====================================================
    lines.append("\n" + "-" * 100)
    lines.append("样本摘要:")
    lines.append("-" * 100)

    for i, item in enumerate(data_list[:20]):
        sample_id = item.get("sample_id", item.get("sample_idx", item.get("lidar_idx", f"index_{i}")))

        lines.append(f"\n[{i}] sample_id = {sample_id}")

        for key in [
            "pts_path",
            "pts_semantic_mask_path",
            "pts_instance_mask_path",
            "super_pts_path",
            "lidar_points",
            "pts_filename",
            "point_cloud_path",
        ]:
            if key in item:
                lines.append(f"  {key}: {short_value(item[key])}")

        if "lidar_points" in item and isinstance(item["lidar_points"], dict):
            for k, v in item["lidar_points"].items():
                lines.append(f"  lidar_points.{k}: {short_value(v)}")

    if len(data_list) > 20:
        lines.append(f"\n其余 {len(data_list) - 20} 个样本省略。")

    return lines


def main():
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_lines = []

    for pkl_path in PKL_FILES:
        lines = analyze_one_pkl(pkl_path)
        all_lines.extend(lines)
        all_lines.append("\n\n")

    text = "\n".join(all_lines)

    print(text)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(text)

    print("\n报告已保存:")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()