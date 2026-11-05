from pathlib import Path
import numpy as np


# =========================================================
# 这里直接指定你的 txt 文件夹路径
# =========================================================
TXT_DIR = Path(r"D:\onef\2time\标注")   # 改成你自己的 txt 文件夹
PATTERN = "*.txt"

# 第 7 列是 label，Python 下标是 6
LABEL_COL = 6


def read_txt(txt_path: Path):
    """读取 txt，兼容空格分隔和逗号分隔。"""
    try:
        arr = np.loadtxt(str(txt_path), dtype=np.float64)
    except Exception:
        arr = np.loadtxt(str(txt_path), dtype=np.float64, delimiter=",")

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    return arr


def check_one_file(txt_path: Path):
    result = {
        "file": txt_path.name,
        "ok": True,
        "errors": [],
        "warnings": [],
        "num_points": 0,
        "labels": [],
        "stem_count": 0,
        "leaf_labels": [],
        "leaf_counts": {},
        "missing_leaf_labels": [],
        "invalid_labels": [],
    }

    try:
        arr = read_txt(txt_path)
    except Exception as e:
        result["ok"] = False
        result["errors"].append(f"读取失败: {repr(e)}")
        return result

    if arr.shape[1] <= LABEL_COL:
        result["ok"] = False
        result["errors"].append(
            f"列数不足：当前 {arr.shape[1]} 列，至少需要 7 列，label 应在第 7 列"
        )
        return result

    finite_mask = np.isfinite(arr).all(axis=1)
    if not finite_mask.all():
        bad_num = int((~finite_mask).sum())
        result["warnings"].append(f"存在 {bad_num} 行 NaN/Inf，检查时已忽略")
        arr = arr[finite_mask]

    if arr.shape[0] == 0:
        result["ok"] = False
        result["errors"].append("有效点数为 0")
        return result

    raw_label_float = arr[:, LABEL_COL]

    rounded = np.rint(raw_label_float)
    not_integer_mask = ~np.isclose(raw_label_float, rounded, atol=1e-6)

    if np.any(not_integer_mask):
        bad_values = np.unique(raw_label_float[not_integer_mask])
        result["ok"] = False
        result["errors"].append(
            f"存在非整数标签: {bad_values[:20].tolist()}"
        )

    labels = rounded.astype(np.int64)

    result["num_points"] = int(labels.shape[0])
    unique_labels = sorted(np.unique(labels).tolist())
    result["labels"] = unique_labels

    # =====================================================
    # 规则 1：必须有 label=0，表示茎
    # =====================================================
    stem_count = int((labels == 0).sum())
    result["stem_count"] = stem_count

    if stem_count == 0:
        result["ok"] = False
        result["errors"].append("缺少 label=0，也就是没有茎标签")

    # =====================================================
    # 规则 2：叶片标签必须从 11 开始连续
    # 允许：
    #   11
    #   11,12
    #   11,12,13
    # 不允许：
    #   12
    #   11,13
    # =====================================================
    leaf_labels = sorted([x for x in unique_labels if x >= 11])
    result["leaf_labels"] = leaf_labels

    for lab in leaf_labels:
        result["leaf_counts"][int(lab)] = int((labels == lab).sum())

    if len(leaf_labels) == 0:
        result["ok"] = False
        result["errors"].append("没有任何叶片标签，至少应包含 11")
    else:
        expected_leaf_labels = list(range(11, max(leaf_labels) + 1))
        missing = [x for x in expected_leaf_labels if x not in leaf_labels]
        result["missing_leaf_labels"] = missing

        if leaf_labels[0] != 11:
            result["ok"] = False
            result["errors"].append(
                f"叶片标签必须从 11 开始，但当前最小叶片标签是 {leaf_labels[0]}"
            )

        if missing:
            result["ok"] = False
            result["errors"].append(
                f"叶片标签不连续：当前 leaf={leaf_labels}，缺失 {missing}"
            )

    # =====================================================
    # 规则 3：不允许出现非法标签
    # 合法标签 = 1 + 连续叶片标签 11,12,13...
    # =====================================================
    valid_labels = {0}
    if len(leaf_labels) > 0:
        valid_labels.update(range(11, max(leaf_labels) + 1))

    invalid_labels = sorted([x for x in unique_labels if x not in valid_labels])
    result["invalid_labels"] = invalid_labels

    if invalid_labels:
        result["ok"] = False
        result["errors"].append(
            f"存在非法标签: {invalid_labels}。合法标签应为 0 和连续的 11,12,13..."
        )

    return result


def print_result(result):
    status = "OK" if result["ok"] else "ERROR"

    print("\n" + "=" * 80)
    print(f"[{status}] {result['file']}")
    print("-" * 80)
    print(f"点数: {result['num_points']}")
    print(f"所有标签: {result['labels']}")
    print(f"茎 label=0 点数: {result['stem_count']}")
    print(f"叶片标签: {result['leaf_labels']}")

    if result["leaf_counts"]:
        print("叶片点数统计:")
        for lab, cnt in result["leaf_counts"].items():
            print(f"  label {lab}: {cnt}")

    if result["missing_leaf_labels"]:
        print(f"缺失叶片标签: {result['missing_leaf_labels']}")

    if result["invalid_labels"]:
        print(f"非法标签: {result['invalid_labels']}")

    if result["warnings"]:
        print("警告:")
        for w in result["warnings"]:
            print(f"  - {w}")

    if result["errors"]:
        print("错误:")
        for e in result["errors"]:
            print(f"  - {e}")


def save_report(results, report_path: Path):
    report_path.parent.mkdir(parents=True, exist_ok=True)

    ok_count = sum(1 for r in results if r["ok"])
    error_count = sum(1 for r in results if not r["ok"])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"检查文件夹: {TXT_DIR}\n")
        f.write(f"txt 数量: {len(results)}\n")
        f.write(f"通过: {ok_count}\n")
        f.write(f"失败: {error_count}\n\n")

        for r in results:
            status = "OK" if r["ok"] else "ERROR"
            f.write("=" * 80 + "\n")
            f.write(f"[{status}] {r['file']}\n")
            f.write(f"点数: {r['num_points']}\n")
            f.write(f"所有标签: {r['labels']}\n")
            f.write(f"茎 label=0 点数: {r['stem_count']}\n")
            f.write(f"叶片标签: {r['leaf_labels']}\n")
            f.write(f"叶片点数统计: {r['leaf_counts']}\n")
            f.write(f"缺失叶片标签: {r['missing_leaf_labels']}\n")
            f.write(f"非法标签: {r['invalid_labels']}\n")

            if r["warnings"]:
                f.write("警告:\n")
                for w in r["warnings"]:
                    f.write(f"  - {w}\n")

            if r["errors"]:
                f.write("错误:\n")
                for e in r["errors"]:
                    f.write(f"  - {e}\n")

            f.write("\n")


def main():
    print("检查文件夹:", TXT_DIR)
    print("匹配规则:", PATTERN)

    txt_files = sorted(TXT_DIR.glob(PATTERN))

    if len(txt_files) == 0:
        raise FileNotFoundError(f"没有找到 txt 文件: {TXT_DIR / PATTERN}")

    print("txt 数量:", len(txt_files))

    results = []
    ok_count = 0
    error_count = 0

    for txt_path in txt_files:
        result = check_one_file(txt_path)
        results.append(result)
        print_result(result)

        if result["ok"]:
            ok_count += 1
        else:
            error_count += 1

    print("\n" + "=" * 80)
    print("总结果")
    print("=" * 80)
    print(f"总文件数: {len(txt_files)}")
    print(f"通过: {ok_count}")
    print(f"失败: {error_count}")

    if error_count > 0:
        print("\n失败文件:")
        for r in results:
            if not r["ok"]:
                print(f"  - {r['file']}")
    else:
        print("\n所有 txt 标签检查通过。")
if __name__ == "__main__":
    main()