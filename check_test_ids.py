from mmengine.fileio import load
from pathlib import Path

path = Path(
    r"D:\onef\2time\oneformer_rgb"
    r"\corn_infos_test.pkl"
)

data = load(str(path))

print("数据类型:", type(data))

if isinstance(data, dict):
    print("顶层字段:", list(data.keys()))

    if "data_list" in data:
        samples = data["data_list"]
    elif "infos" in data:
        samples = data["infos"]
    else:
        raise RuntimeError(
            "未发现data_list或infos，请检查数据结构"
        )
else:
    samples = data

print("\n测试集样本数量:", len(samples))

for i, info in enumerate(samples):
    print(f"\n索引 {i}")
    print("sample_id:", info.get("sample_id"))
    print("sample_idx:", info.get("sample_idx"))

    for key in [
        "pts_path",
        "pts_filename",
        "lidar_points",
        "point_cloud_path",
    ]:
        if key in info:
            print(key, ":", info[key])