import pickle
from pathlib import Path


DATA_ROOT = Path(r"D:\pythoncode\oneformer3d-main\data\corn_dataset").resolve()

PKL_FILES = [
    DATA_ROOT / "corn_infos_train.pkl",
    DATA_ROOT / "corn_infos_val.pkl",
    DATA_ROOT / "corn_infos_test.pkl",
]


def fix_one_pkl(pkl_path: Path):
    if not pkl_path.exists():
        print(f"跳过，不存在: {pkl_path}")
        return

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, list):
        data = {"data_list": data}

    if "data_list" not in data:
        raise RuntimeError(f"{pkl_path} 里没有 data_list，结构不对")

    data["metainfo"] = {
        "classes": ("stem", "leaf"),
        "dataset_name": "S3DIS",
    }

    for item in data["data_list"]:
        sid = item["sample_id"]

        pts_path = DATA_ROOT / f"{sid}_pts.npy"
        sem_path = DATA_ROOT / f"{sid}_sem.npy"
        ins_path = DATA_ROOT / f"{sid}_ins.npy"

        if not pts_path.exists():
            print(f"警告：缺少点云文件: {pts_path}")
        if not sem_path.exists():
            print(f"警告：缺少语义标签文件: {sem_path}")
        if not ins_path.exists():
            print(f"警告：缺少实例标签文件: {ins_path}")

        item["pts_path"] = str(pts_path)
        item["pts_semantic_mask_path"] = str(sem_path)
        item["pts_instance_mask_path"] = str(ins_path)

        item["lidar_points"] = {
            "lidar_path": str(pts_path),
            "num_pts_feats": 3,
        }

        if "instances" not in item:
            item["instances"] = []

    with open(pkl_path, "wb") as f:
        pickle.dump(data, f)

    print(f"已修复: {pkl_path}")
    print(f"样本数: {len(data['data_list'])}")


if __name__ == "__main__":
    for pkl_file in PKL_FILES:
        fix_one_pkl(pkl_file)