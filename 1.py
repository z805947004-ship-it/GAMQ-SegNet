import os
import glob
import pickle


SEARCH_DIRS = [
    r"D:\onef\2time\oneformer_rgb",
    r"D:\pythoncode\oneformer3d-main",
]


def inspect_obj(obj, prefix="", depth=0):
    if depth > 4:
        return

    indent = "  " * depth

    if isinstance(obj, dict):
        print(f"{indent}{prefix}dict keys = {list(obj.keys())[:30]}")

        for k, v in list(obj.items())[:10]:
            inspect_obj(v, str(k), depth + 1)

    elif isinstance(obj, list):
        print(f"{indent}{prefix}list length = {len(obj)}")

        for i, v in enumerate(obj[:3]):
            inspect_obj(v, f"[{i}]", depth + 1)

    elif isinstance(obj, tuple):
        print(f"{indent}{prefix}tuple length = {len(obj)}")

    elif isinstance(obj, str):
        print(f"{indent}{prefix} = {obj}")

    else:
        print(
            f"{indent}{prefix}: "
            f"{type(obj).__name__}"
        )


files = []

for root in SEARCH_DIRS:
    if not os.path.exists(root):
        continue

    files.extend(
        glob.glob(
            os.path.join(root, "**", "*corn*info*.pkl"),
            recursive=True
        )
    )

    files.extend(
        glob.glob(
            os.path.join(root, "**", "*infos*.pkl"),
            recursive=True
        )
    )

files = sorted(set(files))

print("=" * 100)
print("找到 PKL 文件：")
print("=" * 100)

for f in files:
    print(f)

print("\n总数:", len(files))


for f in files:

    print("\n" + "=" * 100)
    print("FILE:", f)
    print("=" * 100)

    try:
        with open(f, "rb") as fp:
            obj = pickle.load(fp)

        inspect_obj(obj)

    except Exception as e:
        print("读取失败:", repr(e))