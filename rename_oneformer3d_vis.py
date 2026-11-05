# -*- coding: utf-8 -*-
"""Rename/copy legacy 0..10 OneFormer3D visualization files using verified sample IDs.

Example (PowerShell, from OneFormer3D project root):
    python rename_oneformer3d_vis.py --src 'work_dirs\full_vis' --dst 'work_dirs\full_vis_named'

By default, only pred_sem and pred_inst files (.ply/.npy) are copied.
Add --include-gt to copy GT files with model-independent names.
Use --move only after checking the copied results.
"""

import argparse
import csv
import re
import shutil
from pathlib import Path

# Verified against corn_infos_test.pkl and distinct point counts in the test logs.
SAMPLES = [
    ('1149-12(2)', 10553),
    ('384-2(2)', 14812),
    ('1149-23(2)', 16373),
    ('1149-24(2)', 14473),
    ('384-31(2)', 12179),
    ('384-3(2)', 15600),
    ('1149-13(2)', 8621),
    ('1149-36(2)', 19631),
    ('384-26(2)', 16294),
    ('336-21(2)', 8110),
    ('336-23(2)', 5390),
]

PATTERN = re.compile(r'^([0-9]+)_(pred_sem|pred_inst|gt_sem|gt_inst)\.(ply|npy)$', re.IGNORECASE)


def ply_vertex_count(path):
    """Read point count from the PLY header; do not load or modify point data."""
    with path.open('rb') as f:
        first = f.readline().strip()
        if first != b'ply':
            raise ValueError(f'不是有效的PLY文件: {path}')
        for _ in range(200):
            line = f.readline()
            if not line:
                break
            m = re.match(rb'element\s+vertex\s+(\d+)\s*$', line.strip())
            if m:
                return int(m.group(1))
            if line.strip() == b'end_header':
                break
    raise ValueError(f'没有从PLY头部读取到vertex数量: {path}')


def main():
    parser = argparse.ArgumentParser(description='OneFormer3D 0..10 文件重命名为真实样本ID')
    parser.add_argument('--src', required=True, help='旧文件所在目录，如 work_dirs/full_vis')
    parser.add_argument('--dst', required=True, help='新文件输出目录，如 work_dirs/full_vis_named')
    parser.add_argument('--model', default='OneFormer3D', help='预测文件名称前缀，默认 OneFormer3D')
    parser.add_argument('--include-gt', action='store_true', help='也复制GT；GT不加模型前缀')
    parser.add_argument('--move', action='store_true', help='移动而非复制；不建议首次运行使用')
    parser.add_argument('--dry-run', action='store_true', help='只预览映射，不写入任何文件')
    args = parser.parse_args()

    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()
    if not src.is_dir():
        parser.error(f'输入目录不存在: {src}')
    if src == dst and not args.move:
        parser.error('为了避免混杂旧文件，请将 --dst 设置为不同目录')
    if not args.model or re.search(r'[\\/:*?"<>|]', args.model):
        parser.error('--model 前缀无效')

    matches = []
    for old_path in sorted(src.iterdir()):
        if not old_path.is_file():
            continue
        match = PATTERN.fullmatch(old_path.name)
        if match is None:
            continue
        idx, kind, ext = int(match.group(1)), match.group(2).lower(), match.group(3).lower()
        if idx >= len(SAMPLES):
            raise ValueError(f'旧编号超出0~10范围: {old_path.name}')
        if kind.startswith('gt_') and not args.include_gt:
            continue

        sample_id, expected_n = SAMPLES[idx]
        if ext == 'ply':
            actual_n = ply_vertex_count(old_path)
            if actual_n != expected_n:
                raise ValueError(
                    f'点数不匹配，停止处理: {old_path.name}；'
                    f'旧编号{idx}应有{expected_n}点，实际{actual_n}点。'
                    '请检查是否使用了不同数据集或采样设置。'
                )

        if kind.startswith('pred_'):
            new_name = f'{args.model}_{sample_id}_{kind}.{ext}'
        else:
            new_name = f'{sample_id}_{kind}.{ext}'
        matches.append((old_path, dst / new_name, idx, sample_id, expected_n))

    if not matches:
        print('没有找到需要处理的文件。预期文件名例如: 0_pred_sem.ply')
        return

    # Collision-check the entire batch before any copy/move.
    seen = set()
    for old_path, new_path, *_ in matches:
        if new_path in seen:
            raise FileExistsError(f'发生目标文件名冲突: {new_path}')
        seen.add(new_path)
        if new_path.exists():
            raise FileExistsError(f'目标文件已存在，为防止覆盖已停止: {new_path}')

    print(f'源目录: {src}')
    print(f'目标目录: {dst}')
    print(f'操作: {"仅预览" if args.dry_run else "移动" if args.move else "复制（保留原文件）"}')
    print(f'匹配文件: {len(matches)}\n')

    if not args.dry_run:
        dst.mkdir(parents=True, exist_ok=True)
    for n, (old_path, new_path, idx, sample_id, expected_n) in enumerate(matches, 1):
        print(f'[{n}/{len(matches)}] {old_path.name} -> {new_path.name}  (N={expected_n})')
        if not args.dry_run:
            if args.move:
                shutil.move(str(old_path), str(new_path))
            else:
                shutil.copy2(old_path, new_path)

    if not args.dry_run:
        manifest = dst / 'rename_manifest.csv'
        with manifest.open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['old_file', 'new_file', 'index', 'sample_id', 'expected_points'])
            for old_path, new_path, idx, sample_id, expected_n in matches:
                writer.writerow([old_path.name, new_path.name, idx, sample_id, expected_n])
        print(f'\n完成。映射记录: {manifest}')
    else:
        print('\n预览完成，未修改文件。')


if __name__ == '__main__':
    main()
