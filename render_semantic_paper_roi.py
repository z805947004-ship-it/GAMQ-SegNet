# -*- coding: utf-8 -*-
"""多模型语义分割论文可视化：统一视角 + 10图同步ROI + 输入RGB无框。

面板共11张：Input_RGB, GT, PointNet++, DGCNN, RandLA-Net, PointMLP,
PMSPA-Net, SLNet, 3PNet, OneFormer3D, GAMQ-SegNet。

直接读取原PLY，按旧版 render_semantic_paper_v2.py 的相机/点大小/白底绘制，
先在GUI预览，然后在任意一张非Input图上拖动鼠标选ROI，其余9张实时同步。
左下角预览对应区域，点击 Confirm & Save 才导出11张独立PNG、拼版图和ROI坐标。
输入图保持原样不加框、不加放大图。整个过程只处理可视化，不修改PLY。

依赖: numpy, matplotlib, Pillow, tkinter（Windows/conda通用）。

用法同原版: python render_semantic_paper_roi.py --reference ... --raw ...
    --files <全部9个模型的预测PLY> --out-dir ...

额外选项：
  --preview-cols 6     预览列数（11张默认两行，每行6列）
  --roi x0 y0 x1 y1   已知像素ROI时绕过GUI，直接导出（供批量/复核）
  --no-montage         不额外导出拼版图
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
if '--roi' in sys.argv or '--help' in sys.argv or '-h' in sys.argv:
    matplotlib.use('Agg')
else:
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button
from PIL import Image, ImageDraw, ImageFont

TYPE_MAP = {
    'char': 'i1', 'uchar': 'u1', 'short': '<i2', 'ushort': '<u2',
    'int': '<i4', 'uint': '<u4', 'float': '<f4', 'double': '<f8',
    'int8': 'i1', 'uint8': 'u1', 'int16': '<i2', 'uint16': '<u2',
    'int32': '<i4', 'uint32': '<u4', 'float32': '<f4', 'float64': '<f8',
}
MODEL_ORDER = ('PointNet++', 'DGCNN', 'RandLA-Net', 'PointMLP',
               'PMSPA-Net', 'SLNet', '3PNet', 'OneFormer3D', 'GAMQ-SegNet')
PANEL_ORDER = ('Input_RGB', 'GT', *MODEL_ORDER)
BOX_RED = '#e02020'


def read_ply(path, require_rgb=False):
    """复用旧版算法：仅支持binary_little_endian、顶点带XYZ和可选RGB。"""
    with open(path, 'rb') as f:
        header = []
        while True:
            row = f.readline()
            if not row:
                raise ValueError(f'Missing end_header: {path}')
            txt = row.decode('ascii', errors='replace').strip()
            header.append(txt)
            if txt == 'end_header':
                break
        if 'format binary_little_endian 1.0' not in header:
            raise ValueError(f'Only binary_little_endian PLY supported: {path}')
        n = None
        vertex = False
        props = []
        for h in header:
            p = h.split()
            if p[:2] == ['element', 'vertex']:
                n, vertex = int(p[2]), True
            elif p[:1] == ['element']:
                vertex = False
            elif vertex and p[:1] == ['property']:
                if len(p) != 3 or p[1] not in TYPE_MAP:
                    raise ValueError(f'Unsupported vertex property: {h}')
                props.append((p[2], TYPE_MAP[p[1]]))
        if n is None or not all(k in [x[0] for x in props] for k in ('x', 'y', 'z')):
            raise ValueError(f'Invalid vertex XYZ: {path}')
        data = np.fromfile(f, dtype=np.dtype(props), count=n)
    if len(data) != n:
        raise ValueError(f'PLY truncated: {path}')
    xyz = np.stack([data[k].astype(np.float64) for k in ('x', 'y', 'z')], axis=1)
    if not np.isfinite(xyz).all():
        raise ValueError(f'Nonfinite coordinates: {path}')
    if all(k in data.dtype.names for k in ('red', 'green', 'blue')):
        rgb = np.stack([data[k] for k in ('red', 'green', 'blue')], axis=1).astype(np.float64)
        if rgb.max() > 1.5:
            rgb /= 255.0
        rgb = np.clip(rgb, 0.0, 1.0)
    elif require_rgb:
        raise ValueError(f'Original RGB PLY lacks RGB: {path}')
    else:
        print(f'[WARNING] Missing RGB in {path}, display as gray.')
        rgb = np.full((len(xyz), 3), 0.3)
    return xyz, rgb


def unit(x):
    n = np.linalg.norm(x)
    if n < 1e-10:
        raise ValueError('Cannot construct view from zero-length vector')
    return x / n


def camera_from_gt(xyz, rgb, flip=False):
    stem = (rgb[:, 1] > rgb[:, 0]) & (rgb[:, 1] > rgb[:, 2])
    leaf = rgb[:, 0] > rgb[:, 1]
    if stem.sum() < 20 or leaf.sum() < 20:
        raise ValueError('GT must contain green stem and orange leaf points')
    s, l = xyz[stem], xyz[leaf]
    _, vec = np.linalg.eigh(np.cov(s.T))
    up = unit(vec[:, -1])
    if np.dot(up, l.mean(0) - s.mean(0)) < 0:
        up = -up
    q = l - l.mean(0)
    q -= np.outer(q @ up, up)
    _, vec = np.linalg.eigh(np.cov(q.T))
    right = unit(vec[:, -1] - np.dot(vec[:, -1], up) * up)
    if right[0] < 0:
        right = -right
    if flip:
        right = -right
    return right, up


def canvas_png(uv, rgb, xlim, ylim, width, height, point_size):
    """与v2相同的scatter视角、坐标轴及输出尺寸；输出PIL图便于快速交互。"""
    fig = Figure(figsize=(width / 150.0, height / 150.0), dpi=150, facecolor='white')
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1], facecolor='white')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.scatter(uv[:, 0], uv[:, 1], c=rgb, s=point_size,
               marker='.', linewidths=0, rasterized=True)
    canvas.draw()
    arr = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
    return Image.fromarray(arr, 'RGB')


def normalized_roi(roi, width, height, min_px=6):
    x0, y0, x1, y1 = [float(v) for v in roi]
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    x0, x1 = int(round(max(0, x0))), int(round(min(width, x1)))
    y0, y1 = int(round(max(0, y0))), int(round(min(height, y1)))
    if x1 - x0 < min_px or y1 - y0 < min_px:
        return None
    return x0, y0, x1, y1


def add_roi_and_zoom(src, roi):
    """相同像素ROI；每张图独立生成左下放大图，Input跳过此函数。"""
    result = src.copy()
    w, h = result.size
    x0, y0, x1, y1 = roi
    draw = ImageDraw.Draw(result)
    line_width = max(2, int(round(w / 450)))
    for step in range(line_width):
        draw.rectangle((x0 - step, y0 - step, x1 + step, y1 + step),
                       outline=(224, 32, 32), width=1)
    # 先从未画框的原图提取ROI，避免把红框误认为模型结果。
    roi_img = src.crop((x0, y0, x1, y1))
    left = max(8, int(round(w * 0.03)))
    lower_margin = max(8, int(round(h * 0.03)))
    inset_w = int(round(w * 0.46))
    inset_h = int(round(h * 0.29))
    # 等比例保留局部几何，不拉伸，空白处用纯白填充。
    inset = Image.new('RGB', (inset_w, inset_h), 'white')
    sc = min((inset_w - 10) / roi_img.width, (inset_h - 10) / roi_img.height)
    new_size = (max(1, int(roi_img.width * sc)), max(1, int(roi_img.height * sc)))
    resampler = Image.Resampling.NEAREST if sc > 1.0 else Image.Resampling.LANCZOS
    scaled = roi_img.resize(new_size, resampler)
    inset.paste(scaled, ((inset_w - new_size[0]) // 2,
                         (inset_h - new_size[1]) // 2))
    top = h - lower_margin - inset_h
    result.paste(inset, (left, top))
    draw = ImageDraw.Draw(result)
    draw.rectangle((left, top, left + inset_w - 1, top + inset_h - 1),
                   outline=(224, 32, 32), width=line_width)
    return result


def make_montage(ordered, outfile, sample):
    """11图的带列名对比拼版，仍输出单张图供论文自由排版。"""
    cols = 6
    rows = int(np.ceil(len(ordered) / cols))
    tile_w = 330
    tile_h = 450
    header_h = 28
    pad = 18
    montage = Image.new('RGB', (cols * (tile_w + pad) + pad,
                                rows * (tile_h + header_h + pad) + 62), 'white')
    pen = ImageDraw.Draw(montage)
    pen.text((pad, 12), f'Semantic segmentation: {sample}  |  Shared ROI (Input without ROI)', fill='black')
    for idx, (name, im) in enumerate(ordered):
        r, c = divmod(idx, cols)
        ox = pad + c * (tile_w + pad)
        oy = 50 + r * (tile_h + header_h + pad)
        thumbnail = im.copy()
        thumbnail.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
        montage.paste(thumbnail, (ox + (tile_w - thumbnail.width) // 2, oy + header_h))
        pen.text((ox + 5, oy + 5), name, fill='black')
    montage.save(outfile)


class InteractiveROI:
    """任何非Input面板绘制共享ROI，按像素坐标同步其他9个面板。"""

    def __init__(self, image_map, sample, cols=6):
        self.images = image_map
        self.names = list(image_map)
        self.sample = sample
        self.cols = cols
        self.roi = None
        self.result = None
        self.source_ax = None
        self.start_xy = None
        self.dragging = False
        self.axes = []
        self.patches = []
        self.zoom_artists = []
        self.preview_img_artists = []
        self.w, self.h = next(iter(image_map.values())).size
        rows = int(np.ceil(len(self.names) / cols))
        # GUI把12图排列在宽窗口；与保存图片的分辨率互不影响。
        self.fig, grid = plt.subplots(rows, cols, figsize=(17.5, 9.5 if rows == 2 else 12.0),
                                     facecolor='white', squeeze=False)
        self.fig.subplots_adjust(left=0.018, right=0.988,
                                 top=0.89, bottom=0.15,
                                 hspace=0.12, wspace=0.05)
        self.fig.suptitle(f'{sample}  |  Drag ROI on GT / any prediction; Input is excluded',
                          fontsize=14, fontweight='semibold', y=0.973)
        self.status = self.fig.text(0.5, 0.105,
                                    'Drag left mouse button. The same red box appears on all 10 semantic panels.',
                                    ha='center', fontsize=11)
        self.ax_to_index = {}
        for k, ax in enumerate(grid.flat):
            if k >= len(self.names):
                ax.axis('off')
                continue
            name = self.names[k]
            self.axes.append(ax)
            self.ax_to_index[ax] = k
            ax.imshow(np.asarray(self.images[name]), extent=(0, self.w, self.h, 0), interpolation='nearest')
            ax.set_xlim(0, self.w)
            ax.set_ylim(self.h, 0)
            ax.set_aspect('equal')
            ax.axis('off')
            ax.set_title(name, fontsize=11, fontweight='bold', pad=4)
            if k == 0:
                self.patches.append(None)
            else:
                rect = Rectangle((0, 0), 0, 0, fill=False, edgecolor=BOX_RED,
                                 linewidth=1.8, linestyle='solid', visible=False, zorder=20)
                ax.add_patch(rect)
                self.patches.append(rect)
            self.zoom_artists.append(None)
        try:
            self.fig.canvas.manager.set_window_title('Semantic ROI (11 models) - ' + sample)
        except (AttributeError, RuntimeError):
            pass
        self.buttons = []
        for cx, label, callback, color in (
            (0.18, 'Confirm & Save', self._confirm, '#d6eddd'),
            (0.40, 'Redraw ROI', self._reset, '#f6e5d9'),
            (0.61, 'Skip Sample', self._skip, '#e6e6e6'),
            (0.82, 'Quit Batch', self._quit, '#e6e6e6'),
        ):
            ax_button = self.fig.add_axes([cx - .09, 0.027, 0.18, 0.055])
            btn = Button(ax_button, label, color=color, hovercolor='#d0e2f4')
            btn.on_clicked(callback)
            self.buttons.append(btn)
        canvas = self.fig.canvas
        canvas.mpl_connect('button_press_event', self._press)
        canvas.mpl_connect('motion_notify_event', self._move)
        canvas.mpl_connect('button_release_event', self._release)
        canvas.mpl_connect('close_event', self._closed)

    def _cursor_xy(self, event):
        x, y = self.source_ax.transData.inverted().transform((event.x, event.y))
        return (float(np.clip(x, 0, self.w)), float(np.clip(y, 0, self.h)))

    def _show_boxes(self, roi):
        for i in range(1, len(self.names)):
            rect = self.patches[i]
            if roi is None:
                rect.set_visible(False)
            else:
                x0, y0, x1, y1 = roi
                rect.set_xy((x0, y0))
                rect.set_width(x1 - x0)
                rect.set_height(y1 - y0)
                rect.set_visible(True)
        self.fig.canvas.draw_idle()

    def _remove_insets(self):
        for i in range(1, len(self.names)):
            artist = self.zoom_artists[i]
            if artist is not None:
                artist.remove()
                self.zoom_artists[i] = None

    def _press(self, event):
        idx = self.ax_to_index.get(event.inaxes)
        if event.button != 1 or idx is None or idx == 0 or self.result is not None:
            return
        self._remove_insets()
        self.roi = None
        self.source_ax = event.inaxes
        self.start_xy = self._cursor_xy(event)
        self.dragging = True
        self.status.set_text('Selecting one ROI on 10 semantic outputs...')
        self._show_boxes(None)

    def _move(self, event):
        if not self.dragging:
            return
        x, y = self._cursor_xy(event)
        x0, x1 = sorted((x, self.start_xy[0]))
        y0, y1 = sorted((y, self.start_xy[1]))
        self._show_boxes((x0, y0, x1, y1))

    def _release(self, event):
        if not self.dragging:
            return
        x, y = self._cursor_xy(event)
        self.dragging = False
        self.source_ax = None
        roi = normalized_roi((self.start_xy[0], self.start_xy[1], x, y), self.w, self.h)
        self.start_xy = None
        if roi is None:
            self._show_boxes(None)
            self.status.set_text('ROI too small. Please drag a larger rectangle.')
            return
        self.roi = roi
        self._show_boxes(roi)
        # 在当前GUI中显示每张图各自相同位置的左下角放大图。
        for i in range(1, len(self.names)):
            display = add_roi_and_zoom(self.images[self.names[i]], roi)
            self.zoom_artists[i] = self.axes[i].imshow(
                np.asarray(display), extent=(0, self.w, self.h, 0),
                interpolation='nearest', zorder=10)
        self.status.set_text('GT + 9 models synchronized. Input unchanged. Confirm & Save when ready.')
        self.fig.canvas.draw_idle()

    def _confirm(self, _):
        if self.roi is None:
            self.status.set_text('Draw a ROI on GT or a model first; nothing saved yet.')
            self.fig.canvas.draw_idle()
            return
        self.result = 'save'
        plt.close(self.fig)

    def _reset(self, _):
        self._remove_insets()
        self._show_boxes(None)
        self.roi = None
        self.dragging = False
        self.source_ax = None
        self.start_xy = None
        self.status.set_text('ROI cleared. Drag again on GT or any of the 9 predictions.')
        self.fig.canvas.draw_idle()

    def _skip(self, _):
        self.result = 'skip'
        plt.close(self.fig)

    def _quit(self, _):
        self.result = 'quit'
        plt.close(self.fig)

    def _closed(self, _):
        if self.result is None:
            self.result = 'quit'

    def show(self):
        plt.show(block=True)
        return self.result, self.roi


def process_one(args):
    paths = [args.raw, args.reference, *args.files]
    if len(args.files) != len(MODEL_ORDER):
        raise ValueError(f'Expected 9 prediction PLYs, got {len(args.files)}. Order: {MODEL_ORDER}')
    for f in paths:
        if not f.is_file():
            raise FileNotFoundError(f'Missing file: {f}')
    names = [f.stem for f in paths]
    if len(set(names)) != len(names):
        raise ValueError('Duplicate PLY stems: output PNGs would overwrite each other')
    if args.width <= 2 * args.margin or args.height <= 2 * args.margin:
        raise ValueError('Output dimensions must exceed 2 * margin')
    clouds = [read_ply(f, require_rgb=(f == args.raw)) for f in paths]
    right, up = camera_from_gt(*clouds[1], flip=args.flip_horizontal)
    gtxyz = clouds[1][0]
    rawxyz = clouds[0][0]
    if np.any(np.abs(np.ptp(rawxyz, axis=0) - np.ptp(gtxyz, axis=0)) >
              np.maximum(np.ptp(gtxyz, axis=0), 1e-5) * 0.10):
        print('[WARNING] Raw/GT coordinate extents differ >10%; inspect alignment!')
    uv = [np.column_stack((xyz @ right, xyz @ up)) for xyz, _ in clouds]
    whole = np.vstack(uv)
    low, high = whole.min(axis=0), whole.max(axis=0)
    span = np.maximum(high - low, 1e-8)
    scale = min((args.width - 2 * args.margin) / span[0],
                (args.height - 2 * args.margin) / span[1])
    center = (high + low) / 2
    xlim = (center[0] - args.width / (2 * scale),
            center[0] + args.width / (2 * scale))
    ylim = (center[1] - args.height / (2 * scale),
            center[1] + args.height / (2 * scale))
    # PS1 中的 files 顺序并非图里的固定顺序，将路径按模型前缀重新排序。
    preds = {}
    for path, result in zip(args.files, clouds[2:]):
        model = next((m for m in MODEL_ORDER if path.stem.startswith(m + '_')), None)
        if model is None:
            raise ValueError('Unrecognized prediction model prefix: ' + path.stem)
        if model in preds:
            raise ValueError('Duplicate model: ' + model)
        preds[model] = (path, result)
    missing = [m for m in MODEL_ORDER if m not in preds]
    if missing:
        raise ValueError(f'Missing models: {missing}')
    mapping = {'Input_RGB': (args.raw, clouds[0], uv[0]),
               'GT': (args.reference, clouds[1], uv[1])}
    proj_by_path = dict(zip(paths, uv))
    for model in MODEL_ORDER:
        p, c = preds[model]
        mapping[model] = (p, c, proj_by_path[p])
    images = {}
    for idx, (name, (p, cloud, proj)) in enumerate(mapping.items(), 1):
        print(f'   [{idx:02d}/11] rendering: {name}')
        images[name] = canvas_png(proj, cloud[1], xlim, ylim,
                                  args.width, args.height, args.point_size)
    if args.roi is None:
        preview = InteractiveROI(images, args.reference.stem, cols=args.preview_cols)
        action, roi = preview.show()
    else:
        roi = normalized_roi(args.roi, args.width, args.height)
        if roi is None:
            raise ValueError('Invalid --roi: must select >=6 pixel extent')
        action = 'save'
    if action != 'save':
        print('[NOT SAVED]', action)
        return 2 if action == 'quit' else 0
    outdir = args.out_dir
    outdir.mkdir(parents=True, exist_ok=True)
    ordered = []
    for name, (path, _, _) in mapping.items():
        png_name = 'Input_RGB.png' if name == 'Input_RGB' else 'GT.png' if name == 'GT' else path.stem + '.png'
        result = images[name] if name == 'Input_RGB' else add_roi_and_zoom(images[name], roi)
        result.save(outdir / png_name)
        print('[PNG]', outdir / png_name)
        ordered.append((name, result))
    if not args.no_montage:
        make_montage(ordered, outdir / 'semantic_roi_montage.png', args.reference.stem)
    with (outdir / 'semantic_shared_roi.json').open('w', encoding='utf-8') as f:
        json.dump({'pixel_roi_xyxy': roi, 'image_size_wh': [args.width, args.height],
                   'panels': list(PANEL_ORDER), 'excluded_from_roi': 'Input_RGB',
                   'source_files': {name: str(item[0]) for name, item in mapping.items()},
                   'projection': {'right': right.tolist(), 'up': up.tolist(),
                                  'xlim': list(xlim), 'ylim': list(ylim)}},
                  f, ensure_ascii=False, indent=2)
    print(f'[DONE] Saved {len(ordered)} single images + ROI data to {outdir}')
    return 0


def main():
    p = argparse.ArgumentParser(description='Shared ROI for all 9 comparison models + GT; raw input unchanged')
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--files', type=Path, nargs='+', required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--width', type=int, default=900)
    p.add_argument('--height', type=int, default=1200)
    p.add_argument('--point-size', type=float, default=3.0)
    p.add_argument('--margin', type=int, default=55)
    p.add_argument('--preview-cols', type=int, default=6)
    p.add_argument('--flip-horizontal', action='store_true')
    p.add_argument('--roi', type=int, nargs=4, metavar=('X0', 'Y0', 'X1', 'Y1'),
                   help='Skip interactive window and export with existing shared ROI pixels')
    p.add_argument('--no-montage', action='store_true')
    a = p.parse_args()
    if a.preview_cols < 2 or a.preview_cols > 11:
        p.error('--preview-cols must be 2..11')
    try:
        status = process_one(a)
    except Exception as exc:
        print('[ERROR]', exc, file=sys.stderr)
        raise
    return status


if __name__ == '__main__':
    sys.exit(main())
