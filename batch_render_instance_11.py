
# -*- coding: utf-8 -*-
"""论文用实例分割PNG：11株 x (原始RGB、GT、Baseline、GAASE、AMSQF、Full)：交互同步ROI放大版。

只做可视化，不重新推理、不修改点云/预测标签。
GT给出统一视角、缩放；预测实例通过逐点IoU匹配GT后统一配色（只影响颜色）。

依赖: numpy, matplotlib, scipy, Tkinter (Windows Python/conda常见内置)。
用法: 将原命令中的 batch_render_instance_11.py 换成 batch_render_instance_roi.py。
每株先GUI预览，在任一子图拖红框，所有图同步ROI；确定保存或跳过。
关闭X = 安全停止全部后续样本。加 --no-preview 可兼容原批量渲染。
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg' if ('--no-preview' in sys.argv or '--help' in sys.argv or '-h' in sys.argv) else 'TkAgg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button
from scipy.optimize import linear_sum_assignment

SAMPLES = [
    '1149-12(2)', '384-2(2)', '1149-23(2)', '1149-24(2)',
    '384-31(2)', '384-3(2)', '1149-13(2)', '1149-36(2)',
    '384-26(2)', '336-21(2)', '336-23(2)',
]

PLY_TYPES = {
    'char': 'i1', 'uchar': 'u1', 'short': '<i2', 'ushort': '<u2',
    'int': '<i4', 'uint': '<u4', 'float': '<f4', 'double': '<f8',
    'int8': 'i1', 'uint8': 'u1', 'int16': '<i2', 'uint16': '<u2',
    'int32': '<i4', 'uint32': '<u4', 'float32': '<f4', 'float64': '<f8',
}

# 0是茎；叶片根据GT实例ID使用固定颜色；未分配点用灰色。
STEM_COLOR = np.array([48, 175, 82], dtype=np.float32) / 255
UNKNOWN_COLOR = np.array([160, 160, 160], dtype=np.float32) / 255
LEAF_COLORS = np.array([
    [235, 88, 68], [45, 122, 219], [234, 171, 49],
    [139, 86, 211], [55, 177, 185], [220, 100, 156],
    [147, 174, 49], [239, 130, 47], [90, 96, 220],
    [184, 108, 54], [49, 160, 121], [215, 72, 107],
    [112, 157, 215], [172, 121, 210], [217, 187, 75],
], dtype=np.float32) / 255
EXTRA_COLORS = np.array([
    [68, 67, 76], [202, 48, 159], [64, 155, 148],
    [179, 93, 40], [102, 58, 178], [48, 100, 111],
], dtype=np.float32) / 255


def read_ply(path):
    """读取含XYZ、可选RGB的binary_little_endian/ASCII PLY顶点。"""
    with path.open('rb') as f:
        lines = []
        while True:
            b = f.readline()
            if not b:
                raise ValueError('无end_header: ' + str(path))
            line = b.decode('ascii', errors='replace').strip()
            lines.append(line)
            if line == 'end_header':
                break
        if not lines or lines[0] != 'ply':
            raise ValueError('非PLY文件: ' + str(path))
        fmt = next((s.split()[1] for s in lines if s.startswith('format ')), None)
        if fmt not in ('binary_little_endian', 'ascii'):
            raise ValueError('只支持binary_little_endian或ascii PLY: ' + str(path))
        n = None
        props = []
        elem = None
        seen_earlier_nonvertex = False
        for line in lines:
            tok = line.split()
            if len(tok) >= 3 and tok[0] == 'element':
                elem = tok[1]
                if elem == 'vertex':
                    n = int(tok[2])
                elif n is None and int(tok[2]):
                    seen_earlier_nonvertex = True
            elif elem == 'vertex' and tok[:1] == ['property']:
                if len(tok) != 3 or tok[1] not in PLY_TYPES:
                    raise ValueError('顶点属性不支持: ' + line)
                props.append((tok[2], PLY_TYPES[tok[1]]))
        if seen_earlier_nonvertex:
            raise ValueError('顶点前存在其他element，当前不支持: ' + str(path))
        if n is None or not props:
            raise ValueError('PLY顶点定义缺失: ' + str(path))
        names = [x[0] for x in props]
        if not all(v in names for v in ('x', 'y', 'z')):
            raise ValueError('PLY缺少XYZ属性: ' + str(path))
        if fmt == 'binary_little_endian':
            data = np.fromfile(f, dtype=np.dtype(props), count=n)
        else:
            arr = np.genfromtxt(f, dtype=np.dtype(props), max_rows=n)
            data = np.atleast_1d(arr)
        if len(data) != n:
            raise ValueError('PLY不完整: ' + str(path))
    xyz = np.column_stack([data[c].astype(np.float64) for c in ('x','y','z')])
    if not np.isfinite(xyz).all():
        raise ValueError('坐标非有限数: ' + str(path))
    rgb = None
    if all(s in data.dtype.names for s in ('red', 'green', 'blue')):
        rgb = np.column_stack([data[c].astype(np.float64) for c in ('red','green','blue')])
        if np.max(rgb) > 1.5:
            rgb /= 255.0
        rgb = np.clip(rgb, 0, 1)
    return xyz, rgb


def read_raw(path):
    if path.suffix.lower() == '.ply':
        return read_ply(path)
    if path.suffix.lower() == '.npy':
        points = np.load(path, allow_pickle=False)
        if points.ndim != 2 or points.shape[1] < 6:
            raise ValueError(f'原始NPY不是Nx6 XYZRGB: {path}: {points.shape}')
        xyz = points[:, :3].astype(np.float64)
        rgb = points[:, 3:6].astype(np.float64)
        if np.max(rgb) > 1.5:
            rgb /= 255.0
        return xyz, np.clip(rgb, 0, 1)
    raise ValueError('无法读取原始文件: ' + str(path))


def raw_file(root, sid):
    # 优先选择已知的原始PLY，而不是下游模型数据文件。
    for suffix in ('.ply', '.npy', '_pts.npy'):
        p = root / (sid + suffix)
        if p.is_file():
            return p
    for suffix in ('.ply', '.npy', '_pts.npy'):
        found = list(root.rglob(sid + suffix))
        if len(found) == 1:
            return found[0]
        if len(found) > 1:
            raise ValueError(f'找到多个原始文件，请明确位置: {sid + suffix}: {found}')
    return None


def label_file(folder, prefix, sid, typ):
    return folder / f'{prefix}_{sid}_{typ}.npy'


def ply_file(folder, prefix, sid, typ):
    return folder / f'{prefix}_{sid}_{typ}.ply'


def read_ids(path, n):
    a = np.load(path, allow_pickle=False)
    a = np.asarray(a)
    if a.ndim == 2 and 1 in a.shape:
        a = a.reshape(-1)
    if a.ndim != 1 or a.shape[0] != n:
        raise ValueError(f'实例NPY必须是N长度的逐点ID: {path} shape={a.shape}, N={n}')
    if not np.isfinite(a).all() or not np.equal(a, np.rint(a)).all():
        raise ValueError('实例ID包含非整数/无效值: '+str(path))
    return a.astype(np.int64)


def base_colors(gt):
    unique_gt = sorted(int(z) for z in np.unique(gt) if z not in (-1, 255))
    out = np.repeat(UNKNOWN_COLOR[None], len(gt), axis=0)
    cmap = {}
    for idx, ident in enumerate(z for z in unique_gt if z != 0):
        cmap[ident] = LEAF_COLORS[idx % len(LEAF_COLORS)]
    if 0 in unique_gt:
        cmap[0] = STEM_COLOR
    for ident, color in cmap.items():
        out[gt == ident] = color
    return out, cmap


def matched_pred_colors(pred, gt, gt_palette, threshold=0.05):
    """Hungarian按逐点IoU匹配实例ID；只调整绘图颜色，不改预测ID。"""
    pids = [int(p) for p in np.unique(pred) if p not in (-1, 255)]
    gids = [int(g) for g in np.unique(gt) if g not in (-1, 255)]
    rgb = np.repeat(UNKNOWN_COLOR[None], len(pred), axis=0)
    if not pids:
        return rgb, [], 0
    overlap = np.zeros((len(pids),len(gids)), dtype=np.float64)
    for pi,pid in enumerate(pids):
        a = pred == pid
        for gi,gid in enumerate(gids):
            b = gt == gid
            inter = np.count_nonzero(a & b)
            union = np.count_nonzero(a | b)
            overlap[pi,gi] = inter/union if union else 0
    rows, cols = linear_sum_assignment(-overlap) if len(gids) else ([],[])
    assignments = {}
    for pi, gi in zip(rows,cols):
        if overlap[pi, gi] >= threshold:
            assignments[pids[pi]] = (gids[gi],float(overlap[pi,gi]))
    log = []
    missing_matches = 0
    k_extra = 0
    for pid in pids:
        mask = pred == pid
        if pid in assignments:
            gid,iou = assignments[pid]
            rgb[mask] = gt_palette[gid]
            log.append((pid, gid, round(iou,5), 'matched'))
        else:
            rgb[mask] = EXTRA_COLORS[k_extra % len(EXTRA_COLORS)]
            k_extra += 1
            missing_matches += 1
            log.append((pid, '', '', 'unmatched'))
    return rgb, log, missing_matches


def unit(v):
    n = np.linalg.norm(v)
    if n < 1e-10:
        raise ValueError('无法从GT构建相机方向')
    return v/n


def camera_from_gt(xyz, gt, flip=False):
    # GT实例0代表茎、>0代表叶片，不依赖PLY中任意颜色。
    stem = gt == 0
    leaf = (gt > 0) & (gt != 255)
    if stem.sum() < 20 or leaf.sum() < 20:
        raise ValueError('GT实例0(茎)或叶实例不足20点，无法构建统一视角')
    s,l = xyz[stem], xyz[leaf]
    _,vec = np.linalg.eigh(np.cov(s.T))
    up = unit(vec[:,-1])
    if np.dot(up,l.mean(0)-s.mean(0)) < 0:
        up = -up
    centered = l-l.mean(0)
    perp = centered - (centered@up)[:,None]*up
    _,vec = np.linalg.eigh(np.cov(perp.T))
    right = unit(vec[:,-1]-np.dot(vec[:,-1],up)*up)
    if right[0]<0:
        right = -right
    if flip:
        right = -right
    return right,up


# Six displayed items must share camera axes, bounds, ROI selection, and inset magnification.
# The ROI is stored in GT-derived projected 2D coordinates, not per-model pixel indices.
PANEL_ORDER = ('Input_RGB', 'GT', 'OneFormer3D', 'GAASE', 'AMSQF', 'GAMQ-SegNet')
RECT_COLOR = '#df2525'  # Solid red, both in preview and saved PNG.
INSET_BOUNDS = (0.035, 0.035, 0.44, 0.30)  # Bottom left: x, y, width, height (axes fraction).


def normalized_roi(p0, p1, xlim, ylim, min_fraction=0.008):
    x0, x1 = sorted((float(p0[0]), float(p1[0])))
    y0, y1 = sorted((float(p0[1]), float(p1[1])))
    x0, x1 = max(xlim[0], x0), min(xlim[1], x1)
    y0, y1 = max(ylim[0], y0), min(ylim[1], y1)
    if x1-x0 < (xlim[1]-xlim[0])*min_fraction or y1-y0 < (ylim[1]-ylim[0])*min_fraction:
        return None
    return (x0, x1, y0, y1)


def padded_zoom_bounds(roi, xlim, ylim):
    # Add only whitespace, never stretch geometry. The inset preserves equal XY scale.
    x0, x1, y0, y1 = roi
    display_aspect = ((xlim[1]-xlim[0]) / (ylim[1]-ylim[0]))
    ratio = display_aspect * INSET_BOUNDS[2] / INSET_BOUNDS[3]
    w = (x1-x0) * 1.10
    h = (y1-y0) * 1.10
    w = max(w, h*ratio)
    h = w/ratio
    cx, cy = (x0+x1)*0.5, (y0+y1)*0.5
    return (cx-w/2, cx+w/2), (cy-h/2, cy+h/2)


def plot_points(ax, uv, rgb, size):
    ax.scatter(uv[:,0], uv[:,1], c=rgb, s=size, marker='.', linewidths=0,
               rasterized=True)


def configure_axes(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')


def add_zoom_inset(ax, uv, rgb, roi, xlim, ylim, size):
    zxlim, zylim = padded_zoom_bounds(roi, xlim, ylim)
    inset = ax.inset_axes(INSET_BOUNDS)
    inset.set_facecolor('white')
    region = ((uv[:, 0] >= zxlim[0]) & (uv[:, 0] <= zxlim[1]) &
              (uv[:, 1] >= zylim[0]) & (uv[:, 1] <= zylim[1]))
    if region.any():
        plot_points(inset, uv[region], rgb[region], size*1.45)
    inset.set_xlim(*zxlim)
    inset.set_ylim(*zylim)
    inset.set_aspect('equal', adjustable='box')
    inset.axis('off')
    # Axis off hides spines; draw a separate solid border in axes fraction coordinates.
    inset.add_patch(Rectangle((0, 0), 1, 1, transform=inset.transAxes,
                              edgecolor=RECT_COLOR, facecolor='none', linewidth=2.2,
                              clip_on=False, zorder=20))
    return inset


def save_png(path, uv, rgb, xlim, ylim, width, height, size, roi=None):
    # Explicit Agg figure: no extra popup windows when saving from the GUI.
    dpi = 150
    fig = Figure(figsize=(width/dpi, height/dpi), dpi=dpi, facecolor='white')
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0,0,1,1], facecolor='white')
    configure_axes(ax, xlim, ylim)
    plot_points(ax, uv, rgb, size)
    if roi is not None:
        x0,x1,y0,y1 = roi
        ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0,
                               fill=False, edgecolor=RECT_COLOR,
                               linewidth=2.2, linestyle='solid', zorder=21))
        add_zoom_inset(ax, uv, rgb, roi, xlim, ylim, size)
    fig.savefig(path, dpi=dpi, facecolor='white', pad_inches=0)
    fig.clear()


class SynchronizedROIPreview:
    """Draw ONE ROI on ANY main panel, simultaneously show on ALL panels.

    Mouse press/drag/release: all six red boxes update while dragging.
    On release, all six bottom-left insets show the same projected ROI.
    No files are saved until the user clicks Confirm & Save.
    """
    def __init__(self, sid, projections, xlim, ylim, size, preview_raw_limit=40000):
        self.sid = sid
        self.data = projections
        self.names = [n for n in PANEL_ORDER if n in projections]
        self.xlim, self.ylim = xlim, ylim
        self.size = size
        self.preview_raw_limit = preview_raw_limit
        self.fig, ax_grid = plt.subplots(2, 3, figsize=(16.5, 10.2), facecolor='white')
        self.fig.subplots_adjust(left=0.015, right=0.985, bottom=0.135, top=0.925,
                                 wspace=0.045, hspace=0.11)
        self.fig.suptitle(f'{sid} | Shared ROI / synchronized zoom preview',
                          fontsize=15, fontweight='bold')
        self.axes, self.patches, self.insets = [], [], []
        self.dragging = False
        self.source_ax = None
        self.start_point = None
        self.current_roi = None
        self.result = None
        self.status = self.fig.text(0.5, 0.105,
            'Drag a rectangle in ANY main panel. The same red ROI appears on ALL panels.',
            ha='center', va='center', fontsize=11)
        for idx, (name, ax) in enumerate(zip(self.names, ax_grid.flat)):
            uv,rgb=self.data[name]
            if name=='Input_RGB' and self.preview_raw_limit and len(uv)>self.preview_raw_limit:
                rng=np.random.default_rng(19)
                take=np.sort(rng.choice(len(uv),self.preview_raw_limit,replace=False))
                pv,pr=uv[take],rgb[take]
            else:
                pv,pr=uv,rgb
            configure_axes(ax, xlim, ylim)
            plot_points(ax, pv, pr, max(size, 2.0))
            ax.set_title(name, fontweight='bold', fontsize=12, pad=3)
            rect=Rectangle((0,0), 0,0, fill=False, edgecolor=RECT_COLOR,
                           linewidth=1.6, linestyle='solid', visible=False, zorder=21)
            ax.add_patch(rect)
            self.axes.append(ax)
            self.patches.append(rect)
            self.insets.append(None)
        for ax in list(ax_grid.flat)[len(self.names):]:
            ax.axis('off')
        button_specs = [
            (0.155, 'Confirm & Save', self._confirm, '#dcefdc'),
            (0.37, 'Redraw ROI', self._reset, '#f6e4db'),
            (0.57, 'Skip Sample', self._skip, '#eeeeee'),
            (0.77, 'Quit Batch', self._quit, '#eeeeee'),
        ]
        self.buttons=[]
        for cx, text, cb, color in button_specs:
            b_ax = self.fig.add_axes([cx-0.091, 0.020, 0.182, 0.050])
            btn = Button(b_ax, text, color=color, hovercolor='#c8def5')
            btn.on_clicked(cb)
            self.buttons.append(btn)   # Keep strong refs or callbacks stop working.
        canvas=self.fig.canvas
        canvas.mpl_connect('button_press_event',self._on_press)
        canvas.mpl_connect('motion_notify_event',self._on_move)
        canvas.mpl_connect('button_release_event',self._on_release)
        canvas.mpl_connect('close_event',self._on_close)

    def _mouse_xy(self, event):
        # Convert physical cursor pixels using the AXIS ON WHICH DRAG STARTED.
        # This also works when the cursor leaves the originating panel.
        if self.source_ax is None:
            return None
        x,y=self.source_ax.transData.inverted().transform((event.x,event.y))
        return (float(np.clip(x,*self.xlim)), float(np.clip(y,*self.ylim)))

    def _remove_insets(self):
        for i, ax in enumerate(self.insets):
            if ax is not None:
                ax.remove()
                self.insets[i]=None

    def _update_boxes(self, roi):
        if roi is None:
            for patch in self.patches:
                patch.set_visible(False)
        else:
            x0,x1,y0,y1=roi
            for patch in self.patches:
                patch.set_xy((x0,y0))
                patch.set_width(x1-x0)
                patch.set_height(y1-y0)
                patch.set_visible(True)
        self.fig.canvas.draw_idle()

    def _on_press(self, event):
        if event.button != 1 or event.inaxes not in self.axes or self.result is not None:
            return
        self._remove_insets()
        self.current_roi = None
        self.source_ax = event.inaxes
        self.start_point = self._mouse_xy(event)
        self.dragging = True
        self.status.set_text('Dragging: one shared ROI is applied to every panel...')
        self._update_boxes(None)

    def _on_move(self, event):
        if not self.dragging or self.start_point is None:
            return
        current=self._mouse_xy(event)
        x0,x1=sorted((self.start_point[0],current[0]))
        y0,y1=sorted((self.start_point[1],current[1]))
        self._update_boxes((x0,x1,y0,y1))

    def _on_release(self, event):
        if not self.dragging:
            return
        end=self._mouse_xy(event)
        self.dragging=False
        self.source_ax=None
        roi=normalized_roi(self.start_point,end,self.xlim,self.ylim)
        self.start_point=None
        if roi is None:
            self.status.set_text('ROI is too small. Drag again (or press Redraw ROI).')
            self._update_boxes(None)
            return
        self.current_roi=roi
        self._update_boxes(roi)
        for i,name in enumerate(self.names):
            uv,rgb=self.data[name]
            # Inset shows all points (including high-density raw points), even if
            # the full plant preview was downsampled for smoother interaction.
            self.insets[i]=add_zoom_inset(self.axes[i],uv,rgb,roi,
                                            self.xlim,self.ylim,self.size)
        self.status.set_text('ROI synchronized. Zooms shown at bottom-left. Confirm & Save, or redraw.')
        self.fig.canvas.draw_idle()

    def _confirm(self, _event):
        if self.current_roi is None:
            self.status.set_text('Please select a ROI before saving.')
            self.fig.canvas.draw_idle()
            return
        self.result='save'
        plt.close(self.fig)

    def _reset(self, _event):
        self.dragging=False
        self.source_ax=None
        self.start_point=None
        self.current_roi=None
        self._remove_insets()
        self._update_boxes(None)
        self.status.set_text('ROI cleared. Draw a new red box on ANY main panel.')
        self.fig.canvas.draw_idle()

    def _skip(self, _event):
        self.result='skip'
        plt.close(self.fig)

    def _quit(self, _event):
        self.result='quit'
        plt.close(self.fig)

    def _on_close(self, _event):
        if self.result is None:
            self.result='quit'  # Clicking the window X stops the batch safely.

    def show(self):
        plt.show(block=True)
        return self.result, self.current_roi


def render_one(sid,args):
    dirs = {
        'OneFormer3D': (args.base_dir, 'OneFormer3D'),
        'GAASE': (args.gaase_dir, 'GAASE'),
        'AMSQF': (args.amsqf_dir, 'AMSQF'),
        'GAMQ-SegNet': (args.full_dir,'GAMQ-SegNet'),
    }
    gt_ply = ply_file(args.base_dir,'OneFormer3D',sid,'gt_inst')
    gt_npy = label_file(args.base_dir,'OneFormer3D',sid,'gt_inst')
    needed = [gt_ply,gt_npy]
    for folder,prefix in dirs.values():
        needed.extend([ply_file(folder,prefix,sid,'pred_inst'), label_file(folder,prefix,sid,'pred_inst')])
    absent = [str(p) for p in needed if not p.is_file()]
    if absent:
        raise FileNotFoundError('实例分割文件缺失，检查是否已经按前缀重命名：\n'+'\n'.join(absent))

    ref_xyz,_ = read_ply(gt_ply)
    gt = read_ids(gt_npy,len(ref_xyz))
    if len(gt) == 0:
        raise ValueError('GT空点云: '+sid)
    right,up = camera_from_gt(ref_xyz,gt,args.flip_horizontal)
    raw_path = raw_file(args.raw_root,sid)
    if raw_path is None and not args.allow_missing_raw:
        raise FileNotFoundError('原始RGB点云未找到: '+sid+'，可使用--allow-missing-raw跳过原图')
    raw = None
    if raw_path is not None:
        xyz,rgb=read_raw(raw_path)
        if rgb is None:
            raise ValueError('原始PLY未含RGB颜色：'+str(raw_path))
        if args.max_raw_points and len(xyz)>args.max_raw_points:
            rng=np.random.default_rng(42)
            indices=np.sort(rng.choice(len(xyz),args.max_raw_points,replace=False))
            xyz,rgb=xyz[indices],rgb[indices]
        raw=(xyz,rgb)

    rgb_gt,palette=base_colors(gt)
    outputs={'GT':(ref_xyz,rgb_gt)}
    logs=[]
    for model,(folder,prefix) in dirs.items():
        p=ply_file(folder,prefix,sid,'pred_inst')
        xyz,_=read_ply(p)
        if xyz.shape!=ref_xyz.shape:
            raise ValueError(f'{sid} {model} XYZ点数错误: {xyz.shape} vs {ref_xyz.shape}')
        max_error=float(np.max(np.abs(xyz-ref_xyz)))
        if max_error>args.xyz_tolerance:
            raise ValueError(f'{sid} {model} XYZ不一致(最大坐标差={max_error:.6g})，不进行按序实例颜色匹配')
        ids=read_ids(label_file(folder,prefix,sid,'pred_inst'),len(xyz))
        rgb, mapping, unused=matched_pred_colors(ids,gt,palette,args.min_match_iou)
        outputs[model]=(xyz,rgb)
        for pid,gid,iou,status in mapping:
            logs.append([sid,model,pid,gid,iou,status])
        print(f'   {model:<13} 点数={len(xyz):6d}, 预测实例={len(np.unique(ids)):2d}, GT实例={len(np.unique(gt)):2d}, 未匹配实例={unused}, XYZ误差={max_error:.4g}')

    out_dir=args.out_dir/sid
    proj_ref=np.column_stack((ref_xyz@right,ref_xyz@up))
    low,high=proj_ref.min(axis=0),proj_ref.max(axis=0)
    # 坐标范围只使用GT，避免不同模型或原始高密度点云独立缩放。
    span=np.maximum(high-low,1e-9)
    if args.width <= 2*args.margin or args.height <= 2*args.margin:
        raise ValueError('画布尺寸需大于两倍margin')
    scale=min((args.width-2*args.margin)/span[0],(args.height-2*args.margin)/span[1])
    center=(high+low)*0.5
    xlim=(center[0]-args.width/(2*scale),center[0]+args.width/(2*scale))
    ylim=(center[1]-args.height/(2*scale),center[1]+args.height/(2*scale))
    if raw is not None:
        rx,rr=raw
        projected=np.column_stack((rx@right,rx@up))
        raw_span=np.ptp(projected,axis=0)
        if np.any(np.abs(raw_span-span)>np.maximum(span,1e-9)*0.15):
            print('   [RAW WARNING] 原始点云与GT投影尺寸差异超过15%；原图使用同视角尺度，请人工检查裁剪和坐标系')
        if np.any((projected.min(0)>high) | (projected.max(0)<low)):
            print('   [RAW WARNING] 原始点云与GT位置可能不重合，请检查原图')
        outputs={'Input_RGB': raw, **outputs}

    # Project ONCE using the GT-defined camera; these arrays back every panel.
    projected_outputs={}
    for name,(xyz,rgb) in outputs.items():
        uv=np.column_stack((xyz@right,xyz@up))
        projected_outputs[name]=(uv,rgb)

    roi=None
    if not args.no_preview:
        # Preview comes first: creating PNGs is strictly after Confirm & Save.
        preview=SynchronizedROIPreview(sid,projected_outputs,xlim,ylim,
                                      args.point_size,args.preview_raw_points)
        action,roi=preview.show()
        if action=='quit':
            print(f'   [QUIT] 用户关闭预览，批处理停止；{sid}未保存。')
            return -1
        if action=='skip':
            print(f'   [SKIP] 用户跳过样本：{sid}，未保存。')
            return 0
        if action!='save' or roi is None:
            raise RuntimeError('ROI预览没有返回有效保存决定')

    out_dir.mkdir(parents=True,exist_ok=True)
    # Keep the exact naming convention of the source script.
    for name,(uv,rgb) in projected_outputs.items():
        if name=='Input_RGB':
            png_name=f'Input_RGB_{sid}.png'
        elif name=='GT':
            png_name=f'GT_{sid}_gt_inst.png'
        else:
            png_name=f'{name}_{sid}_pred_inst.png'
        out_path=out_dir/png_name
        save_png(out_path,uv,rgb,xlim,ylim,args.width,args.height,args.point_size,roi)
        print('   [PNG]',out_path.name)

    if roi is not None:
        metadata={
            'sample_id':sid,
            'shared_roi_xy':[float(v) for v in roi],
            'full_view_x_limits':[float(v) for v in xlim],
            'full_view_y_limits':[float(v) for v in ylim],
            'note':'Projected GT XY ROI; same physical ROI, camera orientation, scale and inset for every model.',
        }
        with (out_dir/'shared_roi.json').open('w',encoding='utf-8') as f:
            json.dump(metadata,f,ensure_ascii=False,indent=2)

    with (out_dir/'instance_color_matching.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f)
        writer.writerow(['sample_id','model','pred_inst_id','matched_gt_inst_id','IoU','status'])
        writer.writerows(logs)
    return len(outputs)


def main():
    ap=argparse.ArgumentParser(description='11株玉米：实例GT+4消融配置+原始RGB，统一相机PNG')
    ap.add_argument('--raw-root',type=Path,required=True)
    ap.add_argument('--base-dir',type=Path,required=True)
    ap.add_argument('--gaase-dir',type=Path,required=True)
    ap.add_argument('--amsqf-dir',type=Path,required=True)
    ap.add_argument('--full-dir',type=Path,required=True)
    ap.add_argument('--out-dir',type=Path,required=True)
    ap.add_argument('--samples',nargs='+',default=SAMPLES)
    ap.add_argument('--width',type=int,default=900)
    ap.add_argument('--height',type=int,default=1200)
    ap.add_argument('--margin',type=int,default=55)
    ap.add_argument('--point-size',type=float,default=3.0)
    ap.add_argument('--max-raw-points',type=int,default=250000,help='原始RGB点云最大显示点数；0=不限')
    ap.add_argument('--min-match-iou',type=float,default=0.05)
    ap.add_argument('--xyz-tolerance',type=float,default=1e-4)
    ap.add_argument('--allow-missing-raw',action='store_true')
    ap.add_argument('--flip-horizontal',action='store_true')
    ap.add_argument('--no-preview',action='store_true',help='兼容原脚本：不打开GUI、不画ROI，直接按原方式保存')
    ap.add_argument('--preview-raw-points',type=int,default=40000,
                    help='GUI中原始RGB的预览点数上限（仅预览下采样，放大及保存用原始读取点数）')
    args=ap.parse_args()
    for d in (args.raw_root,args.base_dir,args.gaase_dir,args.amsqf_dir,args.full_dir):
        if not d.is_dir():
            ap.error('目录不存在: '+str(d))
    total=0
    for idx,sid in enumerate(args.samples,1):
        print(f'\n[{idx}/{len(args.samples)}] {sid}')
        produced=render_one(sid,args)
        if produced < 0:
            break
        total+=produced
    print(f'\n[DONE] {len(args.samples)}株，共{total}张实例对比PNG，输出目录：{args.out_dir}')
    print('[NOTE] predicted实例颜色按GT重映射，仅用于视觉对比，不改变预测标签')

if __name__=='__main__':
    main()

