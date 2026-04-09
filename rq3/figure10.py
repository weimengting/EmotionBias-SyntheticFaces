import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
from matplotlib.colorbar import ColorbarBase

sections = {
    'Gender': {
        'rows': ['Male', 'Female', 'Diff.'],
        'vals': [
            [63.4, 40.8, 48.9, 17.0, 21.4, 19.5],
            [51.5, 22.2, 32.9, 0.5, 5.2, 18.4],
            [11.9, 18.6, 16.0, 16.5, 16.2, 1.1],
        ],
        'is_diff': [False, False, True],
    },
    'Race': {
        'rows': ['White', 'non-White', 'Diff.'],
        'vals': [
            [64.1, 42.7, 48.0, 7.1, 21.5, 22.9],
            [53.1, 33.0, 45.2, 12.9, 20.3, 6.1],
            [11.0, 9.7, 2.8, -5.8, 1.2, 16.8],
        ],
        'is_diff': [False, False, True],
    },
    'Age': {
        'rows': ['Young', 'Middle-aged', 'Old'],
        'vals': [
            [55.9, 33.1, 41.9, 11.0, 13.8, 18.3],
            [62.5, 42.6, 54.8, 20.1, 30.9, 54.3],
            [47.3, 37.4, 29.5, -6.2, 27.1, np.nan],
        ],
        'is_diff': [False, False, False],
    },
}

pos_cmap = mcolors.LinearSegmentedColormap.from_list(
    'pos', ['#f5d0d8', '#e8889a', '#c94f72', '#9b2550', '#6b0f30']
)
neg_color = '#e8f3e0'
neg_text = '#4a7a30'
diff_cmap = mcolors.LinearSegmentedColormap.from_list(
    'diff', ['#e8e8e4', '#c8c6c0', '#a0a09a', '#707068']
)

all_main = []
for sec in sections.values():
    for vals, is_diff in zip(sec['vals'], sec['is_diff']):
        if not is_diff:
            all_main.extend([v for v in vals if not np.isnan(v) and v >= 0])
vmin_pos, vmax_pos = 0.0, max(all_main)

all_diff_pos = []
for sec in sections.values():
    for vals, is_diff in zip(sec['vals'], sec['is_diff']):
        if is_diff:
            all_diff_pos.extend([v for v in vals if v >= 0])
dmin, dmax = 0.0, max(all_diff_pos)

# =========================
# 尺寸
# =========================
row_h = 0.48
sep_h = 0.18
n_cols = 6
label_w_inch = 0.9  # 行标签占用的英寸宽度

total_rows = sum(len(s['rows']) for s in sections.values())
n_secs = len(sections)
fig_h = total_rows * row_h + (n_secs - 1) * sep_h + 0.1
fig_w = 7 + label_w_inch

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.set_axis_off()
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# 归一化坐标中标签宽度
label_w_norm = label_w_inch / fig_w
cell_w = (1.0 - label_w_norm) / n_cols

y_cursor = 1.0 - 0.01

sep_after_rows = [2, 5]  # 第2行、第5行之后画分隔线（全局行索引）
global_row = 0

for si, (dim, sec) in enumerate(sections.items()):
    n_rows = len(sec['rows'])

    for ri, (row_label, vals, is_diff) in enumerate(
            zip(sec['rows'], sec['vals'], sec['is_diff'])):

        y_top = y_cursor - ri * (row_h / fig_h)
        y_bot = y_top - row_h / fig_h
        y_mid = (y_top + y_bot) / 2
        cell_h = row_h / fig_h

        # 行标签
        label_style = dict(fontstyle='italic', color='#999990') if is_diff \
            else dict(fontstyle='normal', color='#5a5a58')
        ax.text(label_w_norm - 0.01, y_mid,
                row_label, ha='right', va='center',
                fontsize=9, transform=ax.transAxes,
                **label_style)

        for j, v in enumerate(vals):
            x_left = label_w_norm + j * cell_w
            x_mid = x_left + cell_w * 0.5
            pad = 0.004

            if np.isnan(v):
                rect = FancyBboxPatch(
                    (x_left + pad, y_bot + pad),
                    cell_w - 2 * pad, cell_h - 2 * pad,
                    boxstyle="round,pad=0.003",
                    transform=ax.transAxes,
                    facecolor='#f0ede8', edgecolor='none', zorder=2
                )
                ax.add_patch(rect)
                ax.text(x_mid, y_mid, '\u2013',
                        ha='center', va='center', fontsize=9,
                        color='#b0ada8', transform=ax.transAxes, zorder=3)
                continue

            if is_diff:
                if v < 0:
                    facecolor = neg_color
                    textcolor = neg_text
                else:
                    t = (v - dmin) / (dmax - dmin) if dmax > dmin else 0.5
                    facecolor = diff_cmap(t)
                    textcolor = '#3a3a38'
            else:
                if v < 0:
                    facecolor = neg_color
                    textcolor = neg_text
                else:
                    t = (v - vmin_pos) / (vmax_pos - vmin_pos)
                    facecolor = pos_cmap(t)
                    lum = (0.299 * facecolor[0] +
                           0.587 * facecolor[1] +
                           0.114 * facecolor[2])
                    textcolor = '#fdf0f3' if lum < 0.55 else '#6b0f30'

            rect = FancyBboxPatch(
                (x_left + pad, y_bot + pad),
                cell_w - 2 * pad, cell_h - 2 * pad,
                boxstyle="round,pad=0.003",
                transform=ax.transAxes,
                facecolor=facecolor, edgecolor='none', zorder=2
            )
            ax.add_patch(rect)

            sign = '+' if v >= 0 else ''
            ax.text(x_mid, y_mid, f'{sign}{v:.1f}',
                    ha='center', va='center', fontsize=8.5,
                    color=textcolor, fontweight='bold',
                    transform=ax.transAxes, zorder=3)

    y_cursor -= n_rows * (row_h / fig_h)

    if si < n_secs - 1:
        ax.plot([0, 1], [y_cursor, y_cursor],
                color='#d3d1c7', linewidth=0.7,
                transform=ax.transAxes, zorder=1)
        y_cursor -= sep_h / fig_h

plt.tight_layout(pad=0)
plt.savefig("delta_p_heatmap.png", dpi=300, bbox_inches='tight')
plt.show()

# =========================
# 色条单独保存
# =========================
fig2, cbar_ax = plt.subplots(figsize=(4, 0.45))
fig2.patch.set_facecolor('white')
norm = mcolors.Normalize(vmin=vmin_pos, vmax=vmax_pos)
cb = ColorbarBase(cbar_ax, cmap=pos_cmap, norm=norm, orientation='horizontal')
cb.set_label(r'$\Delta P_{\mathrm{low{-}att}}$ (%)', fontsize=9, color='#5f5e5a')
cb.ax.tick_params(labelsize=8, colors='#5f5e5a')
cb.outline.set_visible(False)
plt.tight_layout()
plt.savefig("delta_p_colorbar.png", dpi=300, bbox_inches='tight')
plt.show()

if __name__ == '__main__':
    print("done")