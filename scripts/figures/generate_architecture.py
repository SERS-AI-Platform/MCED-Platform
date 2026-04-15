"""
uSERS-Net Architecture Diagram — Professional v4
Organ icons + 2-tone per branch + compact layout
Author: SOLUM Healthcare
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path as MPath
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path


# ── Helpers ──────────────────────────────────────────────────

def _darken(hx, f):
    hx = hx.lstrip('#')
    r,g,b = int(hx[:2],16), int(hx[2:4],16), int(hx[4:6],16)
    return f'#{max(0,min(255,int(r*f))):02x}{max(0,min(255,int(g*f))):02x}{max(0,min(255,int(b*f))):02x}'

def _lighten(hx, f):
    hx = hx.lstrip('#')
    r,g,b = int(hx[:2],16), int(hx[2:4],16), int(hx[4:6],16)
    r = int(r + (255-r)*f)
    g = int(g + (255-g)*f)
    b = int(b + (255-b)*f)
    return f'#{min(255,r):02x}{min(255,g):02x}{min(255,b):02x}'


def block(ax, x, y, w, h, d=0.10, color='#4FC3F7', label='',
          sub='', sup='', fs=6, alpha=0.92):
    """3D block with self-colored edges (no harsh black outline)."""
    ec = _darken(color, 0.55)
    ax.add_patch(plt.Polygon([(x,y),(x+w,y),(x+w,y+h),(x,y+h)],
        closed=True, fc=color, ec=ec, alpha=alpha, lw=0.5, zorder=3))
    ax.add_patch(plt.Polygon([(x,y+h),(x+w,y+h),(x+w+d,y+h+d),(x+d,y+h+d)],
        closed=True, fc=_darken(color,0.82), ec=ec, alpha=alpha, lw=0.5, zorder=3))
    ax.add_patch(plt.Polygon([(x+w,y),(x+w+d,y+d),(x+w+d,y+h+d),(x+w,y+h)],
        closed=True, fc=_darken(color,0.65), ec=ec, alpha=alpha, lw=0.5, zorder=3))
    if label:
        ax.text(x+w/2, y+h/2, label, ha='center', va='center',
                fontsize=fs, fontweight='bold', color='white', zorder=5)
    if sub:
        ax.text(x+w/2, y-0.08, sub, ha='center', va='top',
                fontsize=fs-1.5, color='#777', zorder=5)
    if sup:
        ax.text(x+w/2+d/2, y+h+d+0.04, sup, ha='center', va='bottom',
                fontsize=fs-1.5, color='#777', zorder=5)
    return x + w + d


def arr(ax, x1, y1, x2, y2, c='#BBBBBB', lw=0.9):
    ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle='->', color=c, lw=lw,
                                mutation_scale=8), zorder=2)

def harr(ax, x1, x2, y, c='#BBBBBB', lw=0.9):
    arr(ax, x1, y, x2, y, c=c, lw=lw)


# ── Organ Icon Drawers ───────────────────────────────────────

def _draw_organ(ax, cx, cy, size, organ_type, color):
    """Draw simplified organ silhouette at (cx, cy)."""
    s = size
    ec = _darken(color, 0.55)

    if organ_type == 'prostate':
        # walnut / butterfly shape
        t = np.linspace(0, 2*np.pi, 80)
        rx, ry = s*0.40, s*0.30
        xs = cx + rx * np.cos(t) * (1 + 0.2*np.cos(2*t))
        ys = cy + ry * np.sin(t)
        ax.fill(xs, ys, fc=color, ec=ec, lw=0.7, zorder=4)
        # urethra line
        ax.plot([cx, cx], [cy-ry*0.9, cy+ry*0.9], color=ec, lw=0.5, zorder=5, alpha=0.5)

    elif organ_type == 'breast':
        # breast profile — rounded dome shape
        t = np.linspace(0, 2*np.pi, 60)
        rx, ry = s*0.32, s*0.35
        xs = cx + rx*np.cos(t)
        ys = cy + ry*np.sin(t) * (1 + 0.25*np.maximum(0, -np.cos(t)))
        ax.fill(xs, ys, fc=color, ec=ec, lw=0.7, zorder=4)
        # nipple hint
        ax.plot(cx+rx*0.75, cy, 'o', color=_darken(color, 0.7),
                markersize=1.5, zorder=5)

    elif organ_type == 'ovarian':
        # two ovaries connected by fallopian tubes to uterus
        for sign in [-1, 1]:
            t = np.linspace(0, 2*np.pi, 40)
            ox = cx + sign*s*0.28
            ax.fill(ox + s*0.14*np.cos(t), cy + s*0.18*np.sin(t),
                    fc=color, ec=ec, lw=0.7, zorder=4)
            # tube
            tx = np.linspace(cx + sign*s*0.05, ox - sign*s*0.14, 12)
            ty = cy + sign*s*0.08*np.sin(np.linspace(0, np.pi, 12))
            ax.plot(tx, ty, color=ec, lw=0.7, zorder=4)
        # uterus hint (small triangle)
        ax.fill([cx-s*0.06, cx+s*0.06, cx],
                [cy+s*0.06, cy+s*0.06, cy-s*0.12],
                fc=_lighten(color, 0.2), ec=ec, lw=0.5, zorder=4)

    elif organ_type == 'lung':
        # two lobes with bronchi
        for sign in [-1, 1]:
            t = np.linspace(0, 2*np.pi, 60)
            sc = 0.9 if sign == -1 else 1.0  # left lung slightly smaller
            lx = cx + sign*s*0.20 + s*0.20*sc*np.cos(t)
            ly = cy + s*0.30*sc*np.sin(t) * (1.0 + 0.15*np.cos(t))
            ax.fill(lx, ly, fc=color, ec=ec, lw=0.7, zorder=4)
        # trachea + bronchi
        ax.plot([cx, cx], [cy+s*0.32, cy+s*0.48], color=ec, lw=1.0, zorder=5)
        ax.plot([cx, cx-s*0.12], [cy+s*0.32, cy+s*0.15], color=ec, lw=0.7, zorder=5)
        ax.plot([cx, cx+s*0.12], [cy+s*0.32, cy+s*0.15], color=ec, lw=0.7, zorder=5)

    elif organ_type == 'colon':
        # inverted U with ascending/descending
        from matplotlib.patches import FancyArrowPatch
        t = np.linspace(-0.5, np.pi+0.5, 50)
        xs = cx + s*0.35*np.cos(t)
        ys = cy + s*0.28*np.sin(t) + s*0.02
        ax.plot(xs, ys, color=color, lw=s*10, solid_capstyle='round', zorder=4)
        ax.plot(xs, ys, color=ec, lw=s*10+1.2, solid_capstyle='round', zorder=3)

    elif organ_type == 'pancreas':
        # tadpole shape: thick head on right, thin tail on left
        t = np.linspace(-1, 1, 60)
        xs = cx + s*0.42*t
        curve = s*0.04*np.sin(t*2)
        widths = s*0.14 * np.exp(-1.2*(t+0.2)**2) + s*0.04
        upper = cy + curve + widths
        lower = cy + curve - widths
        ax.fill(np.concatenate([xs, xs[::-1]]),
                np.concatenate([upper, lower[::-1]]),
                fc=color, ec=ec, lw=0.7, zorder=4)

    elif organ_type == 'bladder':
        # balloon/pear shape
        t = np.linspace(0, 2*np.pi, 60)
        rx = s*0.30
        ry = s*0.35
        squeeze = 1 - 0.2*np.maximum(0, np.cos(t))
        xs = cx + rx*np.cos(t)*squeeze
        ys = cy + ry*np.sin(t)
        ax.fill(xs, ys, fc=color, ec=ec, lw=0.7, zorder=4)

    elif organ_type == 'healthy':
        # shield with checkmark
        # shield outline
        sx = [cx-s*0.28, cx-s*0.28, cx-s*0.15, cx, cx+s*0.15, cx+s*0.28, cx+s*0.28]
        sy = [cy+s*0.15, cy-s*0.08, cy-s*0.30, cy-s*0.38, cy-s*0.30, cy-s*0.08, cy+s*0.15]
        ax.fill(sx, sy, fc=color, ec=ec, lw=0.7, zorder=4)
        # top arc
        t = np.linspace(0, np.pi, 30)
        ax.fill(cx+s*0.28*np.cos(t), cy+s*0.15+s*0.12*np.sin(t),
                fc=color, ec=ec, lw=0.7, zorder=4)
        # checkmark
        ax.plot([cx-s*0.10, cx-s*0.02, cx+s*0.14],
                [cy-s*0.02, cy-s*0.15, cy+s*0.12],
                color='white', lw=1.8, solid_capstyle='round', zorder=5)


# ── Colors — 2-tone per branch ───────────────────────────────
# LR branch: purple tones
LR1 = '#9575CD'   # light purple
LR2 = '#5E35B1'   # deep purple

# ResNet branch: warm→cool gradient (like reference image)
RN = ['#EF5350', '#FF7043', '#FFA726', '#FFCA28', '#66BB6A', '#42A5F5', '#5C6BC0']
#      stem      L1         L2         L3         L4         GAP       heads

# Stage / output
ST1 = '#E91E63'
ST2 = '#00897B'
BLD = '#FF8A65'

# Organ colors (softer, medical feel)
ORG = {
    'prostate': '#5C6BC0',
    'breast':   '#F06292',
    'ovarian':  '#AB47BC',
    'lung':     '#42A5F5',
    'colon':    '#66BB6A',
    'pancreas': '#FFA726',
    'bladder':  '#26C6DA',
    'healthy':  '#81C784',
}


# ── Main ─────────────────────────────────────────────────────

def draw():
    fig, ax = plt.subplots(figsize=(20, 5.5))
    ax.set_xlim(-0.1, 19)
    ax.set_ylim(-1.5, 4.8)
    ax.set_aspect('equal')
    ax.axis('off')

    g = 0.18   # arrow gap
    sg = 0.05  # small gap in resnet

    # ════════════════════════════════════════════════════════
    # INPUT — SERS (grey) + Clinical (teal) stacked
    # ════════════════════════════════════════════════════════
    inp_x = 0
    sers_h = 2.6
    clin_h = 0.5
    inp_y = 0.0
    block(ax, inp_x, inp_y+clin_h, 0.45, sers_h, d=0.08, color='#546E7A',
          label='SERS\n1800', fs=6.5)
    block(ax, inp_x, inp_y, 0.45, clin_h, d=0.08, color='#26A69A',
          label='Clinical', fs=5)

    fork_x = 0.53

    # ════════════════════════════════════════════════════════
    # BRANCH 1 — Fusion LR (top, y≈3.0)
    # ════════════════════════════════════════════════════════
    b1y = 2.8
    b1h = 0.85

    arr(ax, fork_x, inp_y+clin_h+sers_h*0.72, 0.72, b1y, c='#9E9E9E')

    # Concat
    r = block(ax, 0.75, b1y-b1h/2, 0.55, b1h, d=0.07, color=LR1,
              label='Concat\n936', fs=5.5)
    harr(ax, r, r+g, b1y)

    # Scaler
    r = block(ax, r+g, b1y-b1h/2+0.05, 0.6, b1h-0.1, d=0.07, color=LR1,
              label='Standard\nScaler', fs=5)
    harr(ax, r, r+g, b1y)

    # LR
    r_lr = block(ax, r+g, b1y-b1h/2, 0.9, b1h, d=0.08, color=LR2,
                 label='Logistic\nRegression', fs=5.5)

    # ════════════════════════════════════════════════════════
    # BRANCH 2 — ResNet18-1D (bottom, y≈0.8)
    # ════════════════════════════════════════════════════════
    b2y = 0.8
    bh = 1.0

    arr(ax, fork_x, inp_y+clin_h+sers_h*0.25, 0.72, b2y, c='#9E9E9E')

    # Stem Conv
    h = bh + 0.3
    r = block(ax, 0.75, b2y-h/2, 0.28, h, d=0.05, color=RN[0],
              label='Conv1d', fs=4)
    harr(ax, r, r+sg, b2y, lw=0.6)

    # BN+MP
    h = bh + 0.15
    r = block(ax, r+sg, b2y-h/2, 0.20, h, d=0.04, color=RN[5],
              label='BN\nMP', fs=4)
    harr(ax, r, r+sg, b2y, lw=0.6)

    # Layers 1-4: single block + ×2 label
    layers = [
        (bh,       0.40, 0.06, RN[1], 'Layer1\n32'),
        (bh+0.15,  0.48, 0.08, RN[2], 'Layer2\n64'),
        (bh+0.30,  0.55, 0.10, RN[3], 'Layer3\n128'),
        (bh+0.50,  0.62, 0.12, RN[4], 'Layer4\n256'),
    ]
    for h, w, d, c, lbl in layers:
        r = block(ax, r+sg, b2y-h/2, w, h, d=d, color=c, label=lbl, fs=4.5)
        # ×2 label
        ax.text(r-w/2-d/2, b2y-h/2-0.10, '×2', ha='center', va='top',
                fontsize=4, color=_darken(c, 0.5), fontweight='bold')
        harr(ax, r, r+sg, b2y, lw=0.6)

    # GAP
    h = bh - 0.2
    r = block(ax, r+sg, b2y-h/2, 0.28, h, d=0.05, color=RN[5],
              label='GAP', fs=5)
    gap_r = r

    # Split → heads
    harr(ax, gap_r, gap_r+0.12, b2y, lw=0.6)

    # Binary head
    hd1_y = b2y + 0.45
    arr(ax, gap_r+0.12, b2y, gap_r+0.20, hd1_y, c='#BBBBBB', lw=0.7)
    r_bin = block(ax, gap_r+0.20, hd1_y-0.28, 0.70, 0.56, d=0.06,
                  color=RN[6], label='Binary\nHead', fs=5)

    # Type head
    hd2_y = b2y - 0.45
    arr(ax, gap_r+0.12, b2y, gap_r+0.20, hd2_y, c='#BBBBBB', lw=0.7)
    r_typ = block(ax, gap_r+0.20, hd2_y-0.28, 0.70, 0.56, d=0.06,
                  color=RN[6], label='Type\nHead', fs=5)

    # ════════════════════════════════════════════════════════
    # BLEND → STAGE 1 → STAGE 2 → OUTPUTS
    # ════════════════════════════════════════════════════════
    mid_y = (b1y + hd1_y) / 2

    blend_x = max(r_lr, r_bin) + 0.4

    # LR → Blend
    arr(ax, r_lr, b1y, blend_x, mid_y+0.2, c=_darken(LR2, 0.8), lw=1.1)
    # Binary Head → Blend
    arr(ax, r_bin, hd1_y, blend_x, mid_y-0.2, c=_darken(RN[6], 0.8), lw=1.1)

    # Blend
    bl_h = 0.9
    r_bl = block(ax, blend_x, mid_y-bl_h/2, 0.65, bl_h, d=0.08,
                 color=BLD, label='Blend', fs=6)
    harr(ax, r_bl, r_bl+g, mid_y, lw=1.0)

    # Stage 1
    s1_h = 0.85
    r_s1 = block(ax, r_bl+g, mid_y-s1_h/2, 0.80, s1_h, d=0.08,
                 color=ST1, label='Stage 1\nCancer\nvs Normal', fs=5)

    # ── Non-Cancer output ──
    nc_x = r_s1 + 0.45
    nc_y = mid_y + 0.75
    arr(ax, r_s1, mid_y+0.15, nc_x+0.3, nc_y, c='#81C784', lw=1.0)
    _draw_organ(ax, nc_x+0.3, nc_y, 0.55, 'healthy', ORG['healthy'])
    ax.text(nc_x+0.3, nc_y-0.35, 'Non-Cancer', ha='center', fontsize=5,
            color=_darken(ORG['healthy'], 0.6), fontweight='bold')

    # ── Stage 2 ──
    s2_y = mid_y - 1.4
    s2_x = r_s1 + 0.35
    arr(ax, r_s1, mid_y-0.15, s2_x, s2_y+0.4, c=_darken(ST1, 0.8), lw=1.0)

    s2_h = 0.80
    r_s2 = block(ax, s2_x, s2_y-s2_h/2, 0.80, s2_h, d=0.08,
                 color=ST2, label='Stage 2\nCancer\nType', fs=5)

    # Type Head → Stage 2
    arr(ax, r_typ, hd2_y, s2_x, s2_y, c=_darken(RN[6], 0.8), lw=1.0)

    # ── Cancer type outputs — organ icons ──
    organs = [
        ('PRO', 'Prostate',   'prostate'),
        ('BRE', 'Breast',     'breast'),
        ('OVA', 'Ovarian',    'ovarian'),
        ('LUN', 'Lung',       'lung'),
        ('CRC', 'Colorectal', 'colon'),
        ('PAN', 'Pancreatic', 'pancreas'),
        ('BLC', 'Bladder',    'bladder'),
    ]

    out_x = r_s2 + 0.55
    n = len(organs)
    sp = 0.45
    top = s2_y + (n-1)*sp/2

    for i, (code, name, organ) in enumerate(organs):
        ty = top - i * sp
        oc = ORG[organ]
        arr(ax, r_s2, s2_y, out_x, ty, c=_lighten(oc, 0.3), lw=0.7)
        _draw_organ(ax, out_x+0.35, ty, 0.5, organ, oc)
        ax.text(out_x+0.75, ty, f'{code}  {name}', fontsize=5, fontweight='bold',
                color=_darken(oc, 0.6), va='center')

    # ── Save ──
    out = Path(__file__).parent
    for fmt in ['png','pdf','svg']:
        p = out / f'usersnet_architecture.{fmt}'
        fig.savefig(p, dpi=300, bbox_inches='tight', facecolor='none',
                    edgecolor='none', pad_inches=0.05, transparent=True)
        print(f'Saved: {p}')
    plt.close(fig)


if __name__ == '__main__':
    draw()
