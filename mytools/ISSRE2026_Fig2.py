import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "cm",
        "axes.titlesize": 21,
        "axes.labelsize": 22,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 24,
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "axes.edgecolor": "#cbd5e1",
        "axes.linewidth": 0.9,
        "grid.color": "#e2e8f0",
        "grid.linewidth": 0.8,
    }
)

COLORS = {
    "positive": "#16a34a",
    "middle": "#f59e0b",
    "negative": "#dc2626",
    "threshold": "#64748b",
    "collision_zone": "#fecaca",
}

REGION_FILL = {
    "positive": "#ecfdf5",
    "middle": "#fffbeb",
    "negative": "#fee2e2",
}

THRESHOLD_LOWER = 0.55
THRESHOLD_UPPER = 1.45

FORMULA_LOWER = (
    r"$\mu_{\mathrm{L2\_Err}}^{\mathit{rep}}"
    r" - 0.5\,\sigma_{\mathrm{L2\_Err}}^{\mathit{rep}}$"
)
FORMULA_UPPER = (
    r"$\mu_{\mathrm{L2\_Err}}^{\mathit{rep}}"
    r" + 0.5\,\sigma_{\mathrm{L2\_Err}}^{\mathit{rep}}$"
)
FORMULA_MU = r"$\mu_{\mathrm{L2\_Err}}^{\mathit{rep}}$"

a, b = THRESHOLD_LOWER, THRESHOLD_UPPER
mu_schematic = (a + b) / 2
sigma_schematic = b - a

xmin = 0.0
xmax = mu_schematic + 1.35 * sigma_schematic
curve_y_max = 0.40
rng = np.random.default_rng(42)

Y_SPLIT = 0.5
Y_BOTTOM = 0.0
Y_TOP = 1.0


def windowed_normal(x, mu, sigma, x0, x1):
    """正态密度 × 窗口函数：左右端点在 x0/x1 处高度为 0。"""
    x = np.asarray(x, dtype=float)
    bell = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    half_span = (x1 - x0) / 2.0
    window = ((x - x0) * (x1 - x)) / (half_span**2)
    window = np.clip(window, 0.0, None)
    return bell * window


def curve_height(x):
    h = windowed_normal(x, mu_schematic, sigma_schematic, xmin, xmax)
    peak = windowed_normal(mu_schematic, mu_schematic, sigma_schematic, xmin, xmax)
    return curve_y_max * h / peak


no_collision_y_lo = 0.06
no_collision_y_hi = 0.44
collision_y_lo = 0.56
collision_y_hi = 0.94


def sample_no_collision_points(n_points):
    """x 方向按曲线高度加权；y 在 No collision 区域内均匀分散。"""
    x_grid = np.linspace(xmin, xmax, 400)
    weights = windowed_normal(
        x_grid, mu_schematic, sigma_schematic, xmin, xmax
    )
    weights /= weights.sum()

    xs = rng.choice(x_grid, size=n_points, p=weights)
    ys = rng.uniform(no_collision_y_lo, no_collision_y_hi, size=n_points)
    return xs, ys


# =========================
# 1. 示意数据
# =========================

n_no_collision = 420
n_collision = 12

l2_nc, y_nc = sample_no_collision_points(n_no_collision)
collisions_nc = np.zeros(n_no_collision, dtype=int)

x_grid = np.linspace(xmin, xmax, 400)
col_weights = windowed_normal(
    x_grid, mu_schematic, sigma_schematic, xmin, xmax
)
col_weights /= col_weights.sum()
l2_col = rng.choice(x_grid, size=n_collision, p=col_weights)
y_col = rng.uniform(collision_y_lo, collision_y_hi, size=n_collision)
collisions_col = np.ones(n_collision, dtype=int)

l2_errors = np.concatenate([l2_nc, l2_col])
y_positions = np.concatenate([y_nc, y_col])
collisions = np.concatenate([collisions_nc, collisions_col])

positive = (collisions == 0) & (l2_errors < a)
middle = (collisions == 0) & (l2_errors >= a) & (l2_errors < b)
negative = (collisions == 1) | (l2_errors >= b)

# =========================
# 2. 画图
# =========================

fig, ax = plt.subplots(figsize=(11.5, 5.6))
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)
fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.32)

ymin, ymax = Y_BOTTOM, Y_TOP
no_collision_top = (Y_SPLIT - ymin) / (ymax - ymin)
label_transform = blended_transform_factory(ax.transData, ax.transAxes)

region_specs = [
    (xmin, a, REGION_FILL["positive"]),
    (a, b, REGION_FILL["middle"]),
    (b, xmax, REGION_FILL["negative"]),
]
for x0, x1, fill in region_specs:
    ax.axvspan(
        x0,
        x1,
        ymin=0.0,
        ymax=no_collision_top,
        color=fill,
        alpha=0.95,
        zorder=0,
    )

ax.axhspan(
    Y_SPLIT,
    Y_TOP,
    xmin=0.0,
    xmax=1.0,
    color=COLORS["collision_zone"],
    alpha=0.55,
    zorder=0,
)

for x in (a, b):
    ax.axvline(
        x,
        color=COLORS["threshold"],
        linestyle=(0, (4, 4)),
        linewidth=1.4,
        alpha=0.85,
        zorder=1,
    )

ax.axhline(
    Y_SPLIT,
    color=COLORS["threshold"],
    linestyle=(0, (4, 4)),
    linewidth=1.4,
    alpha=0.85,
    zorder=1,
)

curve_x = np.linspace(xmin, xmax, 400)
curve_y = curve_height(curve_x)
ax.fill_between(
    curve_x,
    Y_BOTTOM,
    curve_y,
    color="#94a3b8",
    alpha=0.12,
    zorder=1,
)
ax.plot(
    curve_x,
    curve_y,
    color=COLORS["threshold"],
    linewidth=1.8,
    alpha=0.75,
    zorder=2,
)

ax.axvline(
    mu_schematic,
    color=COLORS["threshold"],
    linestyle="-",
    linewidth=1.3,
    alpha=0.75,
    zorder=2,
)

scatter_style = dict(s=36, linewidth=0.7, zorder=4)
for mask, label, color in (
    (positive, r"$D_{\mathit{rep}}^{+}$: Positive inputs", COLORS["positive"]),
    (middle, r"$D_{\mathit{rep}}^{\mathit{neutral}}$: Neutral inputs", COLORS["middle"]),
    (negative, r"$D_{\mathit{rep}}^{-}$: Negative inputs", COLORS["negative"]),
):
    ax.scatter(
        l2_errors[mask],
        y_positions[mask],
        c=color,
        label=label,
        edgecolor="white",
        alpha=0.85,
        **scatter_style,
    )

formula_label_offset = 0.16
formula_specs = (
    (a - formula_label_offset, FORMULA_LOWER),
    (mu_schematic, FORMULA_MU),
    (b + formula_label_offset, FORMULA_UPPER),
)
for x_pos, formula in formula_specs:
    ax.text(
        x_pos,
        -0.05,
        formula,
        transform=label_transform,
        ha="center",
        va="top",
        fontsize=27,
        color="#000000",
    )

ax.text(
    0.5,
    -0.21,
    "L2_Err",
    transform=ax.transAxes,
    ha="center",
    va="top",
    fontsize=25,
    color="#000000",
    family="monospace",
)

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_ylabel("isCollision", labelpad=10, fontfamily="monospace", fontsize=24)
ax.set_xticks([])
ax.tick_params(axis="x", which="both", bottom=False)
ax.tick_params(axis="y", labelsize=25)
ax.set_yticks([0.25, 0.75])
ax.set_yticklabels([r"$\mathit{false}$", r"$\mathit{true}$"])

legend = ax.legend(
    frameon=False,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=3,
    alignment="center",
    columnspacing=0.7,
    handletextpad=0.25,
    handlelength=1.0,
    borderaxespad=0.0,
    labelspacing=0.9,
    fontsize=24,
)

for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
for spine in ("left", "bottom"):
    ax.spines[spine].set_color("#cbd5e1")

plt.savefig(
    "l2_collision_three_categories_polished.pdf",
    bbox_inches="tight",
    transparent=True,
)

plt.show()
