import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def draw_color_palette(
    colors: list[str],
    orientation: str = "horizontal",
    rect_width: float = 2.0,
    rect_height: float = 1.0,
    gap: float = 0.15,
    label_position: str = "inside",
    figsize: tuple | None = None,
    title: str | None = None,
) -> plt.Figure:
    """
    Draw a sequence of filled rectangles, one per colour, each labelled
    with its colour name.

    Parameters
    ----------
    colors        : list of colour strings (any format recognised by matplotlib
                    — named colours, hex codes, xkcd names, RGB tuples, …)
    orientation   : 'horizontal'  → rectangles laid out left-to-right
                    'vertical'    → rectangles laid out top-to-bottom
    rect_width    : width of each rectangle (data units)
    rect_height   : height of each rectangle (data units)
    gap           : spacing between consecutive rectangles (data units)
    label_position: 'inside'  → label centred inside the rectangle
                    'below'   → label placed below each rectangle
                    'above'   → label placed above each rectangle
    figsize       : (width, height) in inches; auto-computed if None
    title         : optional figure title

    Returns
    -------
    fig : matplotlib Figure
    """
    n = len(colors)
    if n == 0:
        raise ValueError("The color list must not be empty.")

    orientation = orientation.lower()
    if orientation not in ("horizontal", "vertical"):
        raise ValueError("orientation must be 'horizontal' or 'vertical'.")
    label_position = label_position.lower()
    if label_position not in ("inside", "below", "above"):
        raise ValueError("label_position must be 'inside', 'below', or 'above'.")

    # ── Auto figure size ──────────────────────────────────────────────────────
    if figsize is None:
        if orientation == "horizontal":
            figsize = (n * (rect_width + gap) + 1.0, rect_height + 1.5)
        else:
            figsize = (rect_width + 2.0, n * (rect_height + gap) + 1.0)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#f8f8f8")
    ax.set_facecolor("#f8f8f8")
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Draw rectangles ───────────────────────────────────────────────────────
    step = (rect_width + gap) if orientation == "horizontal" else (rect_height + gap)

    for i, color in enumerate(colors):
        if orientation == "horizontal":
            x = i * step
            y = 0.0
        else:
            x = 0.0
            y = -(i * step)           # grow downward so index 0 is at top

        rect = mpatches.FancyBboxPatch(
            (x, y), rect_width, rect_height,
            boxstyle="round,pad=0.04",
            facecolor=color,
            edgecolor="white",
            linewidth=2,
        )
        ax.add_patch(rect)

        # ── Choose a contrasting label colour ─────────────────────────────────
        try:
            rgb = np.array(plt.matplotlib.colors.to_rgb(color))
            # Perceived luminance (ITU-R BT.601)
            luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
            label_color = "white" if luminance < 0.5 else "black"
        except (ValueError, AttributeError):
            label_color = "black"

        # ── Label position ────────────────────────────────────────────────────
        cx = x + rect_width  / 2       # rectangle centre x
        cy = y + rect_height / 2       # rectangle centre y

        if label_position == "inside":
            tx, ty = cx, cy
            va, ha = "center", "center"
            lc = label_color
            fontsize = min(9, rect_width * 4)
        elif label_position == "below":
            tx, ty = cx, y - 0.12
            va, ha = "top", "center"
            lc = "#333333"
            fontsize = 9
        else:   # above
            tx, ty = cx, y + rect_height + 0.12
            va, ha = "bottom", "center"
            lc = "#333333"
            fontsize = 9

        ax.text(
            tx, ty, str(color),
            ha=ha, va=va,
            fontsize=fontsize,
            fontfamily="monospace",
            color=lc,
            wrap=True,
            clip_on=False,
            rotation=0 if orientation == "horizontal" else 0,
        )

    # ── Axis limits with a small margin ──────────────────────────────────────
    if orientation == "horizontal":
        ax.set_xlim(-gap, n * step)
        ax.set_ylim(
            -0.6 if label_position == "below" else -0.2,
             rect_height + (0.5 if label_position == "above" else 0.2),
        )
    else:
        ax.set_xlim(-0.2, rect_width + 0.2)
        ax.set_ylim(
            -(n * step) - (0.5 if label_position == "below" else 0.1),
            rect_height + (0.5 if label_position == "above" else 0.2),
        )

    if title:
        ax.set_title(title, fontsize=12, fontfamily="monospace",
                     pad=10, color="#333333")

    plt.tight_layout()
    return fig


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Example 1 – named colours, horizontal, labels inside
    colors_named = [
        "xkcd:pale orange", "xkcd:sea blue", "xkcd:pale red",
        "xkcd:sage green",  "xkcd:terra cotta", "xkcd:dull purple",
        "xkcd:teal",
    ]
    fig1 = draw_color_palette(
        colors_named,
        orientation="horizontal",
        label_position="inside",
        rect_width=2.2,
        rect_height=1.2,
        title="Named colours — labels inside",
    )
    fig1.savefig("palette_horizontal.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Example 2 – hex codes, vertical, labels below
    colors_hex = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]
    fig2 = draw_color_palette(
        colors_hex,
        orientation="vertical",
        label_position="below",
        rect_width=2.5,
        rect_height=0.9,
        title="Hex codes — vertical layout",
    )
    fig2.savefig("palette_vertical.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Example 3 – CSS / standard names, horizontal, labels above
    colors_css = ["crimson", "darkorange", "gold", "mediumseagreen",
                  "steelblue", "mediumpurple", "hotpink"]
    fig3 = draw_color_palette(
        colors_css,
        orientation="horizontal",
        label_position="above",
        rect_width=1.8,
        rect_height=1.4,
        title="CSS names — labels above",
    )
    fig3.savefig("palette_above.png", dpi=150, bbox_inches="tight")
    plt.show()
