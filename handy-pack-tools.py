#!/usr/bin/env python3
"""
DISCLAIMER: This code was written by Claude IA (you can probably tell by the comments.
withers_pack_tools.py

Handy Pack Making Tools - single-file edition.

Everything in one file for easy sharing: just this one .py plus Python +
a couple of pip packages, no other files needed.

Run it (double-click, or `python3 withers_pack_tools.py`) and a window
opens with a boot menu, and a "Window" menu (like File/Edit) that can jump
to any tool at any time:

  - Palette Switcher: recolour one or more target images using another
    image's colour palette (hue-family matching + brightness-weighted
    mapping so shading is preserved, not just snapped to a handful of
    existing colours). Never overwrites - always Save As / Save All to a
    folder you choose. Default filename: "{palette name} + {target name}.png".

  - Color Filter: load one or more textures and apply a live,
    multiplicative colour tint using an interactive colour wheel plus
    brightness/intensity sliders - no "Generate" button, the preview just
    refreshes automatically (about every 0.25s) whenever something
    changed. Also Save As / Save All, never overwrites.

Both tools' previews support scroll-wheel zoom (0.25x-4x).

Requirements
------------
- Python 3 with Tkinter (bundled with the standard python.org installer on
  Windows/macOS; on Linux you may need `sudo apt install python3-tk`).
- pip install pillow numpy
- Optional, for drag-and-drop onto the preview boxes: pip install tkinterdnd2
"""

import os
import sys
import math
import colorsys

try:
    import numpy as np
    import tkinter as tk
    from tkinter import filedialog, messagebox
    from PIL import Image, ImageTk
except Exception:
    import traceback
    traceback.print_exc()
    print(
        "\nFailed to start (see error above).\n"
        "Most likely cause: a required package is missing. Try:\n"
        "  pip install pillow numpy\n"
    )
    input("Press Enter to close...")
    sys.exit(1)

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    DND_AVAILABLE = True
except ImportError:
    DND_AVAILABLE = False

APP_TITLE = "Handy Pack Making Tools"
VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")


def _basename_no_ext(path):
    return os.path.splitext(os.path.basename(path))[0]


# =============================================================================
# ENGINE - Palette Switcher (no GUI, pure image-processing logic)
# =============================================================================

# Below this saturation, a colour is considered "neutral" (grey/near-black)
# rather than a genuine member of a coloured accent family. Neutral colours
# are not allowed to be pulled into a minority accent cluster just because
# they happen to share its brightness.
SATURATION_THRESHOLD = 0.35
# Colours within this many degrees of hue (on the 360 deg wheel) are
# considered part of the same hue family when clustering a palette.
HUE_GAP_DEGREES = 40


def luminance(rgb):
    r, g, b = rgb
    return 0.2126 * r + 0.7152 * g + 0.0722 * b  # standard perceptual luminance


def hue_and_saturation(rgb):
    r, g, b = (c / 255.0 for c in rgb)
    h, s, _ = colorsys.rgb_to_hsv(r, g, b)
    return h * 360.0, s


def cluster_by_hue(colors_with_counts):
    """
    Group colours into hue families. Only colours with saturation above
    SATURATION_THRESHOLD participate in hue clustering; low-saturation
    colours are folded into the largest (dominant) cluster.

    Returns a list of clusters, each a list of (color, count), sorted by
    total pixel count descending (index 0 = dominant/majority cluster).
    """
    chromatic = []
    neutral = []
    for color, count in colors_with_counts:
        _, s = hue_and_saturation(color)
        if s >= SATURATION_THRESHOLD:
            chromatic.append((color, count))
        else:
            neutral.append((color, count))

    if not chromatic:
        return [colors_with_counts]

    chromatic.sort(key=lambda cc: hue_and_saturation(cc[0])[0])
    hues = [hue_and_saturation(c)[0] for c, _ in chromatic]

    clusters = [[chromatic[0]]]
    for i in range(1, len(chromatic)):
        gap = hues[i] - hues[i - 1]
        if gap > HUE_GAP_DEGREES:
            clusters.append([])
        clusters[-1].append(chromatic[i])
    if len(clusters) > 1:
        wrap_gap = (hues[0] + 360) - hues[-1]
        if wrap_gap <= HUE_GAP_DEGREES:
            clusters[0] = clusters[-1] + clusters[0]
            clusters.pop()

    clusters.sort(key=lambda cl: sum(c for _, c in cl), reverse=True)

    if neutral:
        clusters[0] = clusters[0] + neutral

    return clusters


def extract_palette_with_counts(img, alpha_threshold=10):
    """Return list of (rgb, pixel_count) for opaque colours, sorted by luminance."""
    img = img.convert("RGBA")
    counts = {}
    for pixel in img.getdata():
        r, g, b, a = pixel
        if a > alpha_threshold:
            counts[(r, g, b)] = counts.get((r, g, b), 0) + 1
    return sorted(counts.items(), key=lambda item: luminance(item[0]))


def weighted_midpoints(colors_with_counts):
    """
    For each colour (sorted by luminance), compute the fraction of the
    image's pixels that are darker than it, at the midpoint of its own
    frequency band - this makes the mapping frequency-aware, so a colour
    covering 90% of pixels dominates the ranking the way it dominates the
    image, instead of being just one entry in a list of distinct colours.
    """
    total = sum(count for _, count in colors_with_counts)
    midpoints = []
    cum = 0
    for color, count in colors_with_counts:
        midpoints.append((color, (cum + count / 2) / total))
        cum += count
    return midpoints


def _sorted_cluster_positions(cluster, palette_pos):
    """Cluster colours sorted by their weighted brightness position (ascending)."""
    return sorted(((c, palette_pos[c]) for c, _ in cluster), key=lambda cp: cp[1])


def _interpolated_color(sorted_positions, tpos):
    """
    Return a colour linearly interpolated between the two palette colours
    that bracket tpos (clamped to the endpoints if tpos is outside the
    cluster's range). Unlike snapping to the nearest existing colour, this
    produces a fresh in-between colour, so two different target shades
    don't collapse onto the same output colour just because they were both
    closest to the same palette entry.
    """
    if len(sorted_positions) == 1:
        return sorted_positions[0][0]

    if tpos <= sorted_positions[0][1]:
        return sorted_positions[0][0]
    if tpos >= sorted_positions[-1][1]:
        return sorted_positions[-1][0]

    for i in range(len(sorted_positions) - 1):
        (c_lo, p_lo), (c_hi, p_hi) = sorted_positions[i], sorted_positions[i + 1]
        if p_lo <= tpos <= p_hi:
            span = p_hi - p_lo
            t = 0.0 if span == 0 else (tpos - p_lo) / span
            return tuple(
                round(c_lo[ch] + (c_hi[ch] - c_lo[ch]) * t) for ch in range(3)
            )
    return sorted_positions[-1][0]  # fallback, shouldn't be reached


def build_color_map(target_colors_with_counts, palette_colors_with_counts, generate_colors=True):
    """
    Map each target colour to an output colour (hue family, then brightness).

    generate_colors=True (default): within the matched hue family, the
    output colour is interpolated between the two neighbouring palette
    colours at the same brightness position, so distinct target shades stay
    distinct instead of several of them collapsing onto one palette colour.

    generate_colors=False: snap to whichever single palette colour in the
    matched family is closest in brightness (the older, simpler behaviour).
    """
    target_clusters = cluster_by_hue(target_colors_with_counts)
    palette_clusters = cluster_by_hue(palette_colors_with_counts)

    target_pos = dict(weighted_midpoints(target_colors_with_counts))
    palette_pos = dict(weighted_midpoints(palette_colors_with_counts))

    color_map = {}
    for cluster_idx, t_cluster in enumerate(target_clusters):
        p_cluster_idx = min(cluster_idx, len(palette_clusters) - 1)
        p_cluster = palette_clusters[p_cluster_idx]

        if generate_colors:
            sorted_positions = _sorted_cluster_positions(p_cluster, palette_pos)
            for tcolor, _ in t_cluster:
                color_map[tcolor] = _interpolated_color(sorted_positions, target_pos[tcolor])
        else:
            for tcolor, _ in t_cluster:
                tpos = target_pos[tcolor]
                best_color = min(
                    p_cluster, key=lambda pc: abs(palette_pos[pc[0]] - tpos)
                )[0]
                color_map[tcolor] = best_color
    return color_map


def recolor_image(palette_img, target_img, alpha_threshold=10, generate_colors=True):
    """Return a new RGBA image: target_img repainted with palette_img's colours."""
    target_img = target_img.convert("RGBA")

    palette_colors = extract_palette_with_counts(palette_img, alpha_threshold)
    target_colors = extract_palette_with_counts(target_img, alpha_threshold)
    color_map = build_color_map(target_colors, palette_colors, generate_colors=generate_colors)

    result = target_img.copy()
    pixels = result.load()
    width, height = result.size
    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            if a > alpha_threshold and (r, g, b) in color_map:
                nr, ng, nb = color_map[(r, g, b)]
                pixels[x, y] = (nr, ng, nb, a)
    return result


# =============================================================================
# ENGINE - Color Filter (no GUI, numpy-vectorized)
# =============================================================================

def hsv_to_rgb_np(h, s, v):
    """
    Vectorized HSV -> RGB. h, s, v are numpy arrays (same shape) with h, s, v
    all in the 0..1 range. Returns an array of shape (..., 3) in 0..1.
    """
    i = np.floor(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i.astype(int) % 6

    conditions = [i == 0, i == 1, i == 2, i == 3, i == 4, i == 5]
    r = np.select(conditions, [v, q, p, p, t, v])
    g = np.select(conditions, [t, v, v, q, p, p])
    b = np.select(conditions, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)


def make_wheel_rgba(size):
    """
    Build an RGBA numpy array (size x size x 4, uint8) of an HSV colour
    wheel: angle around the centre = hue, distance from the centre =
    saturation, value fixed at 1.0. Pixels outside the circle are
    transparent.
    """
    y, x = np.ogrid[0:size, 0:size]
    cx = cy = (size - 1) / 2.0
    radius = size / 2.0 - 2
    dx = x - cx
    dy = y - cy
    r = np.sqrt(dx ** 2 + dy ** 2)
    theta = np.degrees(np.arctan2(-dy, dx)) % 360.0

    h = theta / 360.0
    s = np.clip(r / radius, 0, 1)
    v = np.ones_like(s)

    rgb = hsv_to_rgb_np(h, s, v)
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[..., :3] = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    mask = r <= radius
    rgba[..., 3] = np.where(mask, 255, 0).astype(np.uint8)
    return rgba


def apply_tint_filter(img, hue_degrees, saturation, value, intensity):
    """
    Apply a multiplicative colour tint to img, numpy-vectorized so it's fast
    enough to recompute live while dragging the colour wheel.
    """
    tr, tg, tb = colorsys.hsv_to_rgb(hue_degrees / 360.0, saturation, value)
    max_c = max(tr, tg, tb, 1e-6)
    mult = np.array([tr, tg, tb], dtype=np.float32) / max_c * intensity

    arr = np.array(img.convert("RGBA"), dtype=np.float32)
    arr[..., :3] = np.clip(arr[..., :3] * mult, 0, 255)
    return Image.fromarray(arr.astype(np.uint8), "RGBA")


# =============================================================================
# WIDGET - zoomable image preview (scroll wheel to zoom)
# =============================================================================

ZOOM_STEP = 0.1
ZOOM_MIN = 0.25
ZOOM_MAX = 12.0


class ZoomableImagePreview(tk.Canvas):
    def __init__(self, parent, size=(260, 260), placeholder_text="No image selected", **kwargs):
        kwargs.setdefault("bg", "#f4f4f4")
        kwargs.setdefault("highlightthickness", 1)
        kwargs.setdefault("highlightbackground", "#999999")
        super().__init__(parent, width=size[0], height=size[1], **kwargs)
        self.box_size = size
        self.placeholder_text = placeholder_text

        self._img = None       # current PIL RGBA image, full resolution
        self._zoom = 1.0
        self._photo = None     # keep a reference so Tk doesn't garbage-collect it

        self.bind("<MouseWheel>", self._on_mousewheel)       # Windows / macOS
        self.bind("<Button-4>", self._on_mousewheel_linux_up)    # Linux scroll up
        self.bind("<Button-5>", self._on_mousewheel_linux_down)  # Linux scroll down

        self._draw_placeholder()

    # ------------------------------------------------------------- public --
    def show(self, image_or_path):
        """Display a PIL Image or a file path. Resets zoom to 1x (fit-to-box)."""
        img = Image.open(image_or_path) if isinstance(image_or_path, str) else image_or_path
        self._img = img.convert("RGBA")
        self._zoom = 1.0
        self._render()

    def update_image(self, image_or_path):
        """
        Like show(), but keeps the current zoom level instead of resetting
        to fit-to-box. Use this when redisplaying the *same* picture with
        different content - e.g. a live filter preview redrawing after
        every wheel tweak - so the user's zoom isn't wiped out every time.
        """
        img = Image.open(image_or_path) if isinstance(image_or_path, str) else image_or_path
        self._img = img.convert("RGBA")
        self._render()

    def clear(self, placeholder_text=None):
        if placeholder_text is not None:
            self.placeholder_text = placeholder_text
        self._img = None
        self._photo = None
        self._zoom = 1.0
        self._draw_placeholder()

    # ------------------------------------------------------------ drawing --
    def _base_fit_size(self):
        """Size the image is shown at when zoom == 1x: scaled to fill the
        box as much as possible while keeping its aspect ratio. Small
        textures (e.g. 16x16 pixel art) get scaled UP to fill the box
        instead of showing as a tiny speck - see _render() for how the
        resampling method is chosen so this stays crisp instead of blurry."""
        w, h = self._img.size
        box_w, box_h = self.box_size
        scale = min(box_w / w, box_h / h)
        return max(1, round(w * scale)), max(1, round(h * scale))

    def _render(self):
        if self._img is None:
            self._draw_placeholder()
            return

        base_w, base_h = self._base_fit_size()
        disp_w = max(1, round(base_w * self._zoom))
        disp_h = max(1, round(base_h * self._zoom))

        # Nearest-neighbour when enlarging keeps pixel art crisp (sharp
        # square pixels instead of a blurry smear) - important since most
        # textures here are small (16x16-ish). Lanczos is only used when
        # actually shrinking a larger image down, where it looks cleaner.
        enlarging = disp_w >= self._img.width or disp_h >= self._img.height
        resample = Image.NEAREST if enlarging else Image.LANCZOS

        resized = self._img.resize((disp_w, disp_h), resample)
        bg = Image.new("RGBA", resized.size, (244, 244, 244, 255))
        bg.alpha_composite(resized)

        self._photo = ImageTk.PhotoImage(bg)
        self.delete("all")
        cx, cy = self.box_size[0] / 2, self.box_size[1] / 2
        self.create_image(cx, cy, anchor="center", image=self._photo)

    def _draw_placeholder(self):
        self.delete("all")
        cx, cy = self.box_size[0] / 2, self.box_size[1] / 2
        self.create_text(cx, cy, text=self.placeholder_text, justify="center", fill="#555555")

    # ------------------------------------------------------------- zoom --
    def _zoom_by(self, delta):
        if self._img is None:
            return
        self._zoom = min(ZOOM_MAX, max(ZOOM_MIN, self._zoom + delta))
        self._render()

    def _on_mousewheel(self, event):
        self._zoom_by(ZOOM_STEP if event.delta > 0 else -ZOOM_STEP)

    def _on_mousewheel_linux_up(self, event):
        self._zoom_by(ZOOM_STEP)

    def _on_mousewheel_linux_down(self, event):
        self._zoom_by(-ZOOM_STEP)


# =============================================================================
# WIDGET - interactive HSV colour wheel
# =============================================================================

class ColorWheelCanvas(tk.Canvas):
    def __init__(self, parent, size=200, on_change=None, initial_hue=20.0, initial_sat=0.8, **kwargs):
        super().__init__(parent, width=size, height=size, highlightthickness=0, **kwargs)
        self.size = size
        self.center = (size - 1) / 2.0
        self.radius = size / 2.0 - 2
        self.on_change = on_change

        self.hue = initial_hue    # degrees, 0-360
        self.sat = initial_sat    # 0-1

        self._build_wheel_image()
        self.bind("<Button-1>", self._on_click)
        self.bind("<B1-Motion>", self._on_click)
        self._draw_selector()

    def _build_wheel_image(self):
        rgba = make_wheel_rgba(self.size)
        img = Image.fromarray(rgba, "RGBA")
        self._wheel_photo = ImageTk.PhotoImage(img)
        self.create_image(0, 0, anchor="nw", image=self._wheel_photo, tags="wheel")

    def _on_click(self, event):
        hue, sat = self._point_to_hue_sat(event.x, event.y)
        self.hue, self.sat = hue, sat
        self._draw_selector()
        if self.on_change:
            self.on_change(self.hue, self.sat)

    def _point_to_hue_sat(self, x, y):
        """Pure geometry, kept separate from _on_click so it's easy to test
        without a real Tk event object."""
        dx = x - self.center
        dy = y - self.center
        r = math.hypot(dx, dy)
        if r > self.radius:
            if r > 0:
                scale = self.radius / r
                dx *= scale
                dy *= scale
            r = self.radius
        theta = math.degrees(math.atan2(-dy, dx)) % 360.0
        sat = 0.0 if self.radius == 0 else min(r / self.radius, 1.0)
        return theta, sat

    def _draw_selector(self):
        self.delete("selector")
        theta = math.radians(self.hue)
        r = self.sat * self.radius
        sx = self.center + r * math.cos(theta)
        sy = self.center - r * math.sin(theta)
        self.create_oval(sx - 5, sy - 5, sx + 5, sy + 5, outline="black", width=2, tags="selector")
        self.create_oval(sx - 5, sy - 5, sx + 5, sy + 5, outline="white", width=1, tags="selector")

    def set_hue_sat(self, hue, sat):
        """Let an embedding tool set the wheel's position programmatically."""
        self.hue = hue % 360.0
        self.sat = max(0.0, min(1.0, sat))
        self._draw_selector()

    def get_hue_sat(self):
        return self.hue, self.sat


# =============================================================================
# TOOL - Palette Switcher
# =============================================================================

PALETTE_PREVIEW_SIZE = (260, 260)


def _palette_placeholder_text(base_text):
    hint = "\n\n(drag & drop here, scroll to zoom)" if DND_AVAILABLE else "\n\n(scroll to zoom)"
    return base_text + hint


class PaletteSwitcherTool:
    def __init__(self, root):
        self.root = root

        self.palette_path = None
        self.target_paths = []
        self.target_index = 0
        self.results = []       # [(source_path, PIL image), ...]
        self.result_index = 0

        self._build_layout()

    def stop(self):
        pass  # nothing to cancel - no periodic callbacks in this tool

    # ---------------------------------------------------------------- UI --
    def _build_layout(self):
        columns = tk.Frame(self.root)
        columns.pack(fill="both", expand=True, padx=12, pady=12)
        for i in range(3):
            columns.columnconfigure(i, weight=1)
        columns.rowconfigure(1, weight=1)

        # --- Column 1: palette image ---
        pframe = tk.Frame(columns, relief="groove", borderwidth=1)
        pframe.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=6)
        tk.Label(pframe, text="1. Palette image (colours to copy)", font=("", 10, "bold"), wraplength=240).pack(pady=(10, 6))
        self.palette_preview = ZoomableImagePreview(pframe, PALETTE_PREVIEW_SIZE, _palette_placeholder_text("No image selected"))
        self.palette_preview.pack(padx=10, pady=6)
        if DND_AVAILABLE:
            self._register_drop(self.palette_preview, self._on_drop_palette)
        prow = tk.Frame(pframe)
        prow.pack(pady=(4, 2))
        tk.Button(prow, text="Browse...", command=self.choose_palette).pack(side="left")
        tk.Button(prow, text="Reset", command=self.reset_palette).pack(side="left", padx=(4, 0))
        self.palette_info = tk.Label(pframe, text="", wraplength=240, fg="#555")
        self.palette_info.pack(pady=(0, 10))

        # --- Column 2: target image(s) ---
        tframe = tk.Frame(columns, relief="groove", borderwidth=1)
        tframe.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=6)
        tk.Label(tframe, text="2. Target image(s) (shape to keep)", font=("", 10, "bold"), wraplength=240).pack(pady=(10, 6))
        self.target_preview = ZoomableImagePreview(tframe, PALETTE_PREVIEW_SIZE, _palette_placeholder_text("No image selected"))
        self.target_preview.pack(padx=10, pady=6)
        if DND_AVAILABLE:
            self._register_drop(self.target_preview, self._on_drop_target)
        tnav = tk.Frame(tframe)
        tnav.pack(pady=(4, 2))
        self.target_prev_btn = tk.Button(tnav, text="\u25c0", width=3, command=self.prev_target)
        self.target_prev_btn.pack(side="left")
        self.target_nav_label = tk.Label(tnav, text="0 / 0", width=8)
        self.target_nav_label.pack(side="left")
        self.target_next_btn = tk.Button(tnav, text="\u25b6", width=3, command=self.next_target)
        self.target_next_btn.pack(side="left")
        trow = tk.Frame(tframe)
        trow.pack(pady=(2, 2))
        tk.Button(trow, text="Browse...", command=self.choose_target).pack(side="left")
        tk.Button(trow, text="Reset", command=self.reset_target).pack(side="left", padx=(4, 0))
        self.target_info = tk.Label(tframe, text="", wraplength=240, fg="#555")
        self.target_info.pack(pady=(0, 10))

        # --- Column 3: result ---
        rframe = tk.Frame(columns, relief="groove", borderwidth=1)
        rframe.grid(row=0, column=2, rowspan=2, sticky="nsew", padx=6)
        tk.Label(rframe, text="3. Result", font=("", 10, "bold")).pack(pady=(10, 6))
        self.result_preview = ZoomableImagePreview(rframe, PALETTE_PREVIEW_SIZE, "No image selected\n\n(scroll to zoom)")
        self.result_preview.pack(padx=10, pady=6)
        rnav = tk.Frame(rframe)
        rnav.pack(pady=(4, 2))
        self.result_prev_btn = tk.Button(rnav, text="\u25c0", width=3, command=self.prev_result)
        self.result_prev_btn.pack(side="left")
        self.result_nav_label = tk.Label(rnav, text="0 / 0", width=8)
        self.result_nav_label.pack(side="left")
        self.result_next_btn = tk.Button(rnav, text="\u25b6", width=3, command=self.next_result)
        self.result_next_btn.pack(side="left")
        self.result_info = tk.Label(rframe, text="", wraplength=240, fg="#555")
        self.result_info.pack(pady=(0, 6))
        self.save_current_btn = tk.Button(rframe, text="Save current as...", command=self.save_current_result, state="disabled")
        self.save_current_btn.pack(pady=(0, 4))
        self.save_all_btn = tk.Button(rframe, text="Save all to folder...", command=self.save_all_results, state="disabled")
        self.save_all_btn.pack(pady=(0, 10))

        # --- bottom controls ---
        controls = tk.Frame(self.root)
        controls.pack(fill="x", padx=12, pady=(0, 12))

        tk.Button(controls, text="Generate", command=self.generate, width=14).pack(side="left")

        self.generate_colors_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            controls, text="Generate new colours (don't reuse one colour)",
            variable=self.generate_colors_var
        ).pack(side="left", padx=(12, 0))

        self.status = tk.Label(controls, text="Choose a palette image and target image(s) to begin.", anchor="w")
        self.status.pack(side="left", padx=(16, 0), fill="x", expand=True)

        if not DND_AVAILABLE:
            tk.Label(
                self.root,
                text="Tip: run 'pip install tkinterdnd2' to enable drag-and-drop.",
                fg="#888", font=("", 8)
            ).pack(side="bottom", pady=(0, 4))

    def _register_drop(self, widget, handler):
        widget.drop_target_register(DND_FILES)
        widget.dnd_bind("<<Drop>>", handler)

    def _split_drop_paths(self, event):
        try:
            raw = self.root.tk.splitlist(event.data)
        except Exception:
            raw = [event.data]
        paths = [p.strip("{}") for p in raw]
        return [p for p in paths if p.lower().endswith(VALID_EXTENSIONS)]

    # ------------------------------------------------------------ palette --
    def choose_palette(self):
        path = filedialog.askopenfilename(
            title="Choose the palette image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif *.webp"), ("All files", "*.*")],
        )
        if path:
            self._set_palette(path)

    def _on_drop_palette(self, event):
        valid = self._split_drop_paths(event)
        if not valid:
            messagebox.showwarning("Unsupported file", "That doesn't look like an image file.")
            return
        self._set_palette(valid[0])

    def _set_palette(self, path):
        self.palette_path = path
        self.palette_preview.show(path)
        self.palette_info.configure(text=os.path.basename(path))
        self._clear_results()
        self._update_status()

    def reset_palette(self):
        self.palette_path = None
        self.palette_preview.clear()
        self.palette_info.configure(text="")
        self._clear_results()
        self._update_status()

    # ------------------------------------------------------------- target --
    def choose_target(self):
        paths = filedialog.askopenfilenames(
            title="Choose target image(s)",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif *.webp"), ("All files", "*.*")],
        )
        if paths:
            self._add_targets(list(paths))

    def _on_drop_target(self, event):
        valid = self._split_drop_paths(event)
        if not valid:
            messagebox.showwarning("Unsupported file", "That doesn't look like an image file.")
            return
        self._add_targets(valid)

    def _add_targets(self, paths):
        """Add newly chosen/dropped paths to the existing target list instead
        of replacing it, so multiple separate Browse clicks or drag-and-drops
        accumulate rather than each wiping out the last. Use Reset to start
        over. Duplicate paths are skipped."""
        added_first_index = len(self.target_paths)
        for p in paths:
            if p not in self.target_paths:
                self.target_paths.append(p)
        if len(self.target_paths) == added_first_index:
            return  # everything dropped/chosen was already in the list
        self.target_index = added_first_index
        self._clear_results()
        self._refresh_target_preview()
        self._update_status()

    def reset_target(self):
        self.target_paths = []
        self.target_index = 0
        self.target_preview.clear()
        self.target_info.configure(text="")
        self.target_nav_label.configure(text="0 / 0")
        self._clear_results()
        self._update_status()

    def _refresh_target_preview(self):
        if not self.target_paths:
            self.target_preview.clear()
            self.target_info.configure(text="")
            self.target_nav_label.configure(text="0 / 0")
            return
        path = self.target_paths[self.target_index]
        self.target_preview.show(path)
        self.target_info.configure(text=os.path.basename(path))
        self.target_nav_label.configure(text=f"{self.target_index + 1} / {len(self.target_paths)}")

    def prev_target(self):
        if len(self.target_paths) > 1:
            self.target_index = (self.target_index - 1) % len(self.target_paths)
            self._refresh_target_preview()

    def next_target(self):
        if len(self.target_paths) > 1:
            self.target_index = (self.target_index + 1) % len(self.target_paths)
            self._refresh_target_preview()

    # -------------------------------------------------------------- status --
    def _update_status(self):
        if self.palette_path and self.target_paths:
            self.status.configure(text="Ready. Click Generate.")
        elif not self.palette_path:
            self.status.configure(text="Choose a palette image.")
        else:
            self.status.configure(text="Choose target image(s).")

    # ------------------------------------------------------------ generate --
    def default_result_filename(self, source_path):
        palette_base = _basename_no_ext(self.palette_path) if self.palette_path else "palette"
        target_base = _basename_no_ext(source_path)
        return f"{palette_base} + {target_base}.png"

    def generate(self):
        if not self.palette_path or not self.target_paths:
            messagebox.showwarning("Missing image", "Please choose a palette image and at least one target image first.")
            return
        try:
            self.status.configure(text="Working...")
            self.root.update_idletasks()

            palette_img = Image.open(self.palette_path)
            generate_colors = self.generate_colors_var.get()

            self.results = []
            for path in self.target_paths:
                target_img = Image.open(path)
                result = recolor_image(palette_img, target_img, generate_colors=generate_colors)
                self.results.append((path, result))

            self.result_index = 0
            self._refresh_result_preview()
            self.save_current_btn.configure(state="normal")
            self.save_all_btn.configure(state="normal")
            self.status.configure(text=f"Done. {len(self.results)} image(s) processed.")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            self.status.configure(text="Something went wrong - see error message.")

    def _clear_results(self):
        self.results = []
        self.result_index = 0
        self.result_preview.clear()
        self.result_info.configure(text="")
        self.result_nav_label.configure(text="0 / 0")
        self.save_current_btn.configure(state="disabled")
        self.save_all_btn.configure(state="disabled")

    def _refresh_result_preview(self):
        if not self.results:
            self._clear_results()
            return
        source_path, img = self.results[self.result_index]
        self.result_preview.show(img)
        self.result_info.configure(text=self.default_result_filename(source_path))
        self.result_nav_label.configure(text=f"{self.result_index + 1} / {len(self.results)}")

    def prev_result(self):
        if len(self.results) > 1:
            self.result_index = (self.result_index - 1) % len(self.results)
            self._refresh_result_preview()

    def next_result(self):
        if len(self.results) > 1:
            self.result_index = (self.result_index + 1) % len(self.results)
            self._refresh_result_preview()

    # ---------------------------------------------------------------- save --
    def save_current_result(self):
        if not self.results:
            return
        source_path, img = self.results[self.result_index]
        default_name = self.default_result_filename(source_path)
        path = filedialog.asksaveasfilename(
            title="Save result image",
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[("PNG image", "*.png")],
        )
        if path:
            img.save(path)
            self.status.configure(text=f"Saved to {path}")

    def save_all_results(self):
        if not self.results:
            return
        folder = filedialog.askdirectory(title="Choose a folder to save all results into")
        if not folder:
            return
        for source_path, img in self.results:
            out_path = os.path.join(folder, self.default_result_filename(source_path))
            img.save(out_path)
        self.status.configure(text=f"Saved {len(self.results)} image(s) to {folder}")


# =============================================================================
# TOOL - Color Filter
# =============================================================================

FILTER_PREVIEW_SIZE = (300, 300)
UPDATE_INTERVAL_MS = 250


class ColorWheelFilterTool:
    def __init__(self, root):
        self.root = root

        self.texture_paths = []
        self.texture_index = 0
        self.dirty = False       # True when the wheel/sliders changed since last refresh
        self._after_id = None

        self._build_layout()
        self.start_loop()

    def stop(self):
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
            self._after_id = None

    # --------------------------------------------------------- live loop --
    def start_loop(self):
        self._after_id = self.root.after(UPDATE_INTERVAL_MS, self._tick)

    def _tick(self):
        if self.dirty:
            self.dirty = False
            self._refresh_preview(reset_zoom=False)
        self._after_id = self.root.after(UPDATE_INTERVAL_MS, self._tick)

    def _mark_dirty(self, *_args):
        self.dirty = True

    def _on_wheel_change(self, hue, sat):
        self._mark_dirty()
        self._refresh_hex_display()

    def _on_slider_change(self, *_args):
        self._mark_dirty()
        self._refresh_hex_display()

    # --------------------------------------------------------- hex/RGB input --
    def _current_rgb(self):
        hue, sat = self.wheel.get_hue_sat()
        value = self.value_scale.get()
        r, g, b = colorsys.hsv_to_rgb(hue / 360.0, sat, value)
        return round(r * 255), round(g * 255), round(b * 255)

    def _refresh_hex_display(self):
        r, g, b = self._current_rgb()
        self.current_color_label.configure(text=f"Current: #{r:02x}{g:02x}{b:02x}  ({r}, {g}, {b})")

    def _apply_color_input(self, *_args):
        text = self.color_entry.get().strip()
        rgb = self._parse_color_text(text)
        if rgb is None:
            messagebox.showwarning(
                "Couldn't read that colour",
                "Use a hex code like #ff8800 or RGB like 255,136,0."
            )
            return
        r, g, b = rgb
        h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
        self.wheel.set_hue_sat(h * 360.0, s)
        self.value_scale.set(v)
        self._refresh_hex_display()
        self.dirty = False  # about to refresh immediately, nothing left pending
        self._refresh_preview(reset_zoom=False)  # apply immediately, don't wait for the timer

    @staticmethod
    def _parse_color_text(text):
        text = text.strip()
        if not text:
            return None

        hex_text = text[1:] if text.startswith("#") else text
        if len(hex_text) == 6 and all(c in "0123456789abcdefABCDEF" for c in hex_text):
            try:
                r = int(hex_text[0:2], 16)
                g = int(hex_text[2:4], 16)
                b = int(hex_text[4:6], 16)
                return r, g, b
            except ValueError:
                return None

        parts = [p.strip() for p in text.replace(";", ",").split(",")]
        if len(parts) == 3:
            try:
                r, g, b = (int(p) for p in parts)
                if all(0 <= c <= 255 for c in (r, g, b)):
                    return r, g, b
            except ValueError:
                return None

        return None

    # ---------------------------------------------------------------- UI --
    def _build_layout(self):
        columns = tk.Frame(self.root)
        columns.pack(fill="both", expand=True, padx=12, pady=12)
        columns.columnconfigure(0, weight=0)
        columns.columnconfigure(1, weight=1)
        columns.rowconfigure(0, weight=1)

        # --- left: colour wheel + sliders ---
        left = tk.Frame(columns, relief="groove", borderwidth=1)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 6))

        tk.Label(left, text="Filter colour", font=("", 10, "bold")).pack(pady=(10, 6))
        self.wheel = ColorWheelCanvas(left, size=200, on_change=self._on_wheel_change)
        self.wheel.pack(padx=10, pady=(0, 6))

        self.current_color_label = tk.Label(left, text="", font=("", 9))
        self.current_color_label.pack(pady=(0, 6))

        color_input_row = tk.Frame(left)
        color_input_row.pack(pady=(0, 10))
        tk.Label(color_input_row, text="Hex or R,G,B:").pack(side="left")
        self.color_entry = tk.Entry(color_input_row, width=12)
        self.color_entry.pack(side="left", padx=(4, 4))
        self.color_entry.bind("<Return>", self._apply_color_input)
        tk.Button(color_input_row, text="Set", command=self._apply_color_input).pack(side="left")

        tk.Label(left, text="Brightness (value)").pack()
        self.value_scale = tk.Scale(
            left, from_=0.0, to=1.0, resolution=0.01, orient="horizontal",
            length=200, command=self._on_slider_change
        )
        self.value_scale.set(1.0)
        self.value_scale.pack(pady=(0, 10))

        tk.Label(left, text="Intensity").pack()
        self.intensity_scale = tk.Scale(
            left, from_=0.1, to=6.0, resolution=0.05, orient="horizontal",
            length=200, command=self._mark_dirty
        )
        self.intensity_scale.set(2.0)
        self.intensity_scale.pack(pady=(0, 10))

        self._refresh_hex_display()

        # --- right: texture loader + live preview ---
        right = tk.Frame(columns, relief="groove", borderwidth=1)
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        tk.Label(right, text="Texture(s)", font=("", 10, "bold")).pack(pady=(10, 6))
        placeholder = "No textures loaded\n\n(drag & drop here, scroll to zoom)" if DND_AVAILABLE else "No textures loaded\n\n(scroll to zoom)"
        self.preview = ZoomableImagePreview(right, FILTER_PREVIEW_SIZE, placeholder)
        self.preview.pack(padx=10, pady=6)
        if DND_AVAILABLE:
            self._register_drop(self.preview)

        nav = tk.Frame(right)
        nav.pack(pady=(4, 2))
        self.prev_btn = tk.Button(nav, text="\u25c0", width=3, command=self.prev_texture)
        self.prev_btn.pack(side="left")
        self.nav_label = tk.Label(nav, text="0 / 0", width=10)
        self.nav_label.pack(side="left")
        self.next_btn = tk.Button(nav, text="\u25b6", width=3, command=self.next_texture)
        self.next_btn.pack(side="left")

        btn_row = tk.Frame(right)
        btn_row.pack(pady=(2, 2))
        tk.Button(btn_row, text="Browse...", command=self.choose_textures).pack(side="left")
        tk.Button(btn_row, text="Reset", command=self.reset_textures).pack(side="left", padx=(4, 0))

        self.info_label = tk.Label(right, text="", wraplength=300, fg="#555")
        self.info_label.pack(pady=(0, 8))

        save_row = tk.Frame(right)
        save_row.pack(pady=(0, 10))
        self.save_current_btn = tk.Button(save_row, text="Save current as...", command=self.save_current, state="disabled")
        self.save_current_btn.pack(side="left")
        self.save_all_btn = tk.Button(save_row, text="Save all to folder...", command=self.save_all, state="disabled")
        self.save_all_btn.pack(side="left", padx=(6, 0))

        self.status = tk.Label(self.root, text="Load one or more textures, then adjust the wheel.", anchor="w")
        self.status.pack(fill="x", padx=12, pady=(0, 8))

        if not DND_AVAILABLE:
            tk.Label(
                self.root,
                text="Tip: run 'pip install tkinterdnd2' to enable drag-and-drop.",
                fg="#888", font=("", 8)
            ).pack(side="bottom", pady=(0, 4))

    def _register_drop(self, widget):
        widget.drop_target_register(DND_FILES)
        widget.dnd_bind("<<Drop>>", self._on_drop)

    # ------------------------------------------------------------ loading --
    def choose_textures(self):
        paths = filedialog.askopenfilenames(
            title="Choose one or more texture images",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif *.webp"), ("All files", "*.*")],
        )
        if paths:
            self._add_textures(list(paths))

    def _on_drop(self, event):
        try:
            raw = self.root.tk.splitlist(event.data)
        except Exception:
            raw = [event.data]
        paths = [p.strip("{}") for p in raw]
        valid = [p for p in paths if p.lower().endswith(VALID_EXTENSIONS)]
        if not valid:
            messagebox.showwarning("Unsupported file", "That doesn't look like an image file.")
            return
        self._add_textures(valid)

    def _add_textures(self, paths):
        """Add newly chosen/dropped paths to the existing texture list
        instead of replacing it, so multiple separate Browse clicks or
        drag-and-drops accumulate rather than each wiping out the last.
        Use Reset to start over. Duplicate paths are skipped."""
        added_first_index = len(self.texture_paths)
        for p in paths:
            if p not in self.texture_paths:
                self.texture_paths.append(p)
        if len(self.texture_paths) == added_first_index:
            return  # everything dropped/chosen was already in the list
        self.texture_index = added_first_index
        self.save_current_btn.configure(state="normal")
        self.save_all_btn.configure(state="normal")
        self._refresh_preview()
        self.status.configure(text=f"{len(self.texture_paths)} texture(s) loaded. Adjust the wheel to preview.")

    def reset_textures(self):
        self.texture_paths = []
        self.texture_index = 0
        self.preview.clear()
        self.info_label.configure(text="")
        self.nav_label.configure(text="0 / 0")
        self.save_current_btn.configure(state="disabled")
        self.save_all_btn.configure(state="disabled")
        self.status.configure(text="Load one or more textures, then adjust the wheel.")

    # ---------------------------------------------------------- filtering --
    def _current_filter_params(self):
        hue, sat = self.wheel.get_hue_sat()
        return hue, sat, self.value_scale.get(), self.intensity_scale.get()

    def _refresh_preview(self, reset_zoom=True):
        if not self.texture_paths:
            self.preview.clear()
            self.nav_label.configure(text="0 / 0")
            return
        path = self.texture_paths[self.texture_index]
        try:
            img = Image.open(path)
        except Exception as exc:
            messagebox.showerror("Error", f"Couldn't open {path}:\n{exc}")
            return
        hue, sat, value, intensity = self._current_filter_params()
        filtered = apply_tint_filter(img, hue, sat, value, intensity)
        if reset_zoom:
            self.preview.show(filtered)
        else:
            self.preview.update_image(filtered)
        self.info_label.configure(text=os.path.basename(path))
        self.nav_label.configure(text=f"{self.texture_index + 1} / {len(self.texture_paths)}")

    def prev_texture(self):
        if len(self.texture_paths) > 1:
            self.texture_index = (self.texture_index - 1) % len(self.texture_paths)
            self._refresh_preview()

    def next_texture(self):
        if len(self.texture_paths) > 1:
            self.texture_index = (self.texture_index + 1) % len(self.texture_paths)
            self._refresh_preview()

    # ---------------------------------------------------------------- save --
    def _default_filename(self, source_path):
        return f"{_basename_no_ext(source_path)}_filtered.png"

    def save_current(self):
        if not self.texture_paths:
            return
        path = self.texture_paths[self.texture_index]
        img = Image.open(path)
        hue, sat, value, intensity = self._current_filter_params()
        filtered = apply_tint_filter(img, hue, sat, value, intensity)

        default_name = self._default_filename(path)
        out_path = filedialog.asksaveasfilename(
            title="Save filtered image",
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[("PNG image", "*.png")],
        )
        if out_path:
            filtered.save(out_path)
            self.status.configure(text=f"Saved to {out_path}")

    def save_all(self):
        if not self.texture_paths:
            return
        folder = filedialog.askdirectory(title="Choose a folder to save all filtered textures into")
        if not folder:
            return
        hue, sat, value, intensity = self._current_filter_params()
        for path in self.texture_paths:
            img = Image.open(path)
            filtered = apply_tint_filter(img, hue, sat, value, intensity)
            out_path = os.path.join(folder, self._default_filename(path))
            filtered.save(out_path)
        self.status.configure(text=f"Saved {len(self.texture_paths)} image(s) to {folder}")


# =============================================================================
# SHELL - boot menu, Window menu, tool switching, title bar
# =============================================================================

# Name -> factory that takes root and returns a tool instance.
TOOL_REGISTRY = {
    "Palette Switcher": lambda root: PaletteSwitcherTool(root),
    "Color Filter": lambda root: ColorWheelFilterTool(root),
}


class AppShell:
    def __init__(self, root):
        self.root = root
        self.current_tool = None
        self._build_menu()
        self.show_boot_menu()

    # ------------------------------------------------------------- menu --
    def _build_menu(self):
        menubar = tk.Menu(self.root)

        window_menu = tk.Menu(menubar, tearoff=0)
        window_menu.add_command(label="Boot Menu", command=self.show_boot_menu)
        window_menu.add_separator()
        for name in TOOL_REGISTRY:
            window_menu.add_command(label=name, command=lambda n=name: self.open_tool(n))
        menubar.add_cascade(label="Window", menu=window_menu)

        self.root.config(menu=menubar)

    # ----------------------------------------------------- tool switching --
    def _teardown_current(self):
        if self.current_tool is not None and hasattr(self.current_tool, "stop"):
            self.current_tool.stop()
        self.current_tool = None
        for widget in self.root.winfo_children():
            widget.destroy()

    def show_boot_menu(self):
        self._teardown_current()
        self.root.title(f"{APP_TITLE} - Boot Menu")

        frame = tk.Frame(self.root)
        frame.pack(expand=True, fill="both")

        tk.Label(frame, text="Handy Pack Making Tools", font=("", 16, "bold")).pack(pady=(50, 8))
        tk.Label(frame, text="Choose a tool to open:").pack(pady=(0, 20))

        for name in TOOL_REGISTRY:
            tk.Button(
                frame, text=name, width=32,
                command=lambda n=name: self.open_tool(n)
            ).pack(pady=6)

    def open_tool(self, name):
        if name not in TOOL_REGISTRY:
            return
        self._teardown_current()
        self.root.title(f"{APP_TITLE} - {name}")
        self.current_tool = TOOL_REGISTRY[name](self.root)
        self._add_new_window_button(self.root)

    # ------------------------------------------------------ multi-window --
    def _add_new_window_button(self, container):
        """
        Small 'New Window' button pinned to the top-right corner of a tool
        window (never on the boot menu - callers just don't call this
        there). Lets you pop open any tool in its own independent window,
        so e.g. Palette Switcher and Color Filter can both be open and
        usable side by side, or two windows of the same tool at once.
        Uses .place() so it floats above the tool's own pack/grid layout
        without interfering with it.
        """
        button = tk.Menubutton(container, text="\u2750 New Window", relief="raised")
        menu = tk.Menu(button, tearoff=0)
        for name in TOOL_REGISTRY:
            menu.add_command(label=name, command=lambda n=name: self.open_new_window(n))
        button.configure(menu=menu)
        button.place(relx=1.0, x=-8, y=6, anchor="ne")
        return button

    def open_new_window(self, name):
        if name not in TOOL_REGISTRY:
            return
        top = tk.Toplevel(self.root)
        top.title(f"{APP_TITLE} - {name}")
        top.geometry("920x600")
        top.minsize(760, 480)

        tool_instance = TOOL_REGISTRY[name](top)

        def on_close():
            if hasattr(tool_instance, "stop"):
                tool_instance.stop()
            top.destroy()

        top.protocol("WM_DELETE_WINDOW", on_close)
        self._add_new_window_button(top)
        return top


def main():
    root = TkinterDnD.Tk() if DND_AVAILABLE else tk.Tk()
    root.geometry("920x600")
    root.minsize(760, 480)
    AppShell(root)
    root.mainloop()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # On Windows, double-clicking a .py file runs it in a console window
        # that closes itself the instant the script exits - if something
        # crashes on startup, the error flashes by too fast to read.
        # Catching it here and pausing lets you actually see it.
        import traceback
        traceback.print_exc()
        input("\nSomething went wrong (see error above). Press Enter to close...")
