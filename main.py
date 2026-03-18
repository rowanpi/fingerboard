"""
Violin fingerboard note-position calculator.

Displays fingerboard.png and asks you to click 10 calibration points:
  1:  Anywhere on the NUT line           → captures the nut Y pixel
  2:  Anywhere on the BODY junction line → captures the body Y pixel
  3-10: Two points on each string (G, D, A, E) to define its angle.
        First click near the nut end, second click near the body end.

Physical distance from nut to body = 13 cm.
Full vibrating string length L = 32.8 cm (0.328 m).

Position of the n-th half-step from the nut:
    d(n) = L - L / 2^(n/12)   (in metres)

Notes at the same half-step share the same Y coordinate.
X is interpolated along each string's angled line at that Y.
Notes are plotted all the way to the bottom of the image (not limited to 13 cm).

Press 'r' at any time to reset and start over.
"""

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("macosx")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

IMAGE_PATH = Path(__file__).parent / "fingerBoardFirstPos.png"
STRING_LENGTH_M = 0.328
FINGERBOARD_CM = 13.0
STRINGS = ["G", "D", "A", "E"]
OPEN_NOTES = {
    "G": ("G", 3),
    "D": ("D", 4),
    "A": ("A", 4),
    "E": ("E", 5),
}
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


class ResetException(Exception):
    pass


def half_step_position_m(n: int) -> float:
    """Distance from the nut for the n-th semitone (metres)."""
    return STRING_LENGTH_M - STRING_LENGTH_M / (2 ** (n / 12))


def note_name(open_note: str, open_octave: int, n: int) -> str:
    base_idx = NOTE_NAMES.index(open_note)
    new_idx = (base_idx + n) % 12
    octave = open_octave + (base_idx + n) // 12
    return f"{NOTE_NAMES[new_idx]}{octave}"


class Magnifier:
    """Shows a zoomed inset that follows the mouse cursor."""

    def __init__(self, ax, fig, img, zoom=4, size=80):
        self._ax = ax
        self._fig = fig
        self._img = img
        self._zoom = zoom
        self._size = size
        self._inset = None
        self._cid = None

    def enable(self):
        self._cid = self._fig.canvas.mpl_connect("motion_notify_event", self._on_move)

    def disable(self):
        if self._cid is not None:
            self._fig.canvas.mpl_disconnect(self._cid)
            self._cid = None
        if self._inset is not None:
            self._inset.remove()
            self._inset = None
            self._fig.canvas.draw_idle()

    def _on_move(self, event):
        if event.inaxes != self._ax:
            return
        cx, cy = event.xdata, event.ydata
        if cx is None or cy is None:
            return

        h, w = self._img.shape[:2]
        half = self._size / self._zoom / 2
        x0 = max(0, cx - half)
        x1 = min(w, cx + half)
        y0 = max(0, cy - half)
        y1 = min(h, cy + half)

        if self._inset is not None:
            self._inset.remove()

        # Place the magnifier inset in the top-left corner of the axes.
        self._inset = self._ax.inset_axes(
            [0.02, 0.72, 0.25, 0.25],
            xlim=(x0, x1), ylim=(y1, y0),
        )
        self._inset.imshow(self._img, aspect="auto")
        self._inset.axhline(cy, color="cyan", linewidth=0.5, alpha=0.6)
        self._inset.axvline(cx, color="cyan", linewidth=0.5, alpha=0.6)
        self._inset.set_xticks([])
        self._inset.set_yticks([])
        for spine in self._inset.spines.values():
            spine.set_edgecolor("yellow")
            spine.set_linewidth(1.5)
        self._fig.canvas.draw_idle()


def click(ax, fig, prompt, magnifier=None):
    ax.set_title(prompt + "  [press R to reset]", fontsize=11, color="yellow")
    fig.canvas.draw()
    if magnifier:
        magnifier.enable()
    pts = fig.ginput(1, timeout=0)
    if magnifier:
        magnifier.disable()
    if not pts:
        print("Cancelled.")
        sys.exit(0)
    return pts[0]


def collect_calibration(ax, fig, img):
    step = [0]
    mag = Magnifier(ax, fig, img)

    def label():
        step[0] += 1
        return f"[{step[0]}/10]"

    _, nut_y = click(ax, fig, f"{label()} Click anywhere on the NUT line")
    ax.axhline(nut_y, color="cyan", linewidth=0.8, linestyle="--", alpha=0.6)
    fig.canvas.draw()

    _, body_y = click(ax, fig, f"{label()} Click anywhere on the BODY junction line")
    ax.axhline(body_y, color="magenta", linewidth=0.8, linestyle="--", alpha=0.6)
    fig.canvas.draw()

    string_lines = {}
    for s in STRINGS:
        x1, y1 = click(ax, fig, f"{label()} Click on the {s} string near the NUT end", magnifier=mag)
        ax.plot(x1, y1, "r+", markersize=10, markeredgewidth=1.5)
        fig.canvas.draw()

        x2, y2 = click(ax, fig, f"{label()} Click on the {s} string near the BODY end", magnifier=mag)
        ax.plot(x2, y2, "r+", markersize=10, markeredgewidth=1.5)
        ax.plot([x1, x2], [y1, y2], color="white", linewidth=0.5, linestyle=":", alpha=0.4)
        fig.canvas.draw()

        string_lines[s] = ((x1, y1), (x2, y2))

    return nut_y, body_y, string_lines


def x_on_string_at_y(string_line, y):
    """Interpolate the X position along a string line at a given Y."""
    (x1, y1), (x2, y2) = string_line
    if abs(y2 - y1) < 0.001:
        return (x1 + x2) / 2
    t = (y - y1) / (y2 - y1)
    return x1 + t * (x2 - x1)


def compute_table(nut_y, body_y, string_lines):
    px_per_cm = abs(body_y - nut_y) / FINGERBOARD_CM
    direction = 1.0 if body_y > nut_y else -1.0

    rows = []
    for s in STRINGS:
        open_note, open_oct = OPEN_NOTES[s]
        # Limit notes to the bottom calibration click for this string.
        (_, _), (_, bottom_y) = string_lines[s]

        rows.append({
            "string": s,
            "note": note_name(open_note, open_oct, 0),
            "half_steps": 0,
            "distance_cm": 0.0,
            "px_x": round(x_on_string_at_y(string_lines[s], nut_y)),
            "px_y": round(nut_y),
        })

        for n in range(1, 48):
            d_cm = half_step_position_m(n) * 100
            y = nut_y + direction * d_cm * px_per_cm
            if (direction > 0 and y > bottom_y) or (direction < 0 and y < bottom_y):
                break
            x = x_on_string_at_y(string_lines[s], y)
            rows.append({
                "string": s,
                "note": note_name(open_note, open_oct, n),
                "half_steps": n,
                "distance_cm": round(d_cm, 3),
                "px_x": round(x),
                "px_y": round(y),
            })

    return rows


def overlay_notes(ax, fig, rows):
    colours = {"G": "#ff6666", "D": "#66ccff", "A": "#66ff66", "E": "#ffcc66"}
    for r in rows:
        if r["half_steps"] == 0:
            continue
        c = colours[r["string"]]
        ax.plot(r["px_x"], r["px_y"], "o", color=c, markersize=5,
                markeredgecolor="white", markeredgewidth=0.5)
        ax.annotate(
            r["note"],
            (r["px_x"], r["px_y"]),
            textcoords="offset points",
            xytext=(8, 0),
            fontsize=7,
            color=c,
            fontweight="bold",
        )
    ax.set_title("Note positions on fingerboard", fontsize=11, color="yellow")
    fig.canvas.draw()


def save_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["string", "note", "half_steps", "distance_cm", "px_x", "px_y"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved CSV to {path}")


def run_session(img):
    """One calibration + plot session. Returns True to restart, False to quit."""
    fig, ax = plt.subplots(figsize=(8, 12))
    ax.imshow(img)
    ax.axis("off")
    fig.patch.set_facecolor("black")
    fig.tight_layout(pad=0)

    reset_flag = [False]

    def on_key(event):
        if event.key in ("r", "R"):
            reset_flag[0] = True
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.ion()
    plt.show()

    try:
        nut_y, body_y, string_lines = collect_calibration(ax, fig, img)
    except Exception:
        if reset_flag[0]:
            return True
        raise

    if reset_flag[0]:
        return True

    px_per_cm = abs(body_y - nut_y) / FINGERBOARD_CM

    print(f"\nNut Y: {nut_y:.0f} px")
    print(f"Body Y: {body_y:.0f} px")
    print(f"Scale: {px_per_cm:.2f} px/cm")
    for s in STRINGS:
        (x1, y1), (x2, y2) = string_lines[s]
        print(f"  {s} string: ({x1:.0f},{y1:.0f}) -> ({x2:.0f},{y2:.0f})")

    rows = compute_table(nut_y, body_y, string_lines)

    print(f"\n{'String':<8}{'Note':<8}{'Half steps':<12}{'Distance (cm)':<16}{'Pixel (x, y)'}")
    print("-" * 62)
    for r in rows:
        print(f"{r['string']:<8}{r['note']:<8}{r['half_steps']:<12}{r['distance_cm']:<16.3f}({r['px_x']}, {r['px_y']})")

    overlay_notes(ax, fig, rows)

    csv_path = IMAGE_PATH.parent / "note_positions.csv"
    save_csv(rows, csv_path)

    ax.set_title("Done! Press R to reset, or close the window to quit.", fontsize=11, color="yellow")
    fig.canvas.draw()
    plt.ioff()
    plt.show()

    return reset_flag[0]


def main():
    if not IMAGE_PATH.exists():
        print(f"Image not found: {IMAGE_PATH}")
        sys.exit(1)

    img = mpimg.imread(str(IMAGE_PATH))

    print("\n=== Violin Fingerboard Note Calculator ===")
    print("You will click 10 points on the image:")
    print("  1.  Anywhere on the nut line (sets nut Y)")
    print("  2.  Anywhere on the body junction (sets body Y)")
    print("  3-4.  G string: one click near nut, one near body")
    print("  5-6.  D string: one click near nut, one near body")
    print("  7-8.  A string: one click near nut, one near body")
    print("  9-10. E string: one click near nut, one near body")
    print("\nPress R at any time to reset and start over.\n")

    while True:
        restart = run_session(img)
        if restart:
            print("\n--- Resetting... ---\n")
            continue
        break


if __name__ == "__main__":
    main()
