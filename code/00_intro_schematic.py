"""
00_intro_schematic.py  -  Original schematic for the Introduction page.

Draws the two crane sites described in the sources cited in the Introduction:
  * Rocky Mountain population route (U.S. Fish & Wildlife Service): winters near
    Bosque del Apache NWR (NM), stops at Monte Vista NWR (CO), nests at Grays Lake NWR (ID).
  * Central Platte River (Nebraska Game & Parks): about a million cranes stage in spring.
It is a diagram, not a map: straight arrows are illustrative and are NOT flight paths.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

import config as C

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
site = {s[0]: s for s in C.SITES}

fig, ax = plt.subplots(figsize=(9, 7.2))
fig.patch.set_facecolor("white"); ax.set_facecolor("white")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID)
ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.tick_params(colors=INK2)
ax.set_xlim(-116, -92); ax.set_ylim(31, 46)
ax.set_xlabel("Longitude (degrees)", color=INK2); ax.set_ylabel("Latitude (degrees north)", color=INK2)

def pt(name):
    _, _, lat, lon, _ = site[name]
    return lon, lat

# Rocky Mountain population: winter -> stopover -> nesting (northbound in spring)
route = ["Bosque del Apache NWR", "Monte Vista NWR", "Grays Lake NWR"]
for a, b in zip(route[:-1], route[1:]):
    ax.add_patch(FancyArrowPatch(pt(a), pt(b), arrowstyle="-|>", mutation_scale=22, lw=2.4,
                                 color=BLUE, shrinkA=10, shrinkB=10, alpha=0.9))
roles = {"Bosque del Apache NWR": ("Winter\nBosque del Apache NWR, NM", (8, -30), "left"),
         "Monte Vista NWR": ("Stopover\nMonte Vista NWR, CO\n(20,000+ cranes each spring and fall)", (12, -6), "left"),
         "Grays Lake NWR": ("Nesting\nGrays Lake NWR, ID", (12, 4), "left")}
for name, (label, off, ha) in roles.items():
    x, y = pt(name)
    ax.scatter([x], [y], s=110, color=BLUE, edgecolor="white", linewidth=2, zorder=4)
    ax.annotate(label, (x, y), xytext=off, textcoords="offset points", ha=ha, va="center",
                fontsize=10.5, color=INK, zorder=5)

ax.text(-109.6, 40.6, "Rocky Mountain population,\nnorthbound in spring", ha="right", va="center",
        fontsize=10, color=BLUE, style="italic")

# Central Platte: spring staging area
x, y = pt("Platte River - Kearney")
ax.scatter([x], [y], s=520, color=ORANGE, alpha=0.25, edgecolor="none", zorder=3)
ax.scatter([x], [y], s=110, color=ORANGE, edgecolor="white", linewidth=2, zorder=4)
ax.annotate("Spring staging\nCentral Platte River, NE\n(about a million cranes,\nmid-February to early April)",
            (x, y), xytext=(16, 22), textcoords="offset points", fontsize=10.5, color=INK, va="bottom")

ax.set_title("Two well-known crane sites in the interior West and Great Plains", loc="left",
             fontsize=14, fontweight="bold", color=INK)
ax.text(-115.6, 31.35, "Schematic for orientation only. Arrows are illustrative, not flight paths.\n"
        "Sources: U.S. Fish & Wildlife Service; Nebraska Game & Parks Commission.",
        fontsize=8.5, color=INK2, va="bottom")
fig.tight_layout()
out = C.ROOT / "images" / "intro_flyway_schematic.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
