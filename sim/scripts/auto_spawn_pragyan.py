#!/usr/bin/env python3

from pathlib import Path
import math
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image


# ============================================================
# PATHS
# ============================================================

PROJECT = Path.home() / "Desktop/Project/SIH"

WORLD_FILE = PROJECT / "sim/worlds/lunar_terrain.world"
HEIGHTMAP = PROJECT / "sim/worlds/lunar_heightmap.png"
MODEL_FILE = PROJECT / "sim/models/pragyan_rover/model.sdf"


# ============================================================
# TERRAIN PARAMETERS
# Taken directly from lunar_terrain.world
# ============================================================

TERRAIN_SIZE_X = 120.0
TERRAIN_SIZE_Y = 120.0
TERRAIN_HEIGHT = 10.0
TERRAIN_Z_OFFSET = -5.0


# ============================================================
# ROVER SAFE-SPAWN PARAMETERS
# ============================================================

# Approximate Pragyan footprint.
# We intentionally search a little larger than the physical rover.
ROVER_LENGTH = 1.10
ROVER_WIDTH = 0.95

# Search grid spacing.
SEARCH_STEP = 1.0

# Maximum acceptable terrain slope for initial spawn.
MAX_SLOPE_DEG = 10.0

# Maximum height variation inside the rover footprint.
MAX_HEIGHT_VARIATION = 0.25

# Additional clearance above terrain.
GROUND_CLEARANCE = 0.08


# ============================================================
# LOAD HEIGHTMAP
# ============================================================

if not HEIGHTMAP.exists():
    raise FileNotFoundError(f"Heightmap not found: {HEIGHTMAP}")

if not WORLD_FILE.exists():
    raise FileNotFoundError(f"World not found: {WORLD_FILE}")

img = Image.open(HEIGHTMAP).convert("L")
heightmap = np.asarray(img, dtype=np.float64) / 255.0

H, W = heightmap.shape

print()
print("==============================================")
print(" PRAGYAN TERRAIN-AWARE AUTO SPAWN")
print("==============================================")
print(f"Heightmap resolution : {W} x {H}")
print(f"Terrain size         : {TERRAIN_SIZE_X} x {TERRAIN_SIZE_Y} m")
print(f"Terrain height range : {TERRAIN_Z_OFFSET} to "
      f"{TERRAIN_Z_OFFSET + TERRAIN_HEIGHT} m")
print()


# ============================================================
# HEIGHTMAP -> WORLD COORDINATE
# ============================================================

def world_to_pixel(x, y):
    """
    Gazebo terrain is centered around x=0,y=0.

    Convert world coordinates into heightmap pixel coordinates.
    """

    u = ((x + TERRAIN_SIZE_X / 2.0) / TERRAIN_SIZE_X) * (W - 1)

    # Image row direction is reversed relative to world Y.
    v = ((TERRAIN_SIZE_Y / 2.0 - y) / TERRAIN_SIZE_Y) * (H - 1)

    return u, v


def terrain_height(x, y):
    u, v = world_to_pixel(x, y)

    u = np.clip(u, 0, W - 1)
    v = np.clip(v, 0, H - 1)

    x0 = int(math.floor(u))
    x1 = min(x0 + 1, W - 1)

    y0 = int(math.floor(v))
    y1 = min(y0 + 1, H - 1)

    fx = u - x0
    fy = v - y0

    a = heightmap[y0, x0]
    b = heightmap[y0, x1]
    c = heightmap[y1, x0]
    d = heightmap[y1, x1]

    value = (
        a * (1 - fx) * (1 - fy)
        + b * fx * (1 - fy)
        + c * (1 - fx) * fy
        + d * fx * fy
    )

    return TERRAIN_Z_OFFSET + value * TERRAIN_HEIGHT


# ============================================================
# LOCAL TERRAIN ANALYSIS
# ============================================================

def analyze_patch(cx, cy):
    """
    Analyse terrain under the rover footprint.
    """

    xs = np.linspace(
        cx - ROVER_LENGTH / 2,
        cx + ROVER_LENGTH / 2,
        9
    )

    ys = np.linspace(
        cy - ROVER_WIDTH / 2,
        cy + ROVER_WIDTH / 2,
        9
    )

    points = []

    for x in xs:
        for y in ys:
            points.append((x, y, terrain_height(x, y)))

    points = np.asarray(points)

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Fit a local plane:
    #
    # z = ax + by + c
    A = np.column_stack((x, y, np.ones_like(x)))

    coeff, _, _, _ = np.linalg.lstsq(A, z, rcond=None)

    a, b, c = coeff

    slope_x = math.atan(a)
    slope_y = math.atan(b)

    slope = math.sqrt(slope_x ** 2 + slope_y ** 2)
    slope_deg = math.degrees(slope)

    height_variation = float(np.max(z) - np.min(z))

    # How much the terrain deviates from the fitted plane.
    plane = a * x + b * y + c
    roughness = float(np.std(z - plane))

    return {
        "center_height": terrain_height(cx, cy),
        "slope_deg": slope_deg,
        "height_variation": height_variation,
        "roughness": roughness,
        "slope_x": slope_x,
        "slope_y": slope_y,
    }


# ============================================================
# SEARCH FOR SAFE SPAWN LOCATION
# ============================================================

best = None

# Keep rover away from extreme edges.
margin = 5.0

xs = np.arange(
    -TERRAIN_SIZE_X / 2 + margin,
    TERRAIN_SIZE_X / 2 - margin,
    SEARCH_STEP
)

ys = np.arange(
    -TERRAIN_SIZE_Y / 2 + margin,
    TERRAIN_SIZE_Y / 2 - margin,
    SEARCH_STEP
)

print("Searching terrain for a safe rover starting location...")
print()

for x in xs:
    for y in ys:

        try:
            result = analyze_patch(x, y)
        except Exception:
            continue

        if result["slope_deg"] > MAX_SLOPE_DEG:
            continue

        if result["height_variation"] > MAX_HEIGHT_VARIATION:
            continue

        # Score:
        # flatter + smoother = better.
        score = (
            result["slope_deg"] * 2.0
            + result["height_variation"] * 4.0
            + result["roughness"] * 10.0
        )

        if best is None or score < best["score"]:

            best = {
                "x": float(x),
                "y": float(y),
                "score": float(score),
                **result,
            }


if best is None:
    raise RuntimeError(
        "No safe spawn location was found. "
        "Increase MAX_SLOPE_DEG or MAX_HEIGHT_VARIATION."
    )


# ============================================================
# CALCULATE ROVER POSE
# ============================================================

x = best["x"]
y = best["y"]

terrain_z = best["center_height"]

# The current rover base_link is above the wheel/contact level.
# Start slightly above the terrain so Gazebo settles naturally.
spawn_z = terrain_z + 0.65 + GROUND_CLEARANCE

# Align rover with terrain slope.
#
# Positive terrain slope in X -> pitch.
# Positive terrain slope in Y -> roll.
pitch = best["slope_x"]
roll = -best["slope_y"]

# Face approximately along +X.
yaw = 0.0


# ============================================================
# MODIFY WORLD
# ============================================================

tree = ET.parse(WORLD_FILE)
root = tree.getroot()

world = root.find("world")

if world is None:
    raise RuntimeError("Could not find <world> in lunar_terrain.world")


# Remove any existing Pragyan include.
for include in list(world.findall("include")):

    uri = include.find("uri")

    if uri is not None and uri.text == "model://pragyan_rover":
        world.remove(include)


# Add new automatically calculated pose.
include = ET.Element("include")

uri = ET.SubElement(include, "uri")
uri.text = "model://pragyan_rover"

pose = ET.SubElement(include, "pose")

pose.text = (
    f"{x:.3f} "
    f"{y:.3f} "
    f"{spawn_z:.3f} "
    f"{roll:.5f} "
    f"{pitch:.5f} "
    f"{yaw:.5f}"
)

world.append(include)


# ============================================================
# WRITE WORLD
# ============================================================

tree.write(
    WORLD_FILE,
    encoding="utf-8",
    xml_declaration=True
)


# ============================================================
# RESULT
# ============================================================

print("==============================================")
print(" SAFE SPAWN FOUND")
print("==============================================")
print(f"X                 : {x:.3f} m")
print(f"Y                 : {y:.3f} m")
print(f"Terrain Z         : {terrain_z:.3f} m")
print(f"Spawn Z           : {spawn_z:.3f} m")
print(f"Roll              : {math.degrees(roll):.2f} deg")
print(f"Pitch             : {math.degrees(pitch):.2f} deg")
print(f"Slope             : {best['slope_deg']:.2f} deg")
print(f"Height variation  : {best['height_variation']:.3f} m")
print(f"Terrain roughness : {best['roughness']:.3f} m")
print()
print("lunar_terrain.world has been updated.")
print("Pragyan will now spawn at this terrain-aware location.")
print("==============================================")
