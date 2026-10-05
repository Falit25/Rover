#!/usr/bin/env python3
"""Create a simple crater heightmap PNG for Gazebo."""
import numpy as np
from PIL import Image

# Create 256x256 heightmap
size = 256
heightmap = np.zeros((size, size), dtype=np.uint8)

center = size // 2
radius = size // 3

for y in range(size):
    for x in range(size):
        dx = x - center
        dy = y - center
        dist = np.sqrt(dx*dx + dy*dy)
        
        if dist < radius:
            # Crater profile: rim high, floor low
            normalized = dist / radius
            # Rim at edge
            if normalized > 0.7:
                height = int(255 * (normalized - 0.7) / 0.3 * 0.3 + 255 * 0.7)
            # Floor
            elif normalized < 0.3:
                height = int(255 * 0.2)
            # Slope
            else:
                height = int(255 * (0.2 + 0.5 * (normalized - 0.3) / 0.4))
            heightmap[y, x] = height
        else:
            heightmap[y, x] = int(255 * 0.5)  # Flat terrain

img = Image.fromarray(heightmap, mode='L')
img.save('/home/blueberry/Desktop/Project/SIH/crater.png')
print("Created crater.png")
