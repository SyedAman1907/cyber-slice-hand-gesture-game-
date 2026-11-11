"""
Collision Detection Module.
Implements continuous line-segment to circle intersection algorithms
for high-speed slicing cursor trails.
"""

import math
from typing import List, Optional, Tuple
from src.game.fruit import Fruit


def line_segment_circle_intersect(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    circle_center: Tuple[float, float],
    circle_radius: float
) -> bool:
    """
    Tests intersection between a 2D line segment (p1 -> p2) and a circle.
    
    Args:
        p1: Start of line segment (x1, y1).
        p2: End of line segment (x2, y2).
        circle_center: Center of circle (cx, cy).
        circle_radius: Radius of circle.
        
    Returns:
        bool: True if segment touches or intersects circle, False otherwise.
    """
    x1, y1 = p1
    x2, y2 = p2
    cx, cy = circle_center

    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy

    if seg_len_sq == 0.0:
        # Segment is a single point
        dist_sq = (cx - x1) ** 2 + (cy - y1) ** 2
        return dist_sq <= (circle_radius * circle_radius)

    # Project circle center onto segment, clamping projection scalar t to [0, 1]
    t = max(0.0, min(1.0, ((cx - x1) * dx + (cy - y1) * dy) / seg_len_sq))

    # Closest point on the segment to circle center
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy

    dist_sq = (cx - closest_x) ** 2 + (cy - closest_y) ** 2
    return dist_sq <= (circle_radius * circle_radius)


def check_trail_fruit_collision(
    trail_points: List[Tuple[float, float]],
    fruit: Fruit,
    min_slice_speed: float = 4.0,
    active_slicing: bool = True
) -> Tuple[bool, float]:
    """
    Tests if a slicing cursor trail intersects an active fruit hitbox.
    
    Args:
        trail_points: Ordered list of recent cursor screen positions (x, y).
        fruit: Fruit entity to test against.
        min_slice_speed: Minimum movement required for a slice to register.
        active_slicing: Whether slicing action/gesture is currently engaged.
        
    Returns:
        Tuple[bool, float]: (Hit flag, horizontal hit velocity for particle impulse)
    """
    if fruit.is_sliced or not active_slicing:
        return False, 0.0

    if len(trail_points) < 2:
        return False, 0.0

    center = (fruit.x, fruit.y)
    radius = fruit.radius

    # Calculate total stroke distance along trail
    total_dist = 0.0
    for j in range(len(trail_points) - 1):
        total_dist += math.hypot(trail_points[j + 1][0] - trail_points[j][0],
                                 trail_points[j + 1][1] - trail_points[j][1])

    if total_dist < min_slice_speed:
        return False, 0.0

    # Check each consecutive segment in the trail
    for i in range(len(trail_points) - 1):
        p1 = trail_points[i]
        p2 = trail_points[i + 1]

        if line_segment_circle_intersect(p1, p2, center, radius):
            hit_vx = p2[0] - p1[0]
            return True, hit_vx

    return False, 0.0
