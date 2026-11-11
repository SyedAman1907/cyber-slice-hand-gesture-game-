"""
Game Fruit & Bomb Entity Module.
Handles physics, parabolic trajectory, slicing mechanics, particle splashes,
and procedural graphic rendering.
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import pygame


@dataclass
class Particle:
    """A visual splash/spark particle."""
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    color: Tuple[int, int, int]
    life: float  # 1.0 down to 0.0
    decay: float

    def update(self, dt: float) -> bool:
        """Updates particle position and life. Returns False when dead."""
        self.x += self.vx * dt * 60.0
        self.y += self.vy * dt * 60.0
        self.vy += 0.3 * dt * 60.0  # gravity
        self.life -= self.decay * dt * 60.0
        return self.life > 0.0

    def draw(self, surface: pygame.Surface) -> None:
        """Draws glowing alpha particle."""
        if self.life <= 0:
            return
        alpha = int(max(0, min(255, self.life * 255)))
        current_rad = max(1, int(self.radius * self.life))
        surf = pygame.Surface((current_rad * 2, current_rad * 2), pygame.SRCALPHA)
        pygame.draw.circle(surf, (*self.color, alpha), (current_rad, current_rad), current_rad)
        surface.blit(surf, (int(self.x - current_rad), int(self.y - current_rad)))


class FruitType:
    WATERMELON = {
        "name": "Watermelon",
        "radius": 46,
        "points": 10,
        "outer_color": (34, 139, 34),     # Forest Green
        "inner_color": (230, 40, 60),     # Vibrant Red
        "splash_color": (255, 60, 80),
        "is_bomb": False
    }
    ORANGE = {
        "name": "Orange",
        "radius": 38,
        "points": 10,
        "outer_color": (255, 140, 0),     # Dark Orange
        "inner_color": (255, 185, 35),    # Light Orange
        "splash_color": (255, 165, 0),
        "is_bomb": False
    }
    APPLE = {
        "name": "Apple",
        "radius": 36,
        "points": 15,
        "outer_color": (205, 30, 30),     # Crimson Red
        "inner_color": (245, 235, 170),   # Pale Cream
        "splash_color": (220, 50, 50),
        "is_bomb": False
    }
    BANANA = {
        "name": "Banana",
        "radius": 34,
        "points": 15,
        "outer_color": (240, 210, 30),    # Bright Yellow
        "inner_color": (255, 250, 205),   # Cream
        "splash_color": (255, 230, 60),
        "is_bomb": False
    }
    STRAWBERRY = {
        "name": "Strawberry",
        "radius": 30,
        "points": 20,
        "outer_color": (225, 25, 60),     # Ruby
        "inner_color": (255, 90, 110),
        "splash_color": (255, 50, 90),
        "is_bomb": False
    }
    BOMB = {
        "name": "Bomb",
        "radius": 40,
        "points": -50,
        "outer_color": (30, 30, 35),      # Charcoal
        "inner_color": (70, 70, 80),      # Metal highlight
        "splash_color": (255, 80, 0),     # Fire orange
        "is_bomb": True
    }

    ALL_FRUITS = [WATERMELON, ORANGE, APPLE, BANANA, STRAWBERRY]


class FruitHalf:
    """Represents a sliced half of a fruit flying away."""

    def __init__(
        self,
        x: float,
        y: float,
        vx: float,
        vy: float,
        radius: float,
        angle: float,
        rot_speed: float,
        side: int,  # -1 for left, +1 for right
        fruit_data: dict
    ):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.angle = angle
        self.rot_speed = rot_speed
        self.side = side
        self.fruit_data = fruit_data
        self.alpha = 255
        self.gravity = 0.45

    def update(self, dt: float) -> bool:
        """Updates physics for the sliced piece. Returns False if offscreen."""
        self.x += self.vx * dt * 60.0
        self.y += self.vy * dt * 60.0
        self.vy += self.gravity * dt * 60.0
        self.angle += self.rot_speed * dt * 60.0
        self.alpha = max(0, self.alpha - int(2.5 * dt * 60.0))
        return self.y < 950 and self.alpha > 0

    def draw(self, surface: pygame.Surface) -> None:
        """Draws semi-circular sliced fruit piece."""
        surf_size = int(self.radius * 2.5)
        surf = pygame.Surface((surf_size, surf_size), pygame.SRCALPHA)
        center = (surf_size // 2, surf_size // 2)

        # Draw half-circle
        rect = pygame.Rect(center[0] - self.radius, center[1] - self.radius,
                           int(self.radius * 2), int(self.radius * 2))
        
        # Draw base
        start_ang = 0 if self.side > 0 else math.pi
        end_ang = math.pi if self.side > 0 else 2 * math.pi
        
        outer = (*self.fruit_data["outer_color"], self.alpha)
        inner = (*self.fruit_data["inner_color"], self.alpha)

        pygame.draw.circle(surf, outer, center, int(self.radius))
        pygame.draw.circle(surf, inner, center, int(self.radius * 0.75))

        # Cut off the other half
        cut_rect = pygame.Rect(
            center[0] if self.side < 0 else 0,
            0,
            surf_size // 2,
            surf_size
        )
        surf.fill((0, 0, 0, 0), cut_rect)

        # Rotate and blit
        rotated = pygame.transform.rotate(surf, self.angle)
        new_rect = rotated.get_rect(center=(int(self.x), int(self.y)))
        surface.blit(rotated, new_rect)


class Fruit:
    """Represents a flying fruit or bomb."""

    def __init__(
        self,
        x: float,
        y: float,
        vx: float,
        vy: float,
        fruit_data: dict,
        gravity: float = 0.40
    ):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.gravity = gravity
        self.fruit_data = fruit_data
        self.radius = fruit_data["radius"]
        self.is_bomb = fruit_data["is_bomb"]

        self.angle = random.uniform(0, 360)
        self.rot_speed = random.uniform(-4.0, 4.0)

        self.is_sliced = False
        self.missed = False
        self.particles: List[Particle] = []
        self.halves: List[FruitHalf] = []
        self.fuse_time = 0.0

    @classmethod
    def spawn(cls, screen_width: int, screen_height: int, bomb_probability: float = 0.15) -> "Fruit":
        """Factory method to spawn a fruit with parabolic launch physics."""
        is_bomb = random.random() < bomb_probability
        data = FruitType.BOMB if is_bomb else random.choice(FruitType.ALL_FRUITS)

        # Spawn along bottom with margins
        margin = 120
        x = random.uniform(margin, screen_width - margin)
        y = screen_height + 40

        # Launch towards center-top
        target_x = screen_width * 0.5 + random.uniform(-250, 250)
        flight_time = random.uniform(55, 75)  # frames

        vx = (target_x - x) / flight_time
        # vy must launch high into top 25-45% of screen
        target_peak_y = random.uniform(screen_height * 0.20, screen_height * 0.42)
        # Using peak formula: h = v^2 / (2g) => v = sqrt(2 * g * h)
        dist_to_peak = y - target_peak_y
        gravity = 0.40
        vy = -math.sqrt(2 * gravity * dist_to_peak)

        return cls(x=x, y=y, vx=vx, vy=vy, fruit_data=data, gravity=gravity)

    def update(self, dt: float, screen_height: int) -> bool:
        """
        Updates physics, particles, and halves.
        Returns False when fruit has completely left play area.
        """
        if not self.is_sliced:
            self.x += self.vx * dt * 60.0
            self.y += self.vy * dt * 60.0
            self.vy += self.gravity * dt * 60.0
            self.angle += self.rot_speed * dt * 60.0
            self.fuse_time += dt

            # Check if fallen below screen without slice
            if self.vy > 0 and self.y > screen_height + 60:
                self.missed = True
                return False
            return True
        else:
            # Update halves and particles
            self.halves = [h for h in self.halves if h.update(dt)]
            self.particles = [p for p in self.particles if p.update(dt)]
            return len(self.halves) > 0 or len(self.particles) > 0

    def slice(self, hit_vx: float = 0.0) -> None:
        """Triggers slicing event, generating split halves and splash particles."""
        if self.is_sliced:
            return
        self.is_sliced = True

        # Generate split halves
        sep_speed = 3.5
        h1 = FruitHalf(
            x=self.x, y=self.y,
            vx=self.vx - sep_speed, vy=self.vy - 1.5,
            radius=self.radius, angle=self.angle,
            rot_speed=-6.0, side=-1, fruit_data=self.fruit_data
        )
        h2 = FruitHalf(
            x=self.x, y=self.y,
            vx=self.vx + sep_speed, vy=self.vy - 1.5,
            radius=self.radius, angle=self.angle,
            rot_speed=6.0, side=1, fruit_data=self.fruit_data
        )
        self.halves = [h1, h2]

        # Generate burst particles
        p_count = 25 if not self.is_bomb else 45
        color = self.fruit_data["splash_color"]
        for _ in range(p_count):
            speed = random.uniform(2.5, 9.0)
            ang = random.uniform(0, 2 * math.pi)
            pvx = math.cos(ang) * speed + hit_vx * 0.2
            pvy = math.sin(ang) * speed
            decay = random.uniform(0.02, 0.045)
            self.particles.append(
                Particle(
                    x=self.x, y=self.y,
                    vx=pvx, vy=pvy,
                    radius=random.uniform(3, 7),
                    color=color if not self.is_bomb else random.choice([(255, 60, 0), (255, 200, 0), (80, 80, 80)]),
                    life=1.0,
                    decay=decay
                )
            )

    def draw(self, surface: pygame.Surface) -> None:
        """Renders the fruit, bomb, split halves, or particles."""
        # Draw particles first
        for p in self.particles:
            p.draw(surface)

        # Draw halves if sliced
        if self.is_sliced:
            for h in self.halves:
                h.draw(surface)
            return

        # Draw intact Fruit or Bomb
        center = (int(self.x), int(self.y))
        r = int(self.radius)

        if self.is_bomb:
            # Bomb body
            pygame.draw.circle(surface, (20, 20, 25), center, r)
            pygame.draw.circle(surface, (50, 50, 60), center, r - 3)
            # Specular highlight
            highlight_pos = (int(self.x - r * 0.3), int(self.y - r * 0.3))
            pygame.draw.circle(surface, (120, 120, 140), highlight_pos, r // 4)

            # Fuse and spark
            fuse_start = (int(self.x), int(self.y - r))
            fuse_end = (int(self.x + 8), int(self.y - r - 16))
            pygame.draw.line(surface, (160, 110, 50), fuse_start, fuse_end, 3)

            # Flickering spark
            spark_color = random.choice([(255, 255, 0), (255, 120, 0), (255, 50, 50)])
            spark_rad = random.randint(4, 8)
            pygame.draw.circle(surface, spark_color, fuse_end, spark_rad)

            # Skull or cross icon
            pygame.draw.line(surface, (200, 50, 50), (center[0] - 8, center[1] - 8), (center[0] + 8, center[1] + 8), 3)
            pygame.draw.line(surface, (200, 50, 50), (center[0] + 8, center[1] - 8), (center[0] - 8, center[1] + 8), 3)
        else:
            # Outer skin
            pygame.draw.circle(surface, self.fruit_data["outer_color"], center, r)
            # Inner flesh
            pygame.draw.circle(surface, self.fruit_data["inner_color"], center, int(r * 0.78))
            # Specular highlight
            highlight_pos = (int(self.x - r * 0.28), int(self.y - r * 0.28))
            pygame.draw.circle(surface, (255, 255, 255, 180), highlight_pos, r // 5)
