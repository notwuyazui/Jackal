"""Pygame adapter for drawing a battle world without polluting game entities."""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import pygame

from game.GameMode import (
    DEBUG_MODE,
    DRAW_BULLET_BOUNDING_BOX,
    DRAW_BULLET_EXPLOSION_RANGE,
    DRAW_HEALTH_BAR,
    DRAW_MOUSE_TARGET_LINE,
    DRAW_OBSTACLE_BOUNDING_BOX,
    DRAW_SIGHT_RANGE,
    USE_TEAR_DROP_VISION,
)
from game.Parameter import EXPLOSION_IMAGE_ADAPT_TO_RANGE

if TYPE_CHECKING:
    from game.BattleWorld import BattleWorld


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


class PygameRenderer:
    """Own all Pygame surfaces, textures, fonts, and draw operations."""

    def __init__(
        self,
        width: int,
        height: int,
        *,
        visible: bool = False,
        title: str = "Jackal",
    ) -> None:
        pygame.init()
        self.visible = visible
        self.surface = (
            pygame.display.set_mode((width, height))
            if visible
            else pygame.Surface((width, height))
        )
        if visible:
            pygame.display.set_caption(title)
        self._images: dict[str, pygame.Surface | None] = {}
        self._health_font: pygame.font.Font | None = None

    def _load_image(self, image_path: str | None) -> pygame.Surface | None:
        if not image_path:
            return None
        if image_path in self._images:
            return self._images[image_path]

        candidates = [image_path]
        if not os.path.isabs(image_path):
            candidates.extend([
                os.path.join(_PROJECT_ROOT, image_path),
                os.path.join(_PROJECT_ROOT, "game", image_path),
            ])
        image = None
        for path in candidates:
            if not path or not os.path.exists(path):
                continue
            try:
                image = pygame.image.load(path)
                break
            except pygame.error:
                continue
        self._images[image_path] = image
        return image

    def draw(
        self,
        world: BattleWorld,
        *,
        mouse_pos: tuple[float, float] | None = None,
        background: tuple[int, int, int] = (50, 50, 70),
    ) -> pygame.Surface:
        self.surface.fill(background)
        offset = world.get_camera_offset()
        self._draw_map(world, offset)
        for unit in world.iter_units():
            self._draw_unit(unit, offset, mouse_pos)
        for bullet in world.iter_bullets():
            self._draw_bullet(bullet, offset)
        return self.surface

    def present(self) -> None:
        if self.visible:
            pygame.display.flip()

    def rgb_array(self):
        """Return an HWC RGB copy suitable for video encoders."""

        return pygame.surfarray.array3d(self.surface).transpose(1, 0, 2)

    def close(self) -> None:
        self._images.clear()
        if self.visible:
            pygame.display.quit()

    def _draw_map(self, world: BattleWorld, offset) -> None:
        for row in world.iter_map_tiles():
            for tile in row:
                image = self._load_image(tile.image_path)
                if image is not None:
                    self.surface.blit(image, (tile.x - offset[0], tile.y - offset[1]))

        if DRAW_OBSTACLE_BOUNDING_BOX or DEBUG_MODE:
            for obstacle in world.get_unit_obstacles():
                pygame.draw.rect(
                    self.surface,
                    (255, 0, 0),
                    obstacle.move(-offset[0], -offset[1]),
                    2,
                )
            for obstacle in world.get_bullet_obstacles():
                pygame.draw.rect(
                    self.surface,
                    (0, 255, 0),
                    obstacle.move(-offset[0], -offset[1]),
                    1,
                )

    def _draw_unit(self, unit, offset, mouse_pos) -> None:
        if not unit.is_alive or not unit.visible:
            return
        x = unit.position[0] - offset[0]
        y = unit.position[1] - offset[1]
        for image_path, angle in (
            (unit.body_image_path, unit.direction_angle),
            (unit.turret_image_path, unit.turret_direction_angle),
        ):
            image = self._load_image(image_path)
            if image is not None:
                rotated = pygame.transform.rotate(image, -angle)
                self.surface.blit(rotated, rotated.get_rect(center=(x, y)))

        if DRAW_HEALTH_BAR or DEBUG_MODE:
            self._draw_health_bar(unit, x, y)
        if DRAW_SIGHT_RANGE or DEBUG_MODE:
            self._draw_sight_range(unit, offset)
        if mouse_pos is not None and (DRAW_MOUSE_TARGET_LINE or DEBUG_MODE):
            pygame.draw.line(self.surface, (255, 0, 255), (x, y), mouse_pos, 1)
            pygame.draw.circle(
                self.surface,
                (255, 0, 255),
                (int(mouse_pos[0]), int(mouse_pos[1])),
                3,
            )

    def _draw_health_bar(self, unit, x: float, y: float) -> None:
        width, height = 40, 8
        bar_x = x - width / 2
        bar_y = y - unit.size[1] / 2 - 15
        pygame.draw.rect(self.surface, (255, 0, 0), (bar_x, bar_y, width, height))
        pygame.draw.rect(
            self.surface,
            (0, 255, 0),
            (bar_x, bar_y, width * unit.health / unit.max_health, height),
        )
        pygame.draw.rect(
            self.surface,
            (255, 255, 255),
            (bar_x, bar_y, width, height),
            1,
        )
        if self._health_font is None:
            self._health_font = pygame.font.Font(None, 12)
        text = self._health_font.render(
            f"{int(unit.health)}/{int(unit.max_health)}", True, (0, 0, 0)
        )
        self.surface.blit(text, text.get_rect(center=(x, bar_y + height / 2)))

    def _draw_sight_range(self, unit, offset) -> None:
        x = unit.position[0] - offset[0]
        y = unit.position[1] - offset[1]
        if not USE_TEAR_DROP_VISION:
            pygame.draw.circle(
                self.surface, (0, 0, 0), (int(x), int(y)), int(unit.sight_range), 1
            )
            return

        forward_angle = math.radians(unit.direction_angle - 90)
        forward_x, forward_y = math.cos(forward_angle), math.sin(forward_angle)
        a = (unit.sight_range + unit.min_sight_range) / 2
        b = (unit.sight_range - unit.min_sight_range) / 2
        points = []
        for index in range(61):
            theta = 2 * math.pi * index / 60
            radius = a + b * math.cos(theta)
            direction_x = forward_x * math.cos(theta) - forward_y * math.sin(theta)
            direction_y = forward_x * math.sin(theta) + forward_y * math.cos(theta)
            points.append((x + radius * direction_x, y + radius * direction_y))
        pygame.draw.polygon(self.surface, (0, 0, 0), points, 1)

    def _draw_bullet(self, bullet, offset) -> None:
        if not bullet.is_active:
            return
        x = bullet.position[0] - offset[0]
        y = bullet.position[1] - offset[1]
        image_path = (
            bullet.explosion_image_path
            if bullet.has_exploded and bullet.explosion_image_path
            else bullet.image_path
        )
        image = self._load_image(image_path)
        if image is not None:
            if (
                bullet.has_exploded
                and bullet.explosion_radius > 0
                and EXPLOSION_IMAGE_ADAPT_TO_RANGE
            ):
                diameter = int(bullet.explosion_radius * 2)
                image = pygame.transform.scale(image, (diameter, diameter))
            rotated = pygame.transform.rotate(image, -bullet.rotation_angle)
            self.surface.blit(rotated, rotated.get_rect(center=(x, y)))

        if bullet.has_exploded and bullet.is_explosive and (
            DEBUG_MODE or DRAW_BULLET_EXPLOSION_RANGE
        ):
            pygame.draw.circle(
                self.surface,
                (255, 100, 100, 128),
                (int(x), int(y)),
                int(bullet.explosion_radius),
                2,
            )
        if (DEBUG_MODE or DRAW_BULLET_BOUNDING_BOX) and bullet.bounding_box:
            pygame.draw.rect(
                self.surface,
                (255, 0, 0),
                bullet.bounding_box.move(-offset[0], -offset[1]),
                1,
            )
