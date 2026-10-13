"""Pygame adapter for drawing a battle world without polluting game entities."""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import pygame

import render.render2D.options as GameMode
from render.render2D.options import EXPLOSION_IMAGE_ADAPT_TO_RANGE
from render.render2D.camera import Camera2D
from render.render2D.assets import ASSET_ROOT, TILES, unit_sprites, projectile_sprite
from engine.contracts.render import BattleManifest, FrameSnapshot

if TYPE_CHECKING:
    from engine.contracts.render import FrameSnapshot


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
        draw_sight_range: bool | None = None,
        draw_health_bar: bool | None = None,
    ) -> None:
        pygame.font.init()
        if visible:
            pygame.display.init()
        self.visible = visible
        self.draw_sight_range = draw_sight_range
        self.draw_health_bar = draw_health_bar
        self.viewport_width = int(width)
        self.viewport_height = int(height)
        if self.viewport_width <= 0 or self.viewport_height <= 0:
            raise ValueError("viewport dimensions must be > 0")
        self.surface = (
            pygame.display.set_mode((self.viewport_width, self.viewport_height))
            if visible
            else pygame.Surface((self.viewport_width, self.viewport_height))
        )
        if visible:
            pygame.display.set_caption(title)
        self.camera = Camera2D((self.viewport_width, self.viewport_height))
        self._images: dict[str, pygame.Surface | None] = {}
        self._tile_images: dict[tuple[str, int], pygame.Surface] = {}
        self._health_font: pygame.font.Font | None = None

    def _load_image(self, image_path: str | None) -> pygame.Surface | None:
        if not image_path:
            return None
        if image_path in self._images:
            return self._images[image_path]

        candidates = [str(ASSET_ROOT / image_path)]
        if os.path.isabs(image_path):
            candidates.extend([
                os.path.join(_PROJECT_ROOT, image_path),
                str(ASSET_ROOT / image_path),
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
        frame: FrameSnapshot,
        *,
        mouse_pos: tuple[float, float] | None = None,
        background: tuple[int, int, int] = (50, 50, 70),
    ) -> pygame.Surface:
        self.surface.fill(background)
        self.camera.world_size = frame.map.size
        offset = self.camera.clamp()
        self._draw_map(frame, offset)
        for unit in frame.units:
            self._draw_unit(
                unit,
                offset,
                mouse_pos,
                frame.rules.tear_drop_vision,
            )
        for bullet in frame.bullets:
            self._draw_bullet(bullet, offset)
        return self.surface

    def initialize(self, manifest: BattleManifest) -> None:
        self.camera.world_size = manifest.map.size
        self.camera.offset[:] = [0.0, 0.0]
        self.camera.clamp()

    def render(self, frame: FrameSnapshot):
        surface = self.draw(frame)
        self.present()
        return surface

    def present(self) -> None:
        if self.visible:
            pygame.display.flip()

    def rgb_array(self):
        """Return an HWC RGB copy suitable for video encoders."""

        return pygame.surfarray.array3d(self.surface).transpose(1, 0, 2)

    def close(self) -> None:
        self._images.clear()
        self._tile_images.clear()
        if self.visible:
            pygame.display.quit()

    def _draw_map(self, frame: FrameSnapshot, offset) -> None:
        for row_index, row in enumerate(frame.map.tiles):
            for col, letter in enumerate(row):
                image = self._load_image(TILES.get(letter))
                if image is not None:
                    size = frame.map.tile_size
                    if image.get_size() != (size, size):
                        key = (letter, size)
                        if key not in self._tile_images:
                            self._tile_images[key] = pygame.transform.smoothscale(image, (size, size))
                        image = self._tile_images[key]
                    self.surface.blit(image, (col * frame.map.tile_size - offset[0],
                                              row_index * frame.map.tile_size - offset[1]))

        if GameMode.DRAW_OBSTACLE_BOUNDING_BOX or GameMode.DEBUG_MODE:
            for obstacle in frame.map.unit_obstacles:
                pygame.draw.rect(
                    self.surface,
                    (255, 0, 0),
                    pygame.Rect(obstacle).move(-offset[0], -offset[1]),
                    2,
                )
            for obstacle in frame.map.bullet_obstacles:
                pygame.draw.rect(
                    self.surface,
                    (0, 255, 0),
                    pygame.Rect(obstacle).move(-offset[0], -offset[1]),
                    1,
                )

    def _draw_unit(
        self,
        unit,
        offset,
        mouse_pos,
        use_tear_drop_vision: bool,
    ) -> None:
        if not unit.alive or not unit.visible:
            return
        x = unit.position[0] - offset[0]
        y = unit.position[1] - offset[1]
        for image_path, angle in zip(unit_sprites(unit.unit_type, unit.team),
                                     (unit.heading, unit.turret_heading)):
            image = self._load_image(image_path)
            if image is not None:
                rotated = pygame.transform.rotate(image, -angle)
                self.surface.blit(rotated, rotated.get_rect(center=(x, y)))

        show_health = GameMode.DRAW_HEALTH_BAR if self.draw_health_bar is None else self.draw_health_bar
        show_sight = GameMode.DRAW_SIGHT_RANGE if self.draw_sight_range is None else self.draw_sight_range
        if show_health or GameMode.DEBUG_MODE:
            self._draw_health_bar(unit, x, y)
        if show_sight or GameMode.DEBUG_MODE:
            self._draw_sight_range(unit, offset, use_tear_drop_vision)
        if GameMode.DRAW_ATTACK_RANGE or GameMode.DEBUG_MODE:
            self._draw_attack_range(unit, offset)
        if (
            unit.id == 0
            and mouse_pos is not None
            and (GameMode.DRAW_MOUSE_TARGET_LINE or GameMode.DEBUG_MODE)
        ):
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

    def _draw_sight_range(
        self,
        unit,
        offset,
        use_tear_drop_vision: bool,
    ) -> None:
        x = unit.position[0] - offset[0]
        y = unit.position[1] - offset[1]
        if not use_tear_drop_vision:
            pygame.draw.circle(
                self.surface, (0, 0, 0), (int(x), int(y)), int(unit.sight_range), 1
            )
            return

        forward_angle = math.radians(unit.heading - 90)
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

    def _draw_attack_range(self, unit, offset) -> None:
        attack_range = unit.attack_range
        if attack_range <= 0.0:
            return
        x = unit.position[0] - offset[0]
        y = unit.position[1] - offset[1]
        pygame.draw.circle(
            self.surface,
            (255, 0, 0),
            (int(x), int(y)),
            int(attack_range),
            1,
        )

    def _draw_bullet(self, bullet, offset) -> None:
        x = bullet.position[0] - offset[0]
        y = bullet.position[1] - offset[1]
        image_path = projectile_sprite(bullet.ammunition, bullet.has_exploded)
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
            GameMode.DEBUG_MODE or GameMode.DRAW_BULLET_EXPLOSION_RANGE
        ):
            pygame.draw.circle(
                self.surface,
                (255, 100, 100, 128),
                (int(x), int(y)),
                int(bullet.explosion_radius),
                2,
            )
        if (
            GameMode.DEBUG_MODE or GameMode.DRAW_BULLET_BOUNDING_BOX
        ) and bullet.hitbox:
            pygame.draw.rect(
                self.surface,
                (255, 0, 0),
                pygame.Rect(bullet.hitbox).move(-offset[0], -offset[1]),
                1,
            )
