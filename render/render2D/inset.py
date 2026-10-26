"""Encode the original Jackal scene without inset-specific overlays."""
import io
import pygame
from render.render2D.pygame_renderer import PygameRenderer


class InsetRenderer(PygameRenderer):
    def __init__(self, width, height):
        super().__init__(width, height, visible=False, draw_sight_range=False,
                         draw_health_bar=True, draw_health_text=False)

    def png(self, snapshot, frame):
        surface = self.draw(snapshot)
        output = io.BytesIO()
        pygame.image.save(surface, output, "frame.png")
        return output.getvalue()
