"""Camera and screen-to-world conversion are presentation state."""
from dataclasses import dataclass, field
from engine.parameters import Direction


@dataclass
class Camera2D:
    viewport: tuple[float, float]
    world_size: tuple[float, float] = (960, 640)
    speed: float = 5.0
    offset: list[float] = field(default_factory=lambda: [0.0, 0.0])

    def clamp(self):
        for i in (0, 1):
            self.offset[i] = min(max(0.0, self.offset[i]),
                                 max(0.0, self.world_size[i] - self.viewport[i]))
        return tuple(self.offset)

    def move(self, direction: Direction):
        dx, dy = {Direction.UP: (0, -1), Direction.DOWN: (0, 1),
                  Direction.LEFT: (-1, 0), Direction.RIGHT: (1, 0)}[direction]
        self.offset[0] += dx * self.speed
        self.offset[1] += dy * self.speed
        self.clamp()

    def to_world(self, screen_position):
        self.clamp()
        return tuple(screen_position[i] + self.offset[i] for i in (0, 1))
