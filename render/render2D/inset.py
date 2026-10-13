"""Original Jackal Pygame surface, with the existing inset HUD and contact FX."""
import io
import pygame
from render.render2D.pygame_renderer import PygameRenderer

class InsetRenderer(PygameRenderer):
    def __init__(self, width, height):
        super().__init__(width, height, visible=False, draw_sight_range=False, draw_health_bar=True)
        self.font = pygame.font.Font(None, 16)
        self.hud_font = pygame.font.Font(None, 22)
        self.fx = []

    def png(self, snapshot, frame):
        surface = self.draw(snapshot)
        self.fx.extend({**e, "time": frame["time"]} for e in frame["events"] if e["kind"] == "impact")
        self.fx = [e for e in self.fx if frame["time"] - e["time"] < 0.24]
        for event in self.fx:
            age = frame["time"] - event["time"]
            radius = max(2, int(3 + 8 * age / 0.24))
            pygame.draw.circle(surface, (255, 190, 65), event["position"], radius, 2)
        for unit in frame["units"]:
            x, y = unit["position"]
            if unit["id"] == 2:
                pygame.draw.circle(surface, (255, 205, 90), (round(x), round(y)), 21, 2)
            color = (115, 220, 255) if unit["team"] == "PLAYER" else (255, 145, 125)
            if not unit["alive"]:
                # Jackal removes dead sprites; the marker preserves the followed ID.
                pygame.draw.line(surface, (140, 140, 140), (x-7, y-7), (x+7, y+7), 2)
                pygame.draw.line(surface, (140, 140, 140), (x-7, y+7), (x+7, y-7), 2)
            label = self.font.render(str(unit["id"]), True, color)
            surface.blit(label, (x+14, y+5))
        # Overlay status in the existing border tiles; the original renderer's
        # 960x640 viewport, sprite positions and map projection stay intact.
        pygame.draw.rect(surface, (18, 28, 34), (0, 0, surface.get_width(), 57))
        alive = frame["result"]["alive"]
        state = "WAITING FOR 3D" if frame["tick"] == 0 else "RUNNING"
        if frame["result"]["finished"]:
            state = {"ENEMY": "RED WINS", "PLAYER": "BLUE WINS", "DRAW": "DRAW"}[frame["result"]["winner"]]
        line = f"Jackal 2D   {frame['time']:.2f} s   STEP {frame['tick']}   Blue {alive['PLAYER']} : {alive['ENEMY']} Red   {state}"
        surface.blit(self.hud_font.render(line, True, (225, 233, 238)), (14, 8))
        for index, unit in enumerate(frame["units"]):
            text = f"#{unit['id']}  {unit['health'] if unit['alive'] else 0}/{unit['max_health']} HP"
            color = (115, 220, 255) if unit["team"] == "PLAYER" else (255, 145, 125)
            surface.blit(self.font.render(text, True, color), (14 + index * 155, 35))
        output = io.BytesIO()
        pygame.image.save(surface, output, "frame.png")
        return output.getvalue()
