"""Shared episode outcome rules for evaluation and rendering transports."""
from engine.contracts.render import FrameSnapshot


def outcome_from_counts(counts: dict[str, int], tick: int, max_steps: int) -> dict:
    if counts["PLAYER"] == 0 and counts["ENEMY"] == 0:
        return {"finished": True, "winner": "DRAW", "reason": "mutual_elimination", "alive": counts}
    for losing, winning in (("PLAYER", "ENEMY"), ("ENEMY", "PLAYER")):
        if counts[losing] == 0:
            return {"finished": True, "winner": winning, "reason": "elimination", "alive": counts}
    finished = tick >= max_steps
    return {"finished": finished, "winner": "DRAW" if finished else None,
            "reason": "timeout" if finished else None, "alive": counts}


def battle_result(frame: FrameSnapshot, max_steps: int) -> dict:
    counts = {team: sum(u.alive and u.team == team for u in frame.units)
              for team in ("PLAYER", "ENEMY")}
    return outcome_from_counts(counts, frame.tick, max_steps)
