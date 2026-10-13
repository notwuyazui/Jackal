"""Regression evidence captured before moving any implementation files."""
import dataclasses
import hashlib
import json
import random
import unittest
from pathlib import Path
from environment import JackalEnv
from environment.presentation.adapter import build_world, export_frame, step_world

FIXTURE = json.loads((Path(__file__).resolve().parents[1]/'fixtures/pre_migration.json').read_text())
UNIT_KEYS = ('id','team','position','heading','turret_heading','speed','angular_speed',
             'acceleration','health','max_health','alive','cooldown','cooldown_duration',
             'hitbox','collision_box','size','visible_ids','kills','damage_dealt')
BULLET_KEYS = ('id','source_id','team','position','velocity','size','lifetime')
EVENT_KEYS = {'fire': ('kind','bullet_id','source_id','position','velocity'),
              'impact': ('kind','bullet_id','source_id','position','target_ids','surface'),
              'damage': ('kind','source_id','target_id','amount'),
              'destroyed': ('kind','source_id','target_id','amount')}


def digest(value):
    def encode(x):
        if hasattr(x,'tolist'): return x.tolist()
        if dataclasses.is_dataclass(x): return dataclasses.asdict(x)
        if isinstance(x,(set,frozenset)): return sorted(x)
        raise TypeError(type(x).__name__)
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',', ':'),
                                    default=encode,allow_nan=False).encode()).hexdigest()


class MigrationBaselineTests(unittest.TestCase):
    def test_every_physics_frame_matches_before_migration(self):
        world = build_world(seed=FIXTURE['seed'])
        for tick, expected in enumerate(FIXTURE['world_hashes']):
            frame = export_frame(world, FIXTURE['max_steps'])
            original = {'tick':frame['tick'],'time':frame['time'],
                        'units':[{k:u[k] for k in UNIT_KEYS} for u in frame['units']],
                        'bullets':[{k:b[k] for k in BULLET_KEYS} for b in frame['bullets']],
                        'events':[{k:e[k] for k in EVENT_KEYS[e['kind']]} for e in frame['events']],
                        'result':frame['result']}
            self.assertEqual(digest(original)[:16],expected,f'Physics diverged at tick {tick}')
            if tick+1 < len(FIXTURE['world_hashes']):
                step_world(world,FIXTURE['dt'])
        self.assertEqual(frame['result'],FIXTURE['result'])

    def test_observations_rewards_masks_and_termination_match(self):
        env = JackalEnv(headless=True,n_agents=3,n_enemies=3,
                        game_state_file='border_map_3t_vs_3t',
                        max_steps=FIXTURE['environment_max_steps'])
        try:
            rng=random.Random(FIXTURE['action_seed'])
            obs,state=env.reset(seed=FIXTURE['seed'])
            self.assertEqual(digest({'obs':obs,'state':state,'mask':env.get_avail_actions()}),
                             FIXTURE['environment_hashes'][0])
            for expected in FIXTURE['environment_hashes'][1:]:
                actions=[rng.choice([i for i,a in enumerate(mask) if a])
                         for mask in env.get_avail_actions()]
                obs,state,reward,done,info=env.step(actions)
                self.assertEqual(digest({'obs':obs,'state':state,'reward':reward,
                                         'done':done,'info':info,'mask':env.get_avail_actions()}),expected)
        finally:
            env.close()
