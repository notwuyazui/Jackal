"""Validate model descriptors before opening a browser or a listening socket."""
import json
from pathlib import Path
import re

CATALOG = Path(__file__).with_name('catalog')


def list_models(catalog: Path = CATALOG) -> tuple[str, ...]:
    models = set()
    for path in sorted(Path(catalog).glob('*.json')):
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            for key in ('id', 'adapter'):
                if not isinstance(data[key], str) or not re.fullmatch(r'[a-z][a-z0-9_-]*', data[key]):
                    raise ValueError(f'invalid {key}')
            if data['id'] in models:
                raise ValueError(f"duplicate model id: {data['id']}")
            if not (Path(catalog).parent / 'adapters' / f"{data['adapter']}.ts").is_file():
                raise ValueError(f"model adapter does not exist: {data['adapter']}")
            models.add(data['id'])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'3D 模型描述无效：{path}: {exc}') from exc
    return tuple(sorted(models))
