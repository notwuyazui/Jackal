"""Legacy recorded demo entry; HTTP transport lives in render."""
import argparse
import json
import math
import threading
from pathlib import Path
from environment.presentation.adapter import STATE_FILE, TRAINING_FILE
from render.api import open_browser_window as open_cot_window
from render.api import create_http_server as LocalDemoServer, create_http_handler as handler_for

def main(argv=None):
    from environment.presentation.session import Demo
    training = json.loads(TRAINING_FILE.read_text())
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--state", type=Path, default=STATE_FILE)
    parser.add_argument("--seed", type=int, default=training["seed"])
    parser.add_argument("--dt", type=float, default=training["env"]["fixed_delta_time"])
    # The training horizon is only 21 s. Evaluation needs time to finish a fight.
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument("--open-browser", action="store_true",
                        help="Open the CoT browser window with Jackal's embedded 2D view")
    args = parser.parse_args(argv)
    if not math.isfinite(args.dt) or args.dt <= 0 or args.max_steps <= 0:
        parser.error("dt and max-steps must be positive")
    demo = Demo(args)
    server = LocalDemoServer(("127.0.0.1", args.port), handler_for(demo))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Jackal authority ready: http://127.0.0.1:{args.port}/api/state", flush=True)
    print(f"Viewer: http://127.0.0.1:5174/?view=3d&token={demo.view_token}", flush=True)
    try:
        if args.open_browser:
            threading.Thread(target=open_cot_window,
                             args=(f"http://127.0.0.1:5174/?view=3d&token={demo.view_token}",), daemon=True).start()
        demo.loop()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        demo.records.close()
        demo.receipts.close()
        demo.renderer.close()


if __name__ == "__main__":
    main()
