"""Browser window presentation helpers."""
from dataclasses import dataclass
import sys
import subprocess
import webbrowser


@dataclass(frozen=True)
class SafariWindow:
    identifier: int

    def is_open(self):
        """Inspect only this window; script errors leave its state unknown."""
        script = '''on run argv
if application "Safari" is not running then return false
tell application "Safari"
    return exists (first window whose id is (item 1 of argv as integer))
end tell
end run'''
        try:
            result = subprocess.run(["osascript", "-e", script, str(self.identifier)],
                                    check=True, timeout=3, capture_output=True, text=True)
            state = result.stdout.strip().lower()
            return {'true': True, 'false': False}.get(state)
        except (subprocess.SubprocessError, OSError):
            return None


def open_cot_window(url):
    """Open a separate Safari window without replacing an existing page."""
    try:
        if sys.platform == "darwin":
            script = '''on run argv
tell application "Safari"
    set targetURL to item 1 of argv
    make new document with properties {URL:targetURL}
    set battleWindowID to id of front window
    activate
    return battleWindowID
end tell
end run'''
            result = subprocess.run(["osascript", "-e", script, url], check=True, timeout=20,
                                    capture_output=True, text=True)
            return SafariWindow(int(result.stdout.strip()))
        else:
            webbrowser.open(url, new=1, autoraise=True)
    except (subprocess.SubprocessError, OSError, ValueError) as error:
        # A locked/unresponsive desktop must not close the simulation.
        print(f"Browser launch unavailable ({type(error).__name__}). Open {url}", flush=True)
