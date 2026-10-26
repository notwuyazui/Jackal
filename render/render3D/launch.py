"""Browser window presentation helpers."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import sys
import subprocess
import tempfile
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


@dataclass
class BrowserProcessWindow:
    """A dedicated Chromium process; no existing browser profile is modified."""
    process: subprocess.Popen
    profile: tempfile.TemporaryDirectory

    def is_open(self):
        return self.process.poll() is None

    def close(self):
        try:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=2)
        finally:
            self.profile.cleanup()


def _windows_browser():
    for executable, relative in (
        ("msedge.exe", "Microsoft/Edge/Application/msedge.exe"),
        ("chrome.exe", "Google/Chrome/Application/chrome.exe"),
    ):
        found = shutil.which(executable)
        if found:
            return found
        try:
            import winreg
            for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
                try:
                    with winreg.OpenKey(hive, "SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\App Paths\\" + executable) as key:
                        value, _ = winreg.QueryValueEx(key, None)
                        if Path(value).is_file():
                            return value
                except OSError:
                    pass
        except ImportError:
            pass
        for variable in ("PROGRAMFILES(X86)", "PROGRAMFILES", "LOCALAPPDATA"):
            directory = os.environ.get(variable)
            if directory:
                candidate = Path(directory) / relative
                if candidate.is_file():
                    return str(candidate)
    return None


def open_cot_window(url):
    """Open a new fullscreen game window on macOS Safari or Windows Chromium."""
    try:
        if sys.platform == "darwin":
            script = '''on run argv
tell application "Safari"
    set targetURL to item 1 of argv
    make new document with properties {URL:targetURL}
    set battleWindowID to id of front window
    activate
end tell
try
    tell application "System Events"
        tell process "Safari"
            repeat 40 times
                if exists front window then exit repeat
                delay 0.05
            end repeat
            tell application "Safari"
                if id of front window is not battleWindowID then error "Battle window is no longer frontmost"
            end tell
            if value of attribute "AXFullScreen" of front window is false then
                set value of attribute "AXFullScreen" of front window to true
            end if
        end tell
    end tell
on error errorMessage number errorNumber
    return (battleWindowID as text) & tab & (errorNumber as text)
end try
return battleWindowID
end run'''
            result = subprocess.run(["osascript", "-e", script, url], check=True, timeout=20,
                                    capture_output=True, text=True)
            identifier, _, fullscreen_error = result.stdout.strip().partition("\t")
            window = SafariWindow(int(identifier))
            if fullscreen_error:
                print("Safari 窗口已打开，macOS 未允许自动全屏。可为启动程序开启“辅助功能”与“自动化（System Events）”权限，"
                      f"或点击战斗画面进入全屏（系统代码 {fullscreen_error}）。", flush=True)
            return window
        elif sys.platform == "win32":
            browser = _windows_browser()
            if browser:
                profile = tempfile.TemporaryDirectory(prefix="jackal-browser-", ignore_cleanup_errors=True)
                try:
                    process = subprocess.Popen(
                        [browser, "--new-window", "--start-fullscreen", "--no-first-run",
                         "--no-default-browser-check", "--disable-background-mode",
                         f"--user-data-dir={profile.name}", url],
                        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                except OSError:
                    profile.cleanup()
                    raise
                return BrowserProcessWindow(process, profile)
            print("未找到 Edge 或 Chrome，将使用默认浏览器；可点击战斗画面进入全屏。", flush=True)
            webbrowser.open(url, new=1, autoraise=True)
        else:
            webbrowser.open(url, new=1, autoraise=True)
    except (subprocess.SubprocessError, OSError, ValueError) as error:
        # A locked/unresponsive desktop must not close the simulation.
        print(f"Browser launch unavailable ({type(error).__name__}). Open {url}", flush=True)
