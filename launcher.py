"""
launcher.py — start Baseline, Approach 1 and Approach 2 on different ports,
              open them in browser tabs, and shut down all at once when you
              press Enter.

Usage:
    python launcher.py

Each app has its own file uploader — upload a different CSV to each tab to
compare approaches side by side.
"""

from __future__ import annotations
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent

JOBS = [
    ('baseline.py',   8501, 'Baseline'),
    ('approach_1.py', 8502, 'Approach 1'),
    ('approach_2.py', 8503, 'Approach 2'),
]

# TIP: to compare TWO datasets at once you do NOT need extra ports. Streamlit
# gives every browser tab its own independent session (separate upload + state),
# so just open the same URL twice — e.g. open http://localhost:8501 in two tabs,
# load AI-MIND in one and HCP in the other.

OPEN_BROWSER      = True
STARTUP_WAIT_SECS = 5


def _port_in_use(port: int) -> bool:
    """Return True if something is already listening on this port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(('127.0.0.1', port)) == 0


def _kill_tree(p: subprocess.Popen) -> None:
    """Kill a process and all its children (works reliably on Windows and POSIX)."""
    if sys.platform == 'win32':
        subprocess.call(
            ['taskkill', '/F', '/T', '/PID', str(p.pid)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    else:
        try:
            import os, signal
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except Exception:
            p.terminate()
    try:
        p.wait(timeout=5)
    except subprocess.TimeoutExpired:
        p.kill()


def main() -> int:
    # Validate scripts
    missing = [s for s, _, _ in JOBS if not (HERE / s).is_file()]
    if missing:
        print(f'ERROR: missing files: {missing}')
        return 1

    # Abort if any port is already occupied — prevents the duplicate-tab problem
    busy = [(label, port) for _, port, label in JOBS if _port_in_use(port)]
    if busy:
        for label, port in busy:
            print(f'ERROR: port {port} ({label}) is already in use.')
        print('\nKill the existing servers first (Task Manager → python.exe → End Task),')
        print('then run launcher.py again.')
        return 1

    procs: list[subprocess.Popen] = []
    print(f'Working directory: {HERE}')
    print(f'Launching {len(JOBS)} Streamlit instance(s)…\n')

    for script, port, label in JOBS:
        cmd = [
            sys.executable, '-m', 'streamlit', 'run', str(HERE / script),
            '--server.port', str(port),
            '--server.headless', 'true',       # suppress Streamlit's own browser open
            '--browser.gatherUsageStats', 'false',
        ]
        try:
            # Do NOT use CREATE_NEW_PROCESS_GROUP — it breaks taskkill /T
            p = subprocess.Popen(cmd)
            procs.append(p)
            print(f'  ✓ {label:<12} pid={p.pid:<6} → http://localhost:{port}')
        except Exception as e:
            print(f'  ✗ FAILED {label}: {e}')

    if not procs:
        print('Nothing started.')
        return 1

    # Wait for each server to actually be reachable before opening the browser
    print(f'\nWaiting for servers to come up (max {STARTUP_WAIT_SECS}s each)…')
    for _, port, label in JOBS:
        for _ in range(STARTUP_WAIT_SECS * 2):
            if _port_in_use(port):
                print(f'  ✓ {label} ready')
                break
            time.sleep(0.5)
        else:
            print(f'  ⚠ {label} did not respond in time — opening anyway')

    if OPEN_BROWSER:
        print('\nOpening browser tabs…')
        for _, port, label in JOBS:
            url = f'http://localhost:{port}'
            webbrowser.open_new_tab(url)
            print(f'  • {label}  →  {url}')
            time.sleep(0.3)   # small gap so tabs open in order

    print('\nAll servers running.')
    print('Press Enter (in THIS terminal) to stop all servers and exit.\n')
    try:
        input()
    except KeyboardInterrupt:
        pass

    print('\nStopping servers…')
    for p in procs:
        _kill_tree(p)
    print('Done.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
