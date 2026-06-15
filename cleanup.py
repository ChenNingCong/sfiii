"""Engine cleanup tool — no more manual `apptainer instance stop`.

  python cleanup.py            # reap orphans: stop engines whose launching PID is dead
  python cleanup.py --all      # stop ALL diambra engine instances on this node (nuke)

Normal runs auto-clean (atexit + SIGTERM/SIGINT + reap-on-start + idempotent
pre-clean by prefix), so this is only for after a kill -9 / catastrophic exit, or
to wipe everything.
"""
import argparse
from engine_manager import reap_orphans, _resolve_apptainer, _stop_instances, _list_instances_global


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="stop ALL engine instances (not just orphans)")
    opt = ap.parse_args()
    if opt.all:
        names = _list_instances_global()
        print(f"stopping ALL {len(names)} instances: {names}")
        _stop_instances(names)
    else:
        n = reap_orphans(verbose=True)
        print(f"reaped {n} orphaned engine group(s)")
