"""Fault-tolerant lifecycle manager for DIAMBRA engine apptainer instances.

Owns the container/server layer *independently* of the trainer so that a single
corrupted or stuck engine can be killed and respawned without touching the others
or the training process.

Design:
  - Fixed mapping  id i  ->  port = base_port + i  and instance name `<prefix>_<i>`.
    A restarted engine reuses the SAME port, so the Python client only has to
    reconnect to a stable address.
  - Parallel spawn (instances first, then servers) — the no-auth patched image
    starts in ~2s and parallelizes cleanly (no lock needed at this layer).
  - Health = apptainer instance alive AND TCP port listening.
  - `restart(i)` rebuilds just engine i.

Uses the no-auth patched image by default so spawn never blocks on login.
Deliberately does NOT import diambra — this module is usable and testable without
the Python client.

CLI (drop-in replacement for startup.py):
    python engine_manager.py --num-envs 16 --base-port 51000          # spawn + supervise
    python engine_manager.py --num-envs 16 --base-port 51000 --kill   # tear down only
"""
import atexit
import glob
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import time
from typing import Optional

PATCHED_IMAGE = "docker://chenningcong227/diambra-engine:patched"
DEFAULT_SIF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "engine_patched.sif")
# registry of running engine groups: one JSON per launching PID. A reaper
# (cleanup.py / reap_orphans) stops instances whose owner PID is dead -> no leaks
# even on kill -9 (which bypasses atexit/signal handlers).
REGISTRY_DIR = "/tmp/diambra_engines/registry"


def _resolve_apptainer() -> str:
    p = shutil.which("apptainer") or shutil.which("singularity")
    if p:
        return p
    raise RuntimeError(
        "apptainer not on PATH. Run `module load apptainer` before launching, "
        "or in an sbatch script load it in the same step.")


def port_listening(port: int, host: str = "localhost") -> bool:
    """True if something accepts a TCP connection on host:port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def _pid_alive(pid: int) -> bool:
    return pid > 0 and os.path.exists(f"/proc/{pid}")


def _stop_instances(names, apptainer=None):
    ap = apptainer or _resolve_apptainer()
    for n in names:
        subprocess.run([ap, "instance", "stop", n],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _list_instances_global():
    ap = _resolve_apptainer()
    out = subprocess.run([ap, "instance", "list"], capture_output=True, text=True).stdout
    return [l.split()[0] for l in out.splitlines()[1:] if l.split()]


def reap_orphans(verbose: bool = True) -> int:
    """Stop every engine group whose launching PID is dead (registry-based), so
    instances never leak even on kill -9. Returns # of groups reaped. Safe to call
    anytime (at startup, from cleanup.py, or a cron)."""
    reaped = 0
    for f in glob.glob(os.path.join(REGISTRY_DIR, "*.json")):
        try:
            reg = json.load(open(f))
        except Exception:
            os.remove(f); continue
        if _pid_alive(int(reg.get("pid", -1))):
            continue
        if verbose:
            print(f"[reap] owner pid {reg.get('pid')} dead -> stopping "
                  f"{len(reg.get('instances', []))} engines ({reg.get('prefix')})")
        _stop_instances(reg.get("instances", []))
        os.remove(f)
        reaped += 1
    return reaped


class EngineManager:
    def __init__(self, num_envs: int, base_port: int = 51000,
                 sif: str = DEFAULT_SIF, prefix: str = "engine",
                 roms_dir: Optional[str] = None, creds: Optional[str] = None,
                 scratch_base: str = "/tmp/diambra_scratch", cpu_pin=None):
        self.num_envs = num_envs
        self.base_port = base_port
        self.sif = sif
        self.prefix = prefix
        # cpu_pin: optional callable(engine_idx)->core-string (e.g. "0-7") that pins
        # the engine SERVER (the MAME process) to those cores via taskset, so a
        # worker<->engine gRPC round-trip stays on one NUMA node / CCD (EPYC).
        self.cpu_pin = cpu_pin
        self.roms_dir = roms_dir or os.path.join(os.getcwd(), "roms")
        self.creds = creds or os.path.expanduser("~/.diambra/credentials")
        # per-engine private $HOME + /tmp scratch. CRITICAL: apptainer auto-mounts
        # the host $HOME and /tmp into every container; the engine writes scratch
        # there during EnvInit, so sharing them makes concurrent make() race and
        # fail with "UNAVAILABLE: Socket closed". --contain + a private workdir/home
        # per engine removes the race entirely (no lock / stagger needed).
        self.scratch_base = scratch_base
        self.apptainer = _resolve_apptainer()
        self._servers: dict[int, subprocess.Popen] = {}

    def _workdir(self, i: int) -> str:
        return os.path.join(self.scratch_base, f"{self.prefix}_{i}")

    # ── naming ────────────────────────────────────────────────────────────────
    def name(self, i: int) -> str:
        return f"{self.prefix}_{i}"

    def port(self, i: int) -> int:
        return self.base_port + i

    def addresses(self) -> str:
        """Value for DIAMBRA_ENVS."""
        return " ".join(f"localhost:{self.port(i)}" for i in range(self.num_envs))

    # ── apptainer commands ──────────────────────────────────────────────────
    def _prep_workdir(self, i: int) -> None:
        """Fresh private scratch (home + tmp) for engine i — cleared on each start."""
        import shutil
        w = self._workdir(i)
        shutil.rmtree(w, ignore_errors=True)
        os.makedirs(os.path.join(w, "home"), exist_ok=True)

    def _start_instance_cmd(self, i: int) -> list[str]:
        # private home + tmp so concurrent engines don't collide on host $HOME//tmp
        w = self._workdir(i)
        home = os.path.join(w, "home")
        # roms mounted read-only (engine never writes there); no creds bind needed
        # for the no-auth patched image.
        return shlex.split(
            f"{self.apptainer} instance start --userns --contain "
            f"--workdir {w} --home {home} "
            f"-B /usr/share/alsa -B /usr/bin/getopt -B /usr/bin/cut "
            f"--bind {self.roms_dir}:/opt/diambraArena/roms:ro {self.sif} {self.name(i)}")

    def _server_cmd(self, i: int) -> list[str]:
        pin = f"taskset -c {self.cpu_pin(i)} " if self.cpu_pin else ""
        return shlex.split(
            f"{pin}{self.apptainer} exec --userns instance://{self.name(i)} "
            f"/bin/diambraEngineServer --envAddress 0.0.0.0:{self.port(i)}")

    def _stop_instance(self, i: int) -> None:
        srv = self._servers.pop(i, None)
        if srv and srv.poll() is None:
            srv.terminate()
            try:
                srv.wait(timeout=5)
            except subprocess.TimeoutExpired:
                srv.kill()
        subprocess.run([self.apptainer, "instance", "stop", self.name(i)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def _start_one(self, i: int, logdir: str = "/tmp/diambra_engines") -> None:
        os.makedirs(logdir, exist_ok=True)
        self._prep_workdir(i)
        # 1) instance (cheap, ~0.3s)
        subprocess.run(self._start_instance_cmd(i), check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # 2) server (background, owns the port)
        log = open(os.path.join(logdir, f"engine_{i}.log"), "w")
        self._servers[i] = subprocess.Popen(self._server_cmd(i), stdout=log, stderr=log)

    def start(self, ids: Optional[list[int]] = None, ready_timeout: float = 90.0,
              max_attempts: int = 3) -> None:
        """Spawn engines (parallel) and block until all requested ports listen.
        FAULT-TOLERANT: under heavy concurrent spawn an engine can lag; instead of
        hard-failing, restart the stragglers and retry up to max_attempts."""
        ids = list(range(self.num_envs)) if ids is None else ids
        reap_orphans(verbose=False)     # stop any dead-owner leaks before we add more
        self._preclean()                # idempotent: stop our own prefix's stale instances
        for i in ids:
            self._prep_workdir(i)
        procs = [subprocess.Popen(self._start_instance_cmd(i),
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                 for i in ids]
        for p in procs:
            p.wait()
        os.makedirs("/tmp/diambra_engines", exist_ok=True)
        for i in ids:
            log = open(f"/tmp/diambra_engines/engine_{i}.log", "w")
            self._servers[i] = subprocess.Popen(self._server_cmd(i), stdout=log, stderr=log)
        self._register(ids)             # track owner PID + auto-stop on exit/SIGTERM/SIGINT

        for attempt in range(max_attempts):
            pending = self.wait_ready(ids, timeout=ready_timeout, raise_on_timeout=False)
            if not pending:
                return
            print(f"[engine_manager] {len(pending)} engines not ready "
                  f"(ports {[self.port(i) for i in pending]}); restarting (attempt {attempt+1}/{max_attempts})")
            for i in pending:
                self._stop_instance(i)
                time.sleep(0.3)
                self._start_one(i)
        pending = self.wait_ready(ids, timeout=ready_timeout, raise_on_timeout=False)
        if pending:
            raise TimeoutError(f"engines not ready after {max_attempts} restarts: "
                               f"ports {[self.port(i) for i in pending]}")

    def wait_ready(self, ids: Optional[list[int]] = None, timeout: float = 60.0,
                   raise_on_timeout: bool = True):
        ids = list(range(self.num_envs)) if ids is None else ids
        deadline = time.time() + timeout
        pending = set(ids)
        while pending and time.time() < deadline:
            for i in list(pending):
                if port_listening(self.port(i)):
                    pending.discard(i)
            if pending:
                time.sleep(0.3)
        if pending and raise_on_timeout:
            raise TimeoutError(f"engines not ready after {timeout}s: ports "
                               f"{[self.port(i) for i in pending]}")
        return list(pending)

    def is_healthy(self, i: int) -> bool:
        # If we own the server Popen, a dead process is definitively unhealthy.
        # If we don't (e.g. a worker process that only manages restarts for its
        # own rank), fall back to the port check alone.
        srv = self._servers.get(i)
        if srv is not None and srv.poll() is not None:
            return False
        return port_listening(self.port(i))

    def restart(self, i: int, ready_timeout: float = 60.0) -> None:
        """Kill + respawn engine i on its fixed port."""
        self._stop_instance(i)
        time.sleep(0.5)  # let the port free
        self._start_one(i)
        self.wait_ready([i], timeout=ready_timeout)

    def health_report(self) -> dict[int, bool]:
        return {i: self.is_healthy(i) for i in range(self.num_envs)}

    # ── auto-cleanup (no manual stopping) ──────────────────────────────────────
    def _list_instances(self):
        out = subprocess.run([self.apptainer, "instance", "list"],
                             capture_output=True, text=True).stdout
        return [l.split()[0] for l in out.splitlines()[1:] if l.split()]

    def _preclean(self) -> None:
        """Stop any leftover instances with our prefix (idempotent relaunch)."""
        try:
            stale = [n for n in self._list_instances() if n.startswith(self.prefix + "_")]
            if stale:
                print(f"[engine_manager] pre-clean: stopping {len(stale)} stale '{self.prefix}_*'")
                _stop_instances(stale, self.apptainer)
        except Exception:
            pass

    def _register(self, ids) -> None:
        """Record owner PID + instance names so a reaper can stop them if we die
        (kill -9), and install atexit/SIGTERM/SIGINT cleanup for graceful exits."""
        os.makedirs(REGISTRY_DIR, exist_ok=True)
        self._owner_file = os.path.join(REGISTRY_DIR, f"{os.getpid()}_{self.prefix}.json")
        try:
            json.dump({"pid": os.getpid(), "prefix": self.prefix,
                       "instances": [self.name(i) for i in ids]}, open(self._owner_file, "w"))
        except Exception:
            pass
        if not getattr(self, "_cleanup_hooked", False):
            atexit.register(self.stop_all)
            for sig in (signal.SIGTERM, signal.SIGINT):
                try:
                    prev = signal.getsignal(sig)
                    def _h(s, f, _prev=prev):
                        self.stop_all()
                        raise SystemExit(0)
                    signal.signal(sig, _h)
                except (ValueError, OSError):
                    pass            # not in main thread (e.g. worker) -> skip
            self._cleanup_hooked = True

    def stop_all(self) -> None:
        for i in range(self.num_envs):
            self._stop_instance(i)
        of = getattr(self, "_owner_file", None)
        if of and os.path.exists(of):
            try: os.remove(of)
            except Exception: pass


# ── CLI: replaces startup.py ─────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=16)
    ap.add_argument("--base-port", type=int, default=51000)
    ap.add_argument("--sif", type=str, default=DEFAULT_SIF)
    ap.add_argument("--prefix", type=str, default="engine")
    ap.add_argument("--kill", action="store_true", help="tear down this prefix's engines and exit")
    ap.add_argument("--reap", action="store_true", help="stop engines whose owner PID is dead, and exit")
    ap.add_argument("--supervise", action="store_true",
                    help="after start, auto-restart any engine that goes unhealthy")
    opt = ap.parse_args()

    if opt.reap:
        print(f"reaped {reap_orphans(verbose=True)} orphan group(s)"); raise SystemExit(0)

    mgr = EngineManager(num_envs=opt.num_envs, base_port=opt.base_port, sif=opt.sif, prefix=opt.prefix)
    if opt.kill:
        mgr.stop_all()
        print("stopped all engines")
        raise SystemExit(0)

    t0 = time.time()
    mgr.start()
    print(f"started {opt.num_envs} engines in {time.time()-t0:.1f}s")
    print("DIAMBRA_ENVS=" + mgr.addresses())
    try:
        while True:
            time.sleep(5)
            if opt.supervise:
                for i, ok in mgr.health_report().items():
                    if not ok:
                        print(f"engine {i} unhealthy -> restarting")
                        mgr.restart(i)
    except KeyboardInterrupt:
        print("shutting down engines...")
    finally:
        mgr.stop_all()
