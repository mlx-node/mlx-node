"""Run one model smoke with RSS, physical-footprint, pressure and time guards.

Usage: python3 scripts/guard-model-memory.py LOG COMMAND [ARGS ...]
Stops only the process group it started; it never changes macOS memory settings.
"""
import ctypes
import errno
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

if len(sys.argv) < 3:
    raise SystemExit(__doc__)
log = Path(sys.argv[1])
log.parent.mkdir(parents=True, exist_ok=True)
peak = 0
peak_footprint = 0
start = time.monotonic()
reason = None
rss_limit_gib = int(os.environ.get('MODEL_GUARD_RSS_GIB', '16'))
if not 8 <= rss_limit_gib <= 104:
    raise SystemExit('MODEL_GUARD_RSS_GIB must be between 8 and 104')
rss_limit = rss_limit_gib * 1024**3
footprint_limit_gib = int(os.environ.get('MODEL_GUARD_FOOTPRINT_GIB', str(rss_limit_gib)))
if not 8 <= footprint_limit_gib <= 104:
    raise SystemExit('MODEL_GUARD_FOOTPRINT_GIB must be between 8 and 104')
footprint_limit = footprint_limit_gib * 1024**3
other_limit_gib = os.environ.get('MODEL_GUARD_OTHER_FOOTPRINT_GIB')
other_limit = int(other_limit_gib) * 1024**3 if other_limit_gib else None
if other_limit is not None and not 2 <= int(other_limit_gib) <= 104:
    raise SystemExit('MODEL_GUARD_OTHER_FOOTPRINT_GIB must be between 2 and 104')
competing_processes = []

# RSS can undercount Metal allocations by tens of GiB. This stable Darwin
# RUSAGE_INFO_V0 layout comes from sys/resource.h; physical footprint includes
# memory charged to the task even when it is absent from ps's RSS column.
class RusageInfoV0(ctypes.Structure):
    _fields_ = [('uuid', ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64) for name in (
            'user_time', 'system_time', 'pkg_idle_wkups', 'interrupt_wkups',
            'pageins', 'wired_size', 'resident_size', 'phys_footprint',
            'proc_start_abstime', 'proc_exit_abstime')]


libproc = ctypes.CDLL('/usr/lib/libproc.dylib', use_errno=True) if sys.platform == 'darwin' else None
if libproc is not None:
    libproc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    libproc.proc_pid_rusage.restype = ctypes.c_int


def physical_footprint(pid):
    if libproc is None:
        return 0
    info = RusageInfoV0()
    if libproc.proc_pid_rusage(pid, 0, ctypes.byref(info)) != 0:
        code = ctypes.get_errno()
        if code == errno.ESRCH:  # Process exited after the group snapshot.
            return 0
        raise OSError(code, f'Cannot read physical footprint for owned PID {pid}')
    return info.phys_footprint


def group_members(group_id):
    """Only live members of the process group started by this guard."""
    snapshot = subprocess.run(['ps', '-axo', 'pid=,pgid=,stat=,rss='],
                              capture_output=True, text=True, timeout=5, check=True)
    return [(int(pid), int(rss) * 1024) for pid, group, state, rss in
            (line.split() for line in snapshot.stdout.splitlines())
            if int(group) == group_id and not state.startswith('Z')]


def competing_footprints(group_id):
    """Read other processes; only the owned group may be stopped by this guard."""
    if other_limit is None or libproc is None:
        return []
    snapshot = subprocess.run(['ps', '-axo', 'pid=,pgid=,stat=,comm='],
                              capture_output=True, text=True, timeout=5, check=True)
    found = []
    for line in snapshot.stdout.splitlines():
        pid, group, state, command = line.strip().split(None, 3)
        if int(group) == group_id or state.startswith('Z'):
            continue
        try:
            footprint = physical_footprint(int(pid))
        except OSError:
            # Some OS-owned processes do not expose rusage to this user.
            continue
        if footprint >= other_limit:
            found.append({'pid': int(pid), 'name': Path(command).name,
                          'physical_footprint_bytes': footprint})
    return found


def stop_group(group_id):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        members = group_members(group_id)
        if not members:
            return
        try:
            os.killpg(group_id, sig)
        except ProcessLookupError:
            return
        except PermissionError:
            # macOS may reject the group signal during launcher teardown.
            # Signal only live members that still belong to our own group.
            for pid, _ in members:
                try:
                    if os.getpgid(pid) == group_id:
                        os.kill(pid, sig)
                except ProcessLookupError:
                    pass
        if sig == signal.SIGTERM:
            deadline = time.monotonic() + 5
            while group_members(group_id) and time.monotonic() < deadline:
                time.sleep(0.1)


with log.open('w') as output:
    child = subprocess.Popen(sys.argv[2:], stdout=output, stderr=subprocess.STDOUT,
                             env={**os.environ, 'MLX_AGENT_METRICS': '0'},
                             start_new_session=True)
    try:
        while child.poll() is None or group_members(child.pid):
            members = group_members(child.pid)
            value = sum(rss for _, rss in members)
            footprint = sum(physical_footprint(pid) for pid, _ in members)
            peak = max(peak, value)
            peak_footprint = max(peak_footprint, footprint)
            competing_processes = competing_footprints(child.pid)
            pressure = subprocess.run(['sysctl', '-n', 'kern.memorystatus_vm_pressure_level'],
                                      capture_output=True, text=True, timeout=5)
            if competing_processes:
                reason = f'Another process exceeded the {other_limit_gib} GiB comparison threshold'
            elif value > rss_limit:
                reason = f'RSS exceeded {rss_limit_gib} GiB'
            elif footprint > footprint_limit:
                reason = f'Physical footprint exceeded {footprint_limit_gib} GiB'
            elif pressure.returncode == 0 and int(pressure.stdout.strip()) >= 2:
                reason = 'macOS reported elevated memory pressure'
            elif time.monotonic() - start > 1800:
                reason = '30 minute smoke-test timeout'
            if reason:
                stop_group(child.pid)
                break
            time.sleep(0.5)
    except Exception as exc:
        reason = reason or f'guard monitoring failed: {exc}'
    finally:
        try:
            stop_group(child.pid)
        except OSError as exc:
            reason = f"{reason or 'guard cleanup failed'}; cannot stop child group: {exc}"
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
# Some launchers return zero after their native worker aborts. Require the
# caller's completion event when supplied, and never accept a fatal-runtime log.
log_text = log.read_text(errors='replace')
required_event = os.environ.get('MODEL_GUARD_REQUIRE_EVENT')
completion_seen = None
if 'fatal runtime error:' in log_text or 'QWEN4_NATIVE_THROW' in log_text and 'aborting' in log_text:
    reason = reason or 'native runtime aborted despite launcher exit status'
if required_event:
    completion_seen = False
    for line in log_text.splitlines():
        try:
            event = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(event, dict) and event.get('event') == required_event:
            completion_seen = True
    if not completion_seen:
        reason = reason or f'missing completion event: {required_event}'
report = {'exit_code': child.returncode, 'completion_event_seen': completion_seen, 'peak_rss_bytes': peak, 'rss_limit_bytes': rss_limit,
          'peak_physical_footprint_bytes': peak_footprint if libproc is not None else None,
          'physical_footprint_limit_bytes': footprint_limit if libproc is not None else None,
          'seconds': time.monotonic() - start, 'guard_stop_reason': reason}
if other_limit is not None:
    report['other_process_threshold_bytes'] = other_limit
    report['competing_processes'] = competing_processes
log.with_suffix('.memory.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report), flush=True)
raise SystemExit(1 if reason or child.returncode else 0)
