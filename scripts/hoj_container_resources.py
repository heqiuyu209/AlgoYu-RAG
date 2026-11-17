"""Read-only Docker resource samples with replayable cgroup v1/v2 evidence.

No model imports, service mutations, environment reads or process command lines.
Private cgroup namespaces require a full-ID Docker exec and matching inspections
on both sides of the read; a namespace root alone is never container identity.
"""
from datetime import datetime, timezone
from pathlib import PurePosixPath
import json
import math
import re
import subprocess
import time


SERVICES = tuple('hoj-master-' + name + '-1' for name in (
    'hoj-rag', 'hoj-backend', 'hoj-judgeserver', 'hoj-mysql',
    'hoj-frontend', 'hoj-nacos', 'hoj-redis'))
IDENTITY_FORMAT = (
    '{"id":{{json .Id}},"name":{{json .Name}},"image":{{json .Image}},'
    '"running":{{json .State.Running}},"startedAt":{{json .State.StartedAt}},'
    '"pid":{{json .State.Pid}},"oomKilled":{{json .State.OOMKilled}},'
    '"cgroupns":{{json .HostConfig.CgroupnsMode}}}')

# Only controller mounts at Docker's standard locations are supported. Reject
# other topologies instead of guessing or accidentally observing the host root.
SAMPLE_BODY = r'''
emit /proc/1/cgroup
printf '@@FILE cgroup-mountinfo\n'
while IFS= read -r line; do
  case "$line" in *' - cgroup '*|*' - cgroup2 '*) printf '%s\n' "$line";; esac
done < /proc/1/mountinfo
printf '\n'
if namespace=$(readlink /proc/1/ns/cgroup 2>/dev/null); then
  printf '@@FILE /proc/1/ns/cgroup\n%s\n\n' "$namespace"
fi
while IFS=: read -r hierarchy controllers member; do
  case ":$controllers:" in
    ::) kind=cgroup2; mount=/sys/fs/cgroup;;
    *:memory:*) kind=cgroup; mount=/sys/fs/cgroup/memory;;
    *) continue;;
  esac
  while IFS= read -r line; do
    case "$line" in *" - $kind "*) ;; *) continue;; esac
    set -f; set -- $line
    root=$4; point=$5
    set +f
    [ "$point" = "$mount" ] || continue
    case "$member" in
      "$root") relative=;;
      "$root"/*) relative=${member#"$root"};;
      /) relative=;;
      *) if [ "$root" = / ]; then relative=$member; else continue; fi;;
    esac
    folder=$mount$relative
    if [ "$kind" = cgroup2 ]; then
      for leaf in memory.stat memory.current memory.peak memory.max memory.swap.current memory.swap.max memory.events; do emit "$folder/$leaf"; done
    else
      for leaf in memory.stat memory.usage_in_bytes memory.max_usage_in_bytes memory.limit_in_bytes memory.memsw.usage_in_bytes memory.memsw.limit_in_bytes memory.failcnt memory.oom_control; do emit "$folder/$leaf"; done
    fi
  done < /proc/1/mountinfo
done < /proc/1/cgroup
emit /proc/sys/kernel/random/boot_id
emit /proc/uptime
emit /proc/meminfo
emit /proc/vmstat
emit /proc/pressure/memory
for folder in /proc/[0-9]*; do
  emit_stat "$folder/stat"
  emit_status "$folder/status"
done
'''
SHELL_FUNCTIONS = r'''
emit() {
  if [ -r "$1" ]; then
    printf '@@FILE %s\n' "$1"
    cat "$1"
    printf '\n'
  fi
}
emit_stat() {
  if IFS= read -r stat_line 2>/dev/null < "$1"; then
    printf '@@FILE %s\n%s\n\n' "$1" "$stat_line"
  fi
}
emit_status() {
  if [ -r "$1" ]; then
    printf '@@FILE %s\n' "$1"
    awk '/^(Name|Pid|Tgid|VmRSS|RssAnon|RssFile|RssShmem|VmSwap|Threads):/ {print}' "$1"
    printf '\n'
  fi
}
'''
SCRIPT = SHELL_FUNCTIONS + SAMPLE_BODY


def command(args, input=None, timeout=8):
    payload = input.encode('utf-8') if input is not None else None
    return subprocess.run(args, input=payload, capture_output=True,
                          timeout=timeout, check=True).stdout.decode('utf-8')


def inspect_identity(container_id, timeout=8):
    return json.loads(command(['docker', 'inspect', '--format', IDENTITY_FORMAT,
                               container_id], timeout=timeout))


def identities():
    rows = command(['docker', 'inspect', '--format', IDENTITY_FORMAT, *SERVICES]).splitlines()
    if len(rows) != len(SERVICES):
        raise ValueError('missing original service')
    result = dict(zip(SERVICES, map(json.loads, rows)))
    for name, row in result.items():
        if row['name'] != '/' + name or not row['running'] or row['oomKilled']:
            raise ValueError('original service identity or running state changed')
    return result


def binding(container_id, before, after):
    if re.fullmatch('[0-9a-f]{64}', container_id) is None:
        raise ValueError('full Docker container ID required')
    keys = ('id', 'name', 'image', 'running', 'startedAt', 'pid', 'oomKilled', 'cgroupns')
    if (any(key not in before or key not in after for key in keys)
            or any(before[key] != after[key] for key in keys)
            or before['id'] != container_id or before['running'] is not True
            or before['oomKilled'] is not False or not isinstance(before['pid'], int)
            or before['pid'] <= 0 or not before['startedAt'] or not before['image']):
        raise ValueError('Docker identity changed across resource observation')
    return dict(dockerExecTarget=container_id, before=before, after=after)


def sections(raw):
    result, current = {}, None
    for line in raw.splitlines():
        if line.startswith('@@FILE '):
            current = line[7:]
            if not current or current in result:
                raise ValueError('duplicate or empty file section')
            result[current] = []
        elif current is not None:
            result[current].append(line)
        elif line.strip():
            raise ValueError('unframed observation')
    return {name: '\n'.join(lines).strip() for name, lines in result.items()}


def table(raw, source='resource table'):
    result = {}
    for line in raw.splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 2:
            raise ValueError(source + ': incomplete counter line: ' + repr(line[:96]))
        key, value, *units = parts
        key = key.rstrip(':')
        if key in result or units not in ([], ['kB']):
            raise ValueError(source + ': duplicate counter or unsupported units')
        try:
            result[key] = int(value) * (1024 if units else 1)
        except ValueError as error:
            raise ValueError(source + ': invalid counter value for ' + key) from error
        if result[key] < 0:
            raise ValueError(source + ': negative resource counter')
    return result


def _controller(files, container_id, evidence):
    candidates = []
    for member_line in files['/proc/1/cgroup'].splitlines():
        _, controllers, member = member_line.split(':', 2)
        if controllers and 'memory' not in controllers.split(','):
            continue
        version = 1 if controllers else 2
        for line in files['cgroup-mountinfo'].splitlines():
            left, right = line.split(' - ', 1)
            fields, tail = left.split(), right.split()
            if tail[0] != ('cgroup' if version == 1 else 'cgroup2'):
                continue
            if version == 1 and 'memory' not in tail[2].split(','):
                continue
            root, mount = fields[3:5]
            standard = '/sys/fs/cgroup/memory' if version == 1 else '/sys/fs/cgroup'
            if mount != standard or any('\\' in path or '..' in path.split('/')
                                         for path in (root, mount, member)):
                raise ValueError('unsupported controller path')
            if member == '/':
                if evidence is None or evidence['before']['cgroupns'] != 'private':
                    raise ValueError('namespace root requires a verified private Docker exec')
                root_parts = PurePosixPath(root).parts
                if root != '/' and container_id not in root_parts and 'docker-' + container_id + '.scope' not in root_parts:
                    raise ValueError('private controller mount names another container')
                folder = mount
                method = 'docker-exec-and-inspection'
            else:
                parts = PurePosixPath(member).parts
                if container_id not in parts and 'docker-' + container_id + '.scope' not in parts:
                    raise ValueError('memory membership does not name this container')
                try:
                    suffix = PurePosixPath(member).relative_to(root)
                except ValueError:
                    raise ValueError('controller membership outside mount root') from None
                folder = str(PurePosixPath(mount) / suffix)
                method = 'kernel-cgroup-path'
            leaf = '/memory.max' if version == 2 else '/memory.limit_in_bytes'
            if folder + leaf in files:
                candidates.append(dict(version=version, membership=member, mountRoot=root,
                                       directory=folder, identityMethod=method))
    if len(candidates) != 1:
        raise ValueError('cannot uniquely locate the container memory controller')
    return candidates[0]


def parse(raw, container_id, evidence=None):
    if re.fullmatch('[0-9a-f]{64}', container_id) is None:
        raise ValueError('full Docker container ID required')
    if evidence is not None:
        if evidence.get('dockerExecTarget') != container_id:
            raise ValueError('Docker exec target changed')
        binding(container_id, evidence['before'], evidence['after'])
    files = sections(raw)
    location = _controller(files, container_id, evidence)
    folder = location['directory'] + '/'
    def number(leaf, unlimited=False):
        text = files[folder + leaf]
        if unlimited and text == 'max':
            return None
        value = int(text)
        if value < 0:
            raise ValueError('negative memory value')
        return value
    stats = table(files[folder + 'memory.stat'], folder + 'memory.stat')
    v2 = location['version'] == 2
    if v2:
        events = table(files[folder + 'memory.events'], folder + 'memory.events')
        memory = dict(memoryCurrentBytes=number('memory.current'), memoryPeakBytes=number('memory.peak'),
            memoryLimitBytes=number('memory.max', True), memoryLimitUnlimited=files[folder + 'memory.max'] == 'max',
            swapCurrentBytes=number('memory.swap.current'), swapHardLimitBytes=number('memory.swap.max', True),
            swapLimitUnlimited=files[folder + 'memory.swap.max'] == 'max',
            memoryLimitHits=events.get('max'), oomEventCount=events.get('oom'), oomKillCount=events.get('oom_kill'),
            underOom=None, combinedMemorySwapLimitBytes=None, memoryEvents=events,
            inactiveFileBytes=stats.get('inactive_file'))
    else:
        oom = table(files[folder + 'memory.oom_control'], folder + 'memory.oom_control')
        memory = dict(memoryCurrentBytes=number('memory.usage_in_bytes'), memoryPeakBytes=number('memory.max_usage_in_bytes'),
            memoryLimitBytes=number('memory.limit_in_bytes'), memoryLimitUnlimited=False,
            swapCurrentBytes=stats.get('total_swap'), swapHardLimitBytes=None, swapLimitUnlimited=None,
            memoryLimitHits=number('memory.failcnt'), oomEventCount=None, oomKillCount=oom.get('oom_kill'),
            underOom=oom.get('under_oom'), combinedMemorySwapLimitBytes=number('memory.memsw.limit_in_bytes'),
            combinedMemorySwapCurrentBytes=number('memory.memsw.usage_in_bytes'),
            memoryEvents=None, inactiveFileBytes=stats.get('total_inactive_file'))
    normalized = dict(controller=location, memoryHierarchyVersion=location['version'], **memory,
        memoryStats=stats, vm=table(files['/proc/vmstat'], '/proc/vmstat'),
        meminfo=table(files['/proc/meminfo'], '/proc/meminfo'),
        bootId=files['/proc/sys/kernel/random/boot_id'],
        uptimeSeconds=float(files['/proc/uptime'].split()[0]),
        cgroupNamespace=files.get('/proc/1/ns/cgroup'),
        pressureMemory=files.get('/proc/pressure/memory'), processes={})
    if location['identityMethod'] == 'docker-exec-and-inspection' and re.fullmatch(
            r'cgroup:\[[0-9]+\]', normalized['cgroupNamespace'] or '') is None:
        raise ValueError('private cgroup namespace evidence unavailable')
    for path, value in files.items():
        if not re.fullmatch(r'/proc/[0-9]+/stat', path):
            continue
        pid = path.split('/')[2]
        fields = value.rsplit(') ', 1)[1].split()
        if len(fields) < 20 or value.split(' ', 1)[0] != pid:
            raise ValueError('incomplete or mismatched process stat')
        status = {}
        for line in files.get('/proc/' + pid + '/status', '').splitlines():
            name, item = line.split(':', 1)
            status.update({name: item.strip()} if name == 'Name' else table(line, '/proc/' + pid + '/status'))
        normalized['processes'][pid] = dict(startTicks=int(fields[19]),
            minorFaults=int(fields[7]), majorFaults=int(fields[9]), status=status)
    result = dict(raw=files, normalized=normalized)
    require_observation_fields(result)
    return result


def require_observation_fields(point):
    value = point['normalized']
    names = ('memoryCurrentBytes', 'memoryPeakBytes', 'swapCurrentBytes',
             'memoryLimitHits', 'oomKillCount', 'inactiveFileBytes')
    if any(not isinstance(value.get(key), int) or value[key] < 0 for key in names):
        raise ValueError('required resource counter unavailable')
    # Reads are non-atomic; current can grow between reading current and peak.
    for name in ('memory', 'swap'):
        key = 'memoryLimitBytes' if name == 'memory' else 'swapHardLimitBytes'
        if name == 'swap' and value['memoryHierarchyVersion'] == 1:
            continue
        unlimited = value[name + 'LimitUnlimited']
        if (unlimited is True and value[key] is not None) or (
                unlimited is False and (not isinstance(value[key], int) or value[key] < 0)):
            raise ValueError('incomplete memory limit observation')
    if value['memoryHierarchyVersion'] == 2:
        if not isinstance(value['oomEventCount'], int) or value['oomEventCount'] < 0:
            raise ValueError('required OOM event counter unavailable')
    elif value['underOom'] not in (0, 1) or value['combinedMemorySwapLimitBytes'] is None:
        raise ValueError('required v1 OOM or memory/swap limit unavailable')
    for field, keys in (('vm', ('pswpin', 'pswpout')), ('meminfo', ('MemTotal', 'MemAvailable'))):
        if any(not isinstance(value[field].get(key), int) or value[field][key] < 0 for key in keys):
            raise ValueError('required resource table incomplete: ' + field)
    if not value['bootId'] or not math.isfinite(value['uptimeSeconds']) or value['uptimeSeconds'] < 0:
        raise ValueError('invalid VM identity or uptime')
    if value['processes'].get('1', {}).get('startTicks', 0) <= 0:
        raise ValueError('container main-process identity unavailable')


def sample(name, identity, deadline=None):
    started = time.monotonic()
    deadline = min(started + 8, deadline) if deadline is not None else started + 8
    def timeout():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('observation deadline exhausted')
        return remaining
    target = identity['id']
    before = inspect_identity(target, timeout())
    raw = command(['docker', 'exec', '-i', target, 'timeout', '-s', 'TERM', '6', 'sh', '-s'], SCRIPT, timeout())
    after = inspect_identity(target, timeout())
    if before != identity or before['name'] != '/' + name:
        raise ValueError('requested original service identity changed')
    evidence = binding(target, before, after)
    if time.monotonic() >= deadline:
        raise TimeoutError('resource response completed after its request deadline')
    return dict(container=name, helperId=target, binding=evidence,
        observedAtUtc=datetime.now(timezone.utc).isoformat(), monotonicStarted=started,
        monotonicEnded=time.monotonic(), **parse(raw, target, evidence))


def delta(first, last):
    if (first['bootId'] != last['bootId'] or last['uptimeSeconds'] < first['uptimeSeconds']
            or first['controller'] != last['controller']
            or first['cgroupNamespace'] != last['cgroupNamespace']):
        raise ValueError('VM or cgroup identity changed')
    def counters(a, b, keys):
        result = {}
        for key in keys:
            if a.get(key) is None or b.get(key) is None:
                result[key] = None
            elif b[key] < a[key]:
                raise ValueError('cumulative resource counter reset: ' + key)
            else:
                result[key] = b[key] - a[key]
        return result
    return dict(vm=counters(first['vm'], last['vm'], ('pswpin', 'pswpout', 'pgfault', 'pgmajfault')),
        memoryEvents=counters(first, last, ('memoryLimitHits', 'oomEventCount', 'oomKillCount')),
        swapCurrentBytesDelta=last['swapCurrentBytes'] - first['swapCurrentBytes'],
        stableProcesses={pid: counters(row, last['processes'][pid], ('minorFaults', 'majorFaults'))
            for pid, row in first['processes'].items() if pid in last['processes']
            and row['startTicks'] == last['processes'][pid]['startTicks']})


def require_stable_window(anchor, point):
    """Use the first window sample, including gaps, never a reset per phase."""
    changes = delta(anchor, point)
    if (anchor['processes']['1']['startTicks'] != point['processes']['1']['startTicks']
            or any(changes['vm'][key] != 0 for key in ('pswpin', 'pswpout'))
            or any(value not in (None, 0) for value in changes['memoryEvents'].values())
            or point['underOom'] == 1):
        raise ValueError('service restarted, paging occurred, or memory/OOM events increased')
    return changes
