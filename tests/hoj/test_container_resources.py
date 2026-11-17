"""Kernel-shaped fixtures for resource and identity safety; no Docker required."""
import copy
import importlib.util
from pathlib import Path
import unittest


SPEC = importlib.util.spec_from_file_location('hoj_container_resources',
    Path(__file__).resolve().parents[2] / 'scripts/hoj_container_resources.py')
resources = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resources)
CID = 'a' * 64
OTHER = 'b' * 64


def identity():
    return dict(id=CID, name='/hoj-test', image='sha256:' + 'c' * 64,
        running=True, startedAt='2026-11-17T00:00:00Z', pid=1234,
        oomKilled=False, cgroupns='private')


def fixture(version=2, private=False, scope=False, rooted=False):
    member = '/' if private else ('/system.slice/docker-' + CID + '.scope' if scope else '/docker/' + CID)
    mount = '/sys/fs/cgroup' if version == 2 else '/sys/fs/cgroup/memory'
    root = member if rooted else '/'
    folder = mount if private or rooted else mount + member
    fields = ['S'] + ['0'] * 19
    fields[7], fields[9], fields[19] = '100', '2', '500'
    files = {'/proc/1/cgroup': ('0::' if version == 2 else '5:memory:') + member,
        'cgroup-mountinfo': '24 20 0:25 ' + root + ' ' + mount +
            ' ro - ' + ('cgroup2 cgroup rw,nsdelegate' if version == 2 else 'cgroup cgroup rw,memory'),
        '/proc/1/ns/cgroup': 'cgroup:[4026532714]',
        '/proc/sys/kernel/random/boot_id': 'test-boot-id', '/proc/uptime': '100.00 25.0',
        '/proc/meminfo': 'MemTotal: 10000000 kB\nMemAvailable: 8000000 kB',
        '/proc/vmstat': 'pswpin 10\npswpout 20\npgfault 50\npgmajfault 2',
        '/proc/1/stat': '1 (process (name)) ' + ' '.join(fields),
        '/proc/1/status': 'Name:\ttest\nPid:\t1\nVmRSS:\t4 kB\nThreads:\t3'}
    if version == 2:
        values = {'memory.stat': 'inactive_file 4096\npgfault 100\npgmajfault 2',
            'memory.current': '32768', 'memory.peak': '65536', 'memory.max': 'max',
            'memory.swap.current': '8192', 'memory.swap.max': 'max',
            'memory.events': 'low 0\nhigh 0\nmax 0\noom 0\noom_kill 0'}
    else:
        values = {'memory.stat': 'total_inactive_file 4096\ntotal_swap 8192\ntotal_pgfault 100',
            'memory.usage_in_bytes': '32768', 'memory.max_usage_in_bytes': '65536',
            'memory.limit_in_bytes': '1048576', 'memory.memsw.usage_in_bytes': '40960',
            'memory.memsw.limit_in_bytes': '1048576', 'memory.failcnt': '0',
            'memory.oom_control': 'oom_kill_disable 0\nunder_oom 0\noom_kill 0'}
    files.update({folder + '/' + leaf: value for leaf, value in values.items()})
    return files


def raw(files):
    return ''.join('@@FILE ' + name + '\n' + value + '\n\n' for name, value in files.items())


def parse(files, evidence=None):
    return resources.parse(raw(files), CID, evidence)


class ContainerResourcesTest(unittest.TestCase):
    def test_host_hierarchies_and_mount_roots(self):
        for version in (1, 2):
            for scope in (False, True):
                for rooted in (False, True):
                    with self.subTest(version=version, scope=scope, rooted=rooted):
                        result = parse(fixture(version, scope=scope, rooted=rooted))
                        self.assertEqual(result['normalized']['memoryHierarchyVersion'], version)
                        self.assertEqual(result['normalized']['swapCurrentBytes'], 8192)
                        self.assertEqual(result['normalized']['memoryCurrentBytes'], 32768)

    def test_private_namespace_requires_verified_exec(self):
        with self.assertRaises(ValueError):
            parse(fixture(private=True))
        info = identity()
        evidence = resources.binding(CID, info, dict(info))
        point = parse(fixture(private=True), evidence)
        self.assertEqual(point['normalized']['controller']['identityMethod'], 'docker-exec-and-inspection')
        self.assertEqual(resources.parse(raw(point['raw']), CID, evidence), point)

    def test_private_namespace_not_accepted_for_host_inspection(self):
        info = identity(); info['cgroupns'] = 'host'
        with self.assertRaises(ValueError):
            parse(fixture(private=True), resources.binding(CID, info, dict(info)))

    def test_namespace_link_unavailable_is_only_allowed_with_kernel_identity(self):
        files = fixture(); del files['/proc/1/ns/cgroup']
        self.assertIsNone(parse(files)['normalized']['cgroupNamespace'])
        files = fixture(private=True); del files['/proc/1/ns/cgroup']
        info = identity()
        with self.assertRaises(ValueError):
            parse(files, resources.binding(CID, info, dict(info)))

    def test_v1_private_mount_root_is_bound_to_this_container(self):
        info = identity(); evidence = resources.binding(CID, info, dict(info))
        files = fixture(1, private=True)
        self.assertEqual(parse(files, evidence)['normalized']['memoryHierarchyVersion'], 1)
        files['cgroup-mountinfo'] = files['cgroup-mountinfo'].replace('0:25 / ', '0:25 /docker/' + OTHER + ' ')
        with self.assertRaises(ValueError):
            parse(files, evidence)

    def test_restart_replacement_and_oom_are_rejected(self):
        for key, value in (('id', OTHER), ('image', 'different'), ('pid', 99),
                ('startedAt', 'later'), ('running', False), ('oomKilled', True),
                ('name', '/replacement'), ('cgroupns', 'host')):
            with self.subTest(key=key), self.assertRaises(ValueError):
                before = identity(); after = dict(before); after[key] = value
                resources.binding(CID, before, after)

    def test_wrong_exec_target_rejected_even_with_matching_inspections(self):
        info = identity(); evidence = resources.binding(CID, info, dict(info))
        evidence['dockerExecTarget'] = OTHER
        with self.assertRaises(ValueError):
            parse(fixture(private=True), evidence)

    def test_unlimited_limits_stay_unlimited(self):
        value = parse(fixture())['normalized']
        self.assertIsNone(value['memoryLimitBytes'])
        self.assertIsNone(value['swapHardLimitBytes'])
        self.assertTrue(value['memoryLimitUnlimited'])
        self.assertTrue(value['swapLimitUnlimited'])
        self.assertIsNone(value['underOom'])
        self.assertIsNone(value['combinedMemorySwapLimitBytes'])

    def test_finite_limits_and_zero_swap_cap_are_actual_values(self):
        files = fixture(); folder = '/sys/fs/cgroup/docker/' + CID + '/'
        files[folder + 'memory.max'] = '1048576'
        files[folder + 'memory.swap.max'] = '0'
        value = parse(files)['normalized']
        self.assertEqual((value['memoryLimitBytes'], value['swapHardLimitBytes']), (1048576, 0))
        self.assertFalse(value['memoryLimitUnlimited'])
        self.assertFalse(value['swapLimitUnlimited'])

    def test_missing_counters_are_not_zero_filled(self):
        for suffix, value in (('memory.events', 'max 0\noom_kill 0'),
                ('memory.stat', 'pgfault 100'), ('memory.swap.current', ''),
                ('memory.peak', '')):
            with self.subTest(suffix=suffix), self.assertRaises((ValueError, KeyError)):
                files = fixture(); files['/sys/fs/cgroup/docker/' + CID + '/' + suffix] = value
                parse(files)

    def test_missing_global_swap_counter_rejected(self):
        files = fixture(); files['/proc/vmstat'] = 'pswpin 0'
        with self.assertRaises(ValueError):
            parse(files)

    def test_wrong_cgroup_and_parent_escape_rejected(self):
        for member in ('/docker/' + OTHER, '/docker/' + CID + '/../' + OTHER):
            files = fixture(); files['/proc/1/cgroup'] = '0::' + member
            with self.assertRaises(ValueError):
                parse(files)

    def test_negative_event_rejected(self):
        files = fixture(); files['/sys/fs/cgroup/docker/' + CID + '/memory.events'] = 'max -1\noom 0\noom_kill 0'
        with self.assertRaises(ValueError):
            parse(files)

    def test_duplicate_raw_sections_and_counters_rejected(self):
        with self.assertRaises(ValueError):
            resources.sections('@@FILE a\n1\n@@FILE a\n2')
        with self.assertRaises(ValueError):
            resources.table('oom 0\noom 1')

    def test_process_names_with_parentheses_and_rss_units(self):
        process = parse(fixture())['normalized']['processes']['1']
        self.assertEqual((process['startTicks'], process['minorFaults'], process['majorFaults']), (500, 100, 2))
        self.assertEqual(process['status']['VmRSS'], 4096)

    def test_missing_main_process_and_invalid_uptime_rejected(self):
        for kind in ('process', 'uptime'):
            files = fixture()
            if kind == 'process':
                del files['/proc/1/stat']
            else:
                files['/proc/uptime'] = 'nan 0'
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                parse(files)

    def test_swap_gauge_can_decrease_but_io_is_counted(self):
        a = parse(fixture())['normalized']; b = copy.deepcopy(a)
        b['swapCurrentBytes'] = 0; b['vm']['pswpin'] += 2
        result = resources.delta(a, b)
        self.assertEqual(result['swapCurrentBytesDelta'], -8192)
        self.assertEqual(result['vm']['pswpin'], 2)

    def test_global_swap_and_event_reset_rejected(self):
        for field in ('pswpin', 'memoryLimitHits'):
            a = parse(fixture())['normalized']; b = copy.deepcopy(a)
            if field == 'pswpin':
                b['vm'][field] = 0
            else:
                a[field] = 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                resources.delta(a, b)

    def test_reboot_namespace_change_and_pid_reuse(self):
        a = parse(fixture())['normalized']
        for key, value in (('bootId', 'other'), ('cgroupNamespace', 'cgroup:[1]')):
            b = copy.deepcopy(a); b[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                resources.delta(a, b)
        b = copy.deepcopy(a); b['processes']['1']['startTicks'] += 1
        self.assertEqual(resources.delta(a, b)['stableProcesses'], {})

    def test_full_window_rejects_new_oom_limits_paging_and_service_restart(self):
        anchor = parse(fixture())['normalized']
        self.assertEqual(resources.require_stable_window(anchor, anchor)['vm']['pswpout'], 0)
        for key in ('pswpin', 'pswpout', 'memoryLimitHits', 'oomEventCount', 'oomKillCount', 'startTicks'):
            point = copy.deepcopy(anchor)
            if key in ('pswpin', 'pswpout'):
                point['vm'][key] += 1
            elif key == 'startTicks':
                point['processes']['1'][key] += 1
            else:
                point[key] += 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                resources.require_stable_window(anchor, point)

    def test_old_oom_events_do_not_hide_new_events(self):
        anchor = parse(fixture())['normalized']; anchor['oomKillCount'] = 1
        resources.require_stable_window(anchor, anchor)
        point = copy.deepcopy(anchor); point['oomKillCount'] += 1
        with self.assertRaises(ValueError):
            resources.require_stable_window(anchor, point)


if __name__ == '__main__':
    unittest.main(verbosity=2)
