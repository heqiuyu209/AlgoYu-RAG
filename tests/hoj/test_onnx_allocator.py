"""Native allocator ownership, ABI and unsupported-platform cleanup boundaries."""
import ctypes as C
from types import SimpleNamespace
import pytest
from learning_os.hoj import onnx_embedder as module


class Function:
    def __init__(self, value=None):
        self.value, self.calls = value, []
    def __call__(self, *args):
        self.calls.append(args)
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


def fixture_api(monkeypatch, *, result=1, owner='same', lookup=1, missing=False):
    trim, version = Function(result), Function(b'2.36')
    malloc, free = Function(), Function()
    functions = {101: trim, 102: version, 103: malloc, 104: free}
    addresses = {id(value): address for address, value in functions.items()}
    def dladdr(address, pointer):
        value = pointer._obj
        value.filename = b'/lib/libc.so.6' if owner != 'different-library' or address.value != 103 else b'/lib/other/libc.so.6'
        value.base = 500 if owner != 'different-base' or address.value != 103 else 600
        return lookup
    class Resolver(Function):
        def __call__(self, *args):
            self.calls.append(args)
            return dladdr(*args)
    resolver = Resolver()
    process = SimpleNamespace(dladdr=resolver, malloc=malloc, free=free)
    libc = SimpleNamespace(gnu_get_libc_version=version)
    if not missing:
        libc.malloc_trim = trim
    fake = SimpleNamespace(**{name: getattr(C, name) for name in
        ('Structure', 'c_char_p', 'c_void_p', 'c_int', 'c_size_t', 'POINTER', 'byref')})
    fake.CDLL = lambda name: process if name is None else libc
    fake.cast = lambda function, kind: SimpleNamespace(value=addresses[id(function)])
    monkeypatch.setattr(module, 'ctypes', fake)
    monkeypatch.setattr(module, 'sys', SimpleNamespace(platform='linux'))
    return trim, version, malloc, free, resolver


@pytest.mark.parametrize('returned,status', [(0, 'nothing_released'), (1, 'released_some')])
def test_trim_uses_correct_abi_once_and_zero_is_a_normal_result(monkeypatch, returned, status):
    trim, version, malloc, free, resolver = fixture_api(monkeypatch, result=returned)
    assert module._release_unused_cpu_pages() == dict(status=status, glibcVersion='2.36')
    assert trim.argtypes == [C.c_size_t] and trim.restype is C.c_int
    assert resolver.argtypes[0] is C.c_void_p and resolver.restype is C.c_int
    assert version.argtypes == [] and version.restype is C.c_char_p
    assert trim.calls == [(0,)] and version.calls == [()]
    assert malloc.calls == free.calls == []


@pytest.mark.parametrize('owner', ['different-library', 'different-base'])
def test_interposed_allocator_is_not_trimmed(monkeypatch, owner):
    trim, _, malloc, free, _ = fixture_api(monkeypatch, owner=owner)
    assert module._release_unused_cpu_pages()['status'] == 'unavailable'
    assert trim.calls == malloc.calls == free.calls == []


@pytest.mark.parametrize('case', ['missing', 'unresolved'])
def test_missing_api_or_unresolved_owner_never_calls_trim(monkeypatch, case):
    trim, *_ = fixture_api(monkeypatch, missing=case == 'missing', lookup=0 if case == 'unresolved' else 1)
    assert module._release_unused_cpu_pages()['status'] == 'unavailable'
    assert trim.calls == []


def test_other_platforms_do_not_load_libc(monkeypatch):
    monkeypatch.setattr(module, 'sys', SimpleNamespace(platform='win32'))
    def forbidden(*args):
        raise AssertionError('libc must not be loaded')
    monkeypatch.setattr(module, 'ctypes', SimpleNamespace(CDLL=forbidden))
    assert module._release_unused_cpu_pages() == dict(status='unavailable', reason='platform')


def test_libc_load_failure_is_an_optional_cleanup_capability(monkeypatch):
    trim, *_ = fixture_api(monkeypatch)
    def unavailable(*args):
        raise OSError('no glibc fixture')
    module.ctypes.CDLL = unavailable
    assert module._release_unused_cpu_pages() == dict(status='unavailable', reason='allocator-api')
    assert trim.calls == []


def test_invalid_native_result_is_not_reported_as_released(monkeypatch):
    trim, *_ = fixture_api(monkeypatch, result=2)
    with pytest.raises(ValueError, match='allocator release'):
        module._release_unused_cpu_pages()
    assert trim.calls == [(0,)]
