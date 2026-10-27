"""Asset identity, model revisions, subprocess limits and strict media receipts."""
import hashlib
import json
import os
import stat
import subprocess
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from learning_os.hoj.auth import now_ms
from learning_os.hoj.contracts import VideoAsset, VideoSlice, ServiceError
from learning_os.hoj.video_extract import VideoExtractor, ENGINE, snapshot


@pytest.fixture
def media(tmp_path, monkeypatch):
    root=tmp_path/'assets';root.mkdir()
    data=b'\x00\x00\x00\x20ftypisom'+b'x'*100
    source=root/('a'*32+'.bin');source.write_bytes(data)
    model=tmp_path/'model';model.mkdir()
    for name in ['model.bin','config.json','tokenizer.json','vocabulary.txt']:(model/name).write_bytes(('local-'+name).encode())
    if os.name == 'nt':
        # Python 3.12 Windows path.stat reports creation time as ctime,
        # whereas fstat reports change time. This fixture emulates Linux;
        # compare real descriptor metadata on both sides without weakening
        # the production snapshot or mocking actual file mutations.
        class DescriptorStatPath(type(root)):
            def stat(self, *, follow_symlinks=True):
                native = super().stat(follow_symlinks=follow_symlinks)
                if not stat.S_ISREG(native.st_mode):
                    return native
                with self.open('rb') as stream:
                    return os.fstat(stream.fileno())

        root, model = DescriptorStatPath(root), DescriptorStatPath(model)
    monkeypatch.setattr('learning_os.hoj.video_extract.sys.platform','linux')
    extractor=VideoExtractor(root,model)
    ref=VideoAsset(assetId='a'*32,sourceHash=hashlib.sha256(data).hexdigest(),byteLength=len(data))
    return extractor,ref,source,model


def probe(ref):
    return ref.model_dump()|dict(durationMs=125000,mime='video/mp4',width=640,height=480,videoCodec='h264',audioCodec='aac',audioIndex=1)


def part(ref,**updates):
    return VideoSlice(**(ref.model_dump()|dict(startMs=60000,endMs=120000,language='zh')|updates))


def result(ref,model_hash,**updates):
    return ref.model_dump(exclude={'expectedModelHash'})|dict(durationMs=125000,language='zh',segments=[dict(startMs=60500,endMs=62000,text='前缀和用于区间求和。')],engine=ENGINE,modelHash=model_hash)|updates


@pytest.mark.parametrize('update',[
    dict(assetId='../private'),dict(assetId='a'*32+'\n'),dict(sourceHash='x'*64),dict(sourceHash='a'*64+'\n'),
    dict(byteLength=True),dict(byteLength=1024**3+1),dict(filePath='/identity/service.key'),
])
def test_asset_contract_cannot_be_a_path_or_oversized_upload(update):
    with pytest.raises(ValidationError):VideoAsset(**(dict(assetId='a'*32,sourceHash='b'*64,byteLength=200)|update))


@pytest.mark.parametrize('update',[dict(startMs=True),dict(startMs=60000,endMs=60000),dict(startMs=0,endMs=60001),dict(language='url'),dict(expectedModelHash='invalid')])
def test_slice_contract_limits_work_before_opening_a_source(update):
    with pytest.raises(ValidationError):VideoSlice(**(dict(assetId='a'*32,sourceHash='b'*64,byteLength=200,startMs=0,endMs=60000)|update))


def test_source_identity_is_verified_and_its_descriptor_closed_after_each_result(media):
    extractor,ref,path,_=media
    opened=[]
    def run(body,fd,*_):
        assert os.fstat(fd).st_size==ref.byteLength
        assert body['sourceFd']==fd and body['action']=='probe' and 'modelPath' not in body
        opened.append(fd)
        return json.dumps(probe(ref)).encode()
    with patch.object(extractor,'run',side_effect=run):
        assert extractor.extract(ref,now_ms()+60000)==probe(ref)
        assert extractor.extract(ref,now_ms()+60000)==probe(ref)
    for fd in opened:
        with pytest.raises(OSError):os.fstat(fd)
    path.write_bytes(path.read_bytes()+b'changed')
    with patch.object(extractor,'run') as worker:
        with pytest.raises(ServiceError) as error:extractor.extract(ref,now_ms()+60000)
        assert error.value.code=='VIDEO_SOURCE_CHANGED';worker.assert_not_called()


def test_changed_source_during_processing_is_never_a_success_receipt(media):
    extractor,ref,path,_=media
    def run(*_):
        path.write_bytes(path.read_bytes()+b'changed')
        return json.dumps(probe(ref)).encode()
    with patch.object(extractor,'run',side_effect=run):
        with pytest.raises(ServiceError) as error:extractor.extract(ref,now_ms()+60000)
    assert error.value.code=='VIDEO_SOURCE_CHANGED'


def test_same_length_source_change_with_restored_mtime_is_not_a_success_receipt(media):
    extractor,ref,path,_=media
    original = path.stat()

    def run(*_):
        before = path.read_bytes()
        path.write_bytes(before[:-1] + b'y')
        os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
        assert path.stat().st_size == ref.byteLength
        assert path.stat().st_mtime_ns == original.st_mtime_ns
        return json.dumps(probe(ref)).encode()

    with patch.object(extractor,'run',side_effect=run):
        with pytest.raises(ServiceError) as error:
            extractor.extract(ref,now_ms()+60000)
    assert error.value.code=='VIDEO_SOURCE_CHANGED'


def collide_ctime(monkeypatch, path):
    # Real Linux tmpfs accepted all ten rapid mutations in the deployment
    # smoke: even nanosecond stat fields can collide within one clock tick.
    # Keep real dev/inode/size/mtime and real bytes; simulate only that ctime.
    ctime = path.stat().st_ctime_ns
    monkeypatch.setattr('learning_os.hoj.video_extract.snapshot',
                        lambda state: snapshot(state)[:-1] + (ctime,))


def mutate_same_metadata(path):
    original = path.stat()
    body = path.read_bytes()
    path.write_bytes(body[:-1] + bytes([body[-1] ^ 1]))
    os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
    assert path.stat().st_size == original.st_size
    assert path.stat().st_mtime_ns == original.st_mtime_ns


@pytest.mark.parametrize('warm', [False, True])
def test_processing_source_change_with_metadata_collision_is_rejected(media, monkeypatch, warm):
    extractor, ref, path, _ = media
    collide_ctime(monkeypatch, path)
    if warm:
        with patch.object(extractor, 'run', return_value=json.dumps(probe(ref)).encode()):
            assert extractor.extract(ref, now_ms()+60000) == probe(ref)
    opened = []

    def changed(_body, fd, *_):
        opened.append(fd)
        mutate_same_metadata(path)
        return json.dumps(probe(ref)).encode()

    with patch.object(extractor, 'run', side_effect=changed), pytest.raises(ServiceError) as error:
        extractor.extract(ref, now_ms()+60000)
    assert error.value.code == 'VIDEO_SOURCE_CHANGED'
    for fd in opened:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_preexisting_source_change_with_metadata_collision_cannot_use_warm_cache(media, monkeypatch):
    extractor, ref, path, _ = media
    collide_ctime(monkeypatch, path)
    with patch.object(extractor, 'run', return_value=json.dumps(probe(ref)).encode()):
        assert extractor.extract(ref, now_ms()+60000) == probe(ref)
    mutate_same_metadata(path)
    with patch.object(extractor, 'run', return_value=json.dumps(probe(ref)).encode()) as worker:
        with pytest.raises(ServiceError) as error:
            extractor.extract(ref, now_ms()+60000)
    assert error.value.code == 'VIDEO_SOURCE_CHANGED'
    worker.assert_not_called()


def test_source_postcheck_rewinds_original_descriptor_offset(media):
    extractor, ref, _, _ = media

    def consumed(_body, fd, *_):
        assert os.lseek(fd, 0, os.SEEK_CUR) == 0
        assert os.read(fd, ref.byteLength)
        assert os.lseek(fd, 0, os.SEEK_CUR) == ref.byteLength
        return json.dumps(probe(ref)).encode()

    with patch.object(extractor, 'run', side_effect=consumed):
        assert extractor.extract(ref, now_ms()+60000) == probe(ref)


def test_replaced_source_path_with_identical_bytes_is_rejected(media):
    extractor, ref, path, _ = media

    def replaced(_body, _fd, *_):
        replacement = path.with_suffix('.replacement')
        replacement.write_bytes(path.read_bytes())
        os.replace(replacement, path)
        return json.dumps(probe(ref)).encode()

    with patch.object(extractor, 'run', side_effect=replaced), pytest.raises(ServiceError) as error:
        extractor.extract(ref, now_ms()+60000)
    assert error.value.code == 'VIDEO_SOURCE_CHANGED'


def test_model_revision_does_not_reuse_metadata_collision(media, monkeypatch):
    extractor, ref, _, model = media
    path = model/'config.json'
    collide_ctime(monkeypatch, path)
    version = extractor.model_hash(now_ms()+60000)
    mutate_same_metadata(path)
    assert extractor.model_hash(now_ms()+60000) != version
    with patch.object(extractor, 'run') as worker, pytest.raises(ServiceError) as error:
        extractor.extract(part(ref, expectedModelHash=version), now_ms()+110000)
    assert error.value.code == 'VIDEO_MODEL_CHANGED'
    worker.assert_not_called()


def test_model_change_during_processing_with_metadata_collision_is_rejected(media, monkeypatch):
    extractor, ref, _, model = media
    path = model/'config.json'
    collide_ctime(monkeypatch, path)
    version = extractor.model_hash(now_ms()+60000)
    selected = part(ref, expectedModelHash=version)

    def changed(*_):
        mutate_same_metadata(path)
        return json.dumps(result(selected, version)).encode()

    with patch.object(extractor, 'run', side_effect=changed), pytest.raises(ServiceError) as error:
        extractor.extract(selected, now_ms()+110000)
    assert error.value.code == 'VIDEO_MODEL_CHANGED'


def test_model_changes_cannot_mix_revisions_in_a_resumed_transcript(media):
    extractor,ref,_,model=media
    selected=part(ref);version=extractor.model_hash(now_ms()+60000)
    assert len(version)==64
    with patch.object(extractor,'run',return_value=json.dumps(result(selected,version)).encode()):
        assert extractor.extract(selected,now_ms()+110000)['modelHash']==version
    (model/'tokenizer.json').write_bytes(b'different local model tokenizer')
    with patch.object(extractor,'run') as worker:
        with pytest.raises(ServiceError) as error:extractor.extract(part(ref,expectedModelHash=version),now_ms()+110000)
        assert error.value.code=='VIDEO_MODEL_CHANGED';worker.assert_not_called()


@pytest.mark.parametrize('update',[
    dict(assetId='b'*32),dict(byteLength=True),dict(durationMs=True),dict(startMs=0),dict(endMs=119999),
    dict(language='private file'),dict(modelHash='f'*64),dict(engine='other'),dict(extra='not part of source'),
    dict(segments=[dict(startMs=59000,endMs=60001,text='outside slice')]),
    dict(segments=[dict(startMs=60000,endMs=125000,text='outside slice')]),
    dict(segments=[dict(startMs=60500,endMs=62000,text='bad\0text')]),
])
def test_corrupt_or_cross_slice_receipts_are_not_trusted(media,update):
    extractor,ref,_,_=media
    request=part(ref);version=extractor.model_hash(now_ms()+60000)
    with patch.object(extractor,'run',return_value=json.dumps(result(request,version,**update)).encode()):
        with pytest.raises(ServiceError) as error:extractor.extract(request,now_ms()+110000)
    assert error.value.code=='INVALID_VIDEO_RESULT'


def test_silence_is_a_valid_empty_checkpoint_and_fixed_failures_do_not_reflect_paths(media):
    extractor,ref,_,_=media
    request=part(ref);version=extractor.model_hash(now_ms()+60000)
    with patch.object(extractor,'run',return_value=json.dumps(result(request,version,segments=[],language='und')).encode()):
        assert extractor.extract(request,now_ms()+110000)['segments']==[]
    for code,status in [('VIDEO_NO_AUDIO',400),('VIDEO_RESOURCE_LIMIT',400),('VIDEO_ASR_FAILED',503),('VIDEO_RUNTIME_UNAVAILABLE',503)]:
        with pytest.raises(ServiceError) as error:extractor.validate(json.dumps({'code':code}).encode(),request,version)
        assert (error.value.status,error.value.code)==(status,code)
    with pytest.raises(ServiceError) as error:extractor.validate(b'{"code":"VIDEO_INVALID","path":"private"}',request,version)
    assert error.value.code=='INVALID_VIDEO_RESULT'


def test_subprocess_has_only_the_source_descriptor_and_no_service_or_provider_credentials(media,monkeypatch):
    extractor,_,_,_=media
    monkeypatch.setenv('HOJ_RAG_SHARED_KEY','must-not-reach-child');monkeypatch.setenv('OPENAI_API_KEY','must-not-reach-child')
    child=MagicMock(pid=43211,returncode=0);child.poll.return_value=0;child.communicate.return_value=(b'{}',b'')
    with patch('learning_os.hoj.video_extract.subprocess.Popen',return_value=child) as spawn,patch('learning_os.hoj.video_extract.os.killpg',create=True):
        assert extractor.run({'sourceFd':17},17,now_ms()+110000,True)==b'{}'
    args=spawn.call_args.kwargs
    assert args['pass_fds']==(17,) and args['start_new_session']
    assert not {'HOJ_RAG_SHARED_KEY','OPENAI_API_KEY'}&args['env'].keys()
    assert args['env']['HF_HUB_OFFLINE']=='1' and args['env']['OMP_NUM_THREADS']=='2'
    assert child.communicate.call_args.kwargs['timeout']==95


def test_timeout_kills_the_whole_decoder_model_group_and_reaps_the_child(media):
    extractor,_,_,_=media
    child=MagicMock(pid=43211,returncode=-9);child.poll.return_value=-9
    child.communicate.side_effect=[subprocess.TimeoutExpired('fixture',1),(b'',b'')]
    with patch('learning_os.hoj.video_extract.subprocess.Popen',return_value=child),patch('learning_os.hoj.video_extract.os.killpg',create=True) as kill:
        with pytest.raises(ServiceError) as error:extractor.run({'sourceFd':17},17,now_ms()+110000,True)
        assert error.value.code=='VIDEO_RESOURCE_LIMIT';assert kill.call_count>=1
        child.stdin.close.assert_called_once();child.stdout.close.assert_called_once()
