import ast
import json
import multiprocessing
import os
import queue
import signal
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from multiprocessing import Process, Queue, resource_tracker, shared_memory
from pathlib import Path


def load_production_functions():
    path = Path('xtuner/v1/datasets/jsonl.py')
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    names = {'tokenize_worker', 'chunk_data_to_queue', 'parallel_execute'}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in functions} == names

    class Logger:
        def debug(self, *args):
            pass

    class UntrackedSharedMemory(shared_memory.SharedMemory):
        def __init__(self, *args, **kwargs):
            register = resource_tracker.register
            resource_tracker.register = lambda *args: None
            try:
                super().__init__(*args, **kwargs)
            finally:
                resource_tracker.register = register

    namespace = {
        '__name__': '__main__', 'Callable': Callable, 'Queue': Queue, 'Process': Process,
        'SharedMemory': UntrackedSharedMemory, 'multiprocessing': multiprocessing,
        'os': os, 'time': time, 'traceback': traceback, 'Empty': queue.Empty,
        'logger': Logger(), 'get_local_world_size': lambda: 1,
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['parallel_execute']


def tokenize(data):
    if data == b'crash':
        os._exit(17)
    if data == b'slow':
        time.sleep(30)
    return {'value': json.loads(data)}


def child(mode, shm_name, offsets, baseline):
    parallel_execute = load_production_functions()
    try:
        result = parallel_execute(tokenize, json.loads(offsets), shm_name, nproc=2, chunksize=1, rank=0)
    except RuntimeError as error:
        if mode == 'bad':
            assert 'JSONDecodeError' in str(error), str(error)
        elif mode == 'crash':
            assert '17' in str(error), str(error)
        else:
            raise
    else:
        assert mode == 'success', 'failed input unexpectedly succeeded'
        assert result == [{'value': 0}, {'value': 1}, {'value': 2}, {'value': 3}], result
    if baseline and mode == 'success':
        for process in multiprocessing.active_children():
            process.join(timeout=2)
    assert not multiprocessing.active_children(), 'tokenizer child processes remain alive'
    print(f'{mode}: completed and cleaned up')


def run(mode, expect_timeout, baseline):
    records = {'success': [b'0', b'1', b'2', b'3'],
               'bad': [b'bad', b'slow'], 'crash': [b'crash', b'slow']}[mode]
    offsets = []
    start = 0
    for index, record in enumerate(records):
        offsets.append((index, (start, start + len(record))))
        start += len(record)
    segment = shared_memory.SharedMemory(create=True, size=start)
    segment.buf[:] = b''.join(records)
    process = subprocess.Popen(
        [sys.executable, __file__, 'child', mode, segment.name, json.dumps(offsets), str(int(baseline))],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True,
    )
    try:
        try:
            stdout, stderr = process.communicate(timeout=4 if expect_timeout else 12)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate(timeout=5)
            if not expect_timeout:
                raise AssertionError(f'{mode}: hung: {stderr}')
            if mode == 'bad':
                assert 'JSONDecodeError' in stderr, stderr
            print(f'{mode}: reproduced a hang after worker failure')
        else:
            if expect_timeout:
                raise AssertionError(f'{mode}: expected a hang, got exit {process.returncode}: {stderr}')
            assert process.returncode == 0, f'{mode}: exit {process.returncode}: {stdout} {stderr}'
            print(stdout.strip())
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        segment.close()
        try:
            segment.unlink()
        except FileNotFoundError:
            pass


if __name__ == '__main__':
    if sys.argv[1] == 'child':
        child(sys.argv[2], sys.argv[3], sys.argv[4], bool(int(sys.argv[5])))
    else:
        baseline = sys.argv[1] == 'baseline'
        run('success', False, baseline)
        run('bad', baseline, baseline)
        run('crash', baseline, baseline)
