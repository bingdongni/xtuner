"""Run the real JSONL module and its tests without the project's GPU stack."""

import ast
import logging
import sys
import threading
import types
from multiprocessing import resource_tracker, shared_memory
from pathlib import Path
from typing import Generic, TypeVar

import pytest

root = Path.cwd()
for name, path in [
    ('xtuner', root / 'xtuner'),
    ('xtuner.v1', root / 'xtuner/v1'),
    ('xtuner.v1.datasets', root / 'xtuner/v1/datasets'),
    ('xtuner.v1.datasets.pt_tokenize_fn', root / 'xtuner/v1/datasets/pt_tokenize_fn'),
]:
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package

# Use the project's SharedMemory implementation without importing unrelated model utilities.
misc = ast.parse((root / 'xtuner/v1/utils/misc.py').read_text(encoding='utf-8'))
shared_memory_if = next(
    node for node in misc.body
    if isinstance(node, ast.If) and 'sys.version_info' in ast.unparse(node.test)
)
namespace = {'sys': sys, 'threading': threading, '_mpshm': shared_memory, '_mprt': resource_tracker}
exec(compile(ast.Module(body=[shared_memory_if], type_ignores=[]), 'misc.py', 'exec'), namespace)

utils = types.ModuleType('xtuner.v1.utils')
utils.__path__ = [str(root / 'xtuner/v1/utils')]
utils.SharedMemory = namespace['SharedMemory']
utils.get_logger = lambda: logging.getLogger('xtuner-test')
utils.log_rank0 = logging.getLogger('xtuner-test')
sys.modules[utils.__name__] = utils

dist_utils = types.ModuleType('xtuner.v1.utils.dist_utils')
dist_utils.get_local_process_group = lambda: None
dist_utils.get_local_world_size = lambda: 1
dist_utils.is_local_rank0 = lambda: True
sys.modules[dist_utils.__name__] = dist_utils

long_text = types.ModuleType('xtuner.v1.datasets.pt_tokenize_fn.long_text')
long_text.LongTextPretrainTokenizeFunction = type('LongTextPretrainTokenizeFunction', (), {})
sys.modules[long_text.__name__] = long_text

dataset_utils = types.ModuleType('xtuner.v1.datasets.utils')
T = TypeVar('T')


class CachableTokenizeFunction(Generic[T]):
    pass


dataset_utils.CachableTokenizeFunction = CachableTokenizeFunction
dataset_utils.calculate_xxhash = lambda data: ''
sys.modules[dataset_utils.__name__] = dataset_utils

raise SystemExit(pytest.main(['--noconftest', '-q', 'tests/datasets/test_jsonl_parallel.py']))
