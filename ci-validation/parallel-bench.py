import importlib.util
import multiprocessing
import statistics
import subprocess
import time
from multiprocessing import shared_memory


spec = importlib.util.spec_from_file_location('parallel_smoke', 'ci-validation/parallel-smoke.py')
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)
baseline_source = subprocess.check_output(
    ['git', 'show', 'HEAD^:xtuner/v1/datasets/jsonl.py'], text=True, encoding='utf-8'
)
implementations = {
    'base': smoke.load_production_functions(baseline_source),
    'fix': smoke.load_production_functions(),
}
count = 100_000
offsets = [(index, (index, index + 1)) for index in range(count)]
segment = shared_memory.SharedMemory(create=True, size=count)
segment.buf[:] = b'1' * count
samples = {'base': [], 'fix': []}
try:
    for order in [('base', 'fix'), ('fix', 'base')] * 2:
        for name in order:
            start = time.perf_counter()
            output = implementations[name](smoke.tokenize, offsets, segment.name, 2, 10, 0)
            for process in multiprocessing.active_children():
                process.join()
            elapsed = time.perf_counter() - start
            assert len(output) == count and output[0] == {'value': 1} and output[-1] == {'value': 1}
            samples[name].append(elapsed)
finally:
    segment.close()
    segment.unlink()

for name, values in samples.items():
    print(name, 'seconds:', ', '.join(f'{value:.3f}' for value in values),
          'median:', f'{statistics.median(values):.3f}')
print('fix/base median:', f'{statistics.median(samples["fix"]) / statistics.median(samples["base"]):.3f}')
