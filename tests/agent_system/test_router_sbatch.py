"""Execute the Slurm workflow on CPU, mocking cluster and inference boundaries."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from test_router_sft import ROOT, record


class RouterSbatchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        self.env = dict(os.environ, WORK_DIR=str(self.root), SLURM_SUBMIT_DIR=str(self.root),
                        SLURM_JOB_ID='12345', CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
                        N_GPUS='8', COLLECT_STEPS='2', SAMPLES_PER_STEP='4096',
                        SEARCH_URL='http://existing-retriever/retrieve', SFT_SUCCESS_ONLY='0', SFT_ROUTE_ONLY='0',
                        SFT_VAL_FRACTION='0.2', SFT_MAX_LENGTH='8192', SFT_SEED='42',
                        SFT_MAX_PER_QUESTION_ROUTE='4', PATH=str(self.bin) + os.pathsep + os.environ['PATH'])
        for name in ('MODEL_DIR', 'SFT_TOKENIZER', 'RUN_DIR', 'TRAIN_ENV', 'SEARCH_DATA_DIR', 'MATH_DATA_DIR',
                     'COLLECTION_JOB_ID', 'GPU_PARTITION', 'CPU_PARTITION', 'SBATCH_FAIL_CALL', 'SBATCH_CLUSTER'):
            self.env.pop(name, None)
        self.run_dir = self.root / 'outputs/router-collect-both-12345'
        self.executable('module', '#!/bin/bash\nexit 0\n')
        self.executable('nvidia-smi', '#!/bin/bash\nexit 0\n')
        self.executable('mktemp', '#!/bin/bash\nmkdir -p "$WORK_DIR/tmp"\nprintf "%s\\n" "$WORK_DIR/tmp"\n')
        self.executable('conda', '#!/bin/bash\nprintf "%s\\n" "$WORK_DIR/conda"\n')
        conda_sh = self.root / 'conda/etc/profile.d/conda.sh'
        conda_sh.parent.mkdir(parents=True)
        conda_sh.write_text('conda() { export CONDA_PREFIX="$WORK_DIR/envs/DrMAS"; }\n')
        python = self.root / 'envs/DrMAS/bin/python'
        python.parent.mkdir(parents=True)
        # Run the real shell/preflight/builder. Only unavailable libraries are
        # stubbed; Parquet encoding and model tokenization are not tested here.
        python.write_text(f'#!{sys.executable}\n' + r'''
import json
import os
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

class Tokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        assert os.environ['CUDA_VISIBLE_DEVICES'] == '', 'Tokenization must run in the CPU stage'
        return cls()
    def apply_chat_template(self, messages, **kwargs):
        return list(range(sum(len(m['content']) for m in messages)))

def ServerArgs(disable_cuda_graph=None, cuda_graph_max_bs=None, max_running_requests=None,
               chunked_prefill_size=None, max_prefill_tokens=None, max_total_tokens=None,
               context_length=None, enable_memory_saver=None):
    pass

for name in ('transformers', 'pyarrow', 'pyarrow.parquet', 'sglang', 'sglang.srt', 'sglang.srt.server_args'):
    sys.modules[name] = ModuleType(name)
sys.modules['transformers'].AutoTokenizer = Tokenizer
sys.modules['pyarrow'].Table = SimpleNamespace(from_pylist=lambda rows: rows)
sys.modules['pyarrow.parquet'].write_table = lambda rows, path: path.write_text(json.dumps(rows))
sys.modules['sglang.srt.server_args'].ServerArgs = ServerArgs
sys.argv = sys.argv[1:]
if sys.argv[0] == '-':
    exec(compile(sys.stdin.read(), '<job-stdin>', 'exec'), {'__name__': '__main__'})
else:
    runpy.run_path(sys.argv[0], run_name='__main__')
''')
        python.chmod(0o755)
        model = self.root / 'model/Qwen2.5-3B-Instruct'
        model.mkdir(parents=True)
        (model / 'config.json').write_text('{}')
        for scene in ('search', 'math'):
            data = self.root / f'data/drmas_{scene}'
            data.mkdir(parents=True)
            (data / 'train.parquet').touch()  # Deliberately no validation dataset.
            launcher = self.root / f'examples/gmas_trainer/run_{scene}.sh'
            launcher.parent.mkdir(parents=True, exist_ok=True)
            launcher.write_text(f'#!/bin/bash\nexec "$CONDA_PREFIX/bin/python" "$WORK_DIR/collector.py" {scene} "$@"\n')
            sources = [record('<think>Original explanation.</think>\n' + ('SEARCH' if scene == 'search' else 'SOLVE'),
                              scene=scene, question=f'Question {i}') for i in range(40)]
            for i, source in enumerate(sources):
                source['episode_pass'] = i % 2
            sources.extend([dict(sources[0]), record('<route>VERFIY</route>', scene=scene)])
            episodes = [{'pass': source['episode_pass'], 'data_source': scene, 'turns': [
                {'agent_id': 'Router Agent', 'turn': 1, 'input': source['input'],
                 'output': source['original_output'], 'is_action_valid': False},
            ]} for source in sources]
            (self.root / f'{scene}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in episodes))
        for relative in ('examples/sft/router/collect.sh', 'examples/sft/router/sbatch_collect.sh',
                         'examples/sft/router/sbatch_build.sh', 'examples/data_preprocess/router_sft.py',
                         'agent_system/code_actions.py'):
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text((ROOT / relative).read_text())
        (self.root / 'collector.py').write_text(r'''
import json
import os
import sys
from pathlib import Path

scene = sys.argv[1]
args = dict(arg.lstrip('+').split('=', 1) for arg in sys.argv[3:] if '=' in arg)
assert args['trainer.rollout_only'] == 'True'
root = Path(os.environ['WORK_DIR'])
with (root / 'events.jsonl').open('a') as stream:
    stream.write(json.dumps({'scene': scene, 'args': args}) + '\n')
output = Path(args['trainer.benchmark.output_dir'])
logs = output / 'rollout_samples'
logs.mkdir(parents=True)
steps = int(args['trainer.total_training_steps'])
for step in range(1, steps + 1):
    (logs / f'step_{step:06d}.jsonl').write_text((root / f'{scene}.jsonl').read_text())
    (logs / f'step_{step:06d}.txt').write_text('Do not feed TXT duplicates to the builder.')
(output / 'summary.json').write_text(json.dumps({
    'complete': os.environ.get('INCOMPLETE_SCENE') != scene, 'completed_steps': steps,
}))
metrics = {'metrics': {'collection/parameter_updates': int(os.environ.get('PARAMETER_UPDATES', '0'))}}
(output / 'metrics.jsonl').write_text((json.dumps(metrics) + '\n') * steps)
if os.environ.get('FAIL_SCENE') == scene:
    sys.exit(17)
''')
        self.executable('sbatch', f'#!{sys.executable}\n' + r'''
import json
import os
import sys
from pathlib import Path

path = Path(os.environ['WORK_DIR']) / 'submissions.jsonl'
previous = path.read_text().splitlines() if path.exists() else []
number = len(previous) + 1
with path.open('a') as stream:
    stream.write(json.dumps({'args': sys.argv[1:], 'run_dir': os.environ.get('RUN_DIR'),
                            'collection_job_id': os.environ.get('COLLECTION_JOB_ID'),
                            'success_only': os.environ.get('SFT_SUCCESS_ONLY')}) + '\n')
if os.environ.get('SBATCH_FAIL_CALL') == str(number):
    sys.exit(19)
print(str(10000 + number) + (';' + os.environ['SBATCH_CLUSTER'] if os.environ.get('SBATCH_CLUSTER') else ''))
''')

    def executable(self, name, content):
        path = self.bin / name
        path.write_text(content)
        path.chmod(0o755)

    def run_gpu(self, scene='both', *overrides):
        return subprocess.run(['bash', str(ROOT / 'examples/sft/router/sbatch_collect.sh'), scene, *overrides],
                              env=self.env, text=True, capture_output=True, timeout=30)

    def run_cpu(self, scene='both'):
        env = dict(self.env, SLURM_JOB_ID='12346', CUDA_VISIBLE_DEVICES='',
                   RUN_DIR=self.env.get('RUN_DIR', str(self.root / f'outputs/router-collect-{scene}-12345')))
        return subprocess.run(['bash', str(ROOT / 'examples/sft/router/sbatch_build.sh'), scene],
                              env=env, text=True, capture_output=True, timeout=30)

    def run_job(self, scene='both', *overrides):
        gpu = self.run_gpu(scene, *overrides)
        if gpu.returncode:
            return gpu
        output = Path(self.env.get('RUN_DIR', self.root / f'outputs/router-collect-{scene}-12345'))
        self.assertFalse((output / 'sft').exists(), 'GPU job must finish without constructing SFT')
        self.assertFalse((output / 'build.log').exists())
        manifest = json.loads((output / 'collection_complete.json').read_text())
        self.assertEqual(manifest['sampling_job_id'], '12345')
        return self.run_cpu(scene)

    def submit(self, scene='both', *overrides):
        return subprocess.run(['bash', str(ROOT / 'examples/sft/router/submit_collect.sh'), scene, *overrides],
                              env=self.env, text=True, capture_output=True, timeout=30)

    def test_both_scenes_collect_then_build_and_report(self):
        result = self.run_job('both', 'trainer.rollout_only=False', 'trainer.total_training_steps=999',
                              'trainer.benchmark.output_dir=/wrong-output', 'env.rollout.n=3')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        events = [json.loads(line) for line in (self.root / 'events.jsonl').read_text().splitlines()]
        self.assertEqual([event['scene'] for event in events], ['search', 'math'])
        for event in events:
            self.assertEqual(event['args']['env.rollout.n'], '3')
            self.assertEqual(event['args']['trainer.total_training_steps'], '2')
        output = self.run_dir / 'sft'
        report = json.loads((output / 'report.json').read_text())
        self.assertEqual(report['router_calls'], 168)
        self.assertEqual(report['accepted_examples'], 80)
        self.assertEqual(report['accepted_questions'], 40)
        self.assertEqual(report['accepted_data_sources'], {'search': 40, 'math': 40})
        self.assertEqual(report['accepted_episode_outcomes'], {'failure': 40, 'success': 40})
        self.assertEqual(sum(report['filter_counts'].values()) + report['accepted_examples'], report['router_calls'])
        self.assertEqual(report['filter_counts']['unknown_role'], 4)
        self.assertEqual(len(report['source_files']), 4)
        self.assertTrue(report['length_checked'] and report['ready_for_training'])
        questions = []
        for split in ('train', 'val'):
            rows = [json.loads(line) for line in (output / f'{split}.jsonl').read_text().splitlines()]
            self.assertTrue((output / f'{split}.parquet').is_file())
            self.assertEqual(report['splits'][split]['examples'], len(rows))
            self.assertEqual(report['splits'][split]['token_lengths']['max'], max(row['token_length'] for row in rows))
            questions.append({row['question_id'] for row in rows})
        self.assertFalse(questions[0] & questions[1])
        self.assertIn('最终保留: 80', (output / 'summary.txt').read_text())
        self.assertIn('最终保留: 80', result.stdout)
        rerun = self.run_job()
        self.assertNotEqual(rerun.returncode, 0)
        self.assertEqual(len((self.root / 'events.jsonl').read_text().splitlines()), 2)

    def test_sampling_failure_stops_before_build(self):
        self.env['FAIL_SCENE'] = 'math'
        result = self.run_job()
        self.assertEqual(result.returncode, 17, result.stdout + result.stderr)
        self.assertFalse((self.run_dir / 'sft').exists())
        self.assertFalse((self.run_dir / 'collection_complete.json').exists())
        self.assertTrue((self.run_dir / 'search/observations/rollout_samples/step_000001.jsonl').exists())

    def test_incomplete_or_updating_collection_is_rejected(self):
        for key, value in [('INCOMPLETE_SCENE', 'search'), ('PARAMETER_UPDATES', '1')]:
            with self.subTest(key=key):
                self.env['RUN_DIR'] = str(self.root / key)
                self.env[key] = value
                result = self.run_job()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('Incomplete collection' if key == 'INCOMPLETE_SCENE'
                              else 'Expected completed fixed-model collection metrics', result.stderr)
                self.assertFalse((self.root / key / 'sft').exists())
                self.env.pop(key)

    def test_cpu_requires_completion_and_rechecks_saved_files(self):
        result = self.run_cpu()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('collection_complete.json', result.stderr)
        gpu = self.run_gpu()
        self.assertEqual(gpu.returncode, 0, gpu.stdout + gpu.stderr)
        (self.run_dir / 'math/observations/rollout_samples/step_000002.jsonl').unlink()
        cpu = self.run_cpu()
        self.assertNotEqual(cpu.returncode, 0)
        self.assertIn('Missing or empty trajectory files', cpu.stderr)
        self.assertFalse((self.run_dir / 'sft').exists())

    def test_submit_connects_separate_gpu_and_cpu_allocations(self):
        self.env.update(SFT_SUCCESS_ONLY='1', SBATCH_CLUSTER='orion')
        result = self.submit('both', 'env.rollout.n=3')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = [json.loads(line) for line in (self.root / 'submissions.jsonl').read_text().splitlines()]
        self.assertEqual(len(calls), 2)
        gpu, cpu = calls
        self.assertIn('--gres=gpu:8', gpu['args'])
        self.assertIn('--partition=h200p', gpu['args'])
        self.assertEqual(gpu['args'][-2:], ['both', 'env.rollout.n=3'])
        self.assertIn('--gres=none', cpu['args'])
        self.assertIn('--partition=cpup', cpu['args'])
        self.assertIn('--dependency=afterok:10001', cpu['args'])
        self.assertIn('--kill-on-invalid-dep=yes', cpu['args'])
        self.assertIn('--clusters=orion', cpu['args'])
        self.assertNotIn('env.rollout.n=3', cpu['args'])
        self.assertEqual(cpu['collection_job_id'], '10001')
        self.assertEqual(Path(cpu['run_dir']).resolve(), self.root.resolve() / 'outputs/router-collect-both-10001')
        self.assertEqual(gpu['success_only'], cpu['success_only'])
        self.assertFalse(Path(cpu['run_dir']).exists(), 'Submission must not occupy the fresh GPU output directory')
        info = Path(cpu['run_dir'] + '.jobs.txt').read_text()
        self.assertIn('CPU_JOB_ID=10002', info)
        self.assertIn('GPU_JOB_ID=10001', info)

    def test_submit_forwards_custom_shared_directory_and_resources(self):
        self.env.update(RUN_DIR='outputs/custom run', CPU_CPUS='4', CPU_MEM='64G',
                        CPU_TIME='02:00:00', GPU_PARTITION='testqueue', GPU_TIME='02:00:00')
        result = self.submit('math')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        gpu, cpu = [json.loads(line) for line in (self.root / 'submissions.jsonl').read_text().splitlines()]
        self.assertEqual(gpu['run_dir'], cpu['run_dir'])
        self.assertEqual(Path(cpu['run_dir']).resolve(), self.root.resolve() / 'outputs/custom run')
        self.assertIn('--partition=testqueue', gpu['args'])
        self.assertIn('--mem=64G', cpu['args'])
        self.assertIn('--cpus-per-task=4', cpu['args'])
        self.assertIn('--time=02:00:00', cpu['args'])

    def test_submit_reports_partial_submission_without_losing_gpu_job(self):
        self.env['SBATCH_FAIL_CALL'] = '2'
        result = self.submit()
        self.assertEqual(result.returncode, 19)
        self.assertIn('GPU job 10001 is already submitted', result.stderr)
        info = self.root / 'outputs/router-collect-both-10001.jobs.txt'
        self.assertIn('CPU_SUBMISSION_FAILED=true', info.read_text())

    def test_submit_does_not_queue_cpu_when_gpu_submission_fails(self):
        self.env['SBATCH_FAIL_CALL'] = '1'
        result = self.submit()
        self.assertEqual(result.returncode, 19)
        self.assertEqual(len((self.root / 'submissions.jsonl').read_text().splitlines()), 1)

    def test_math_only_success_filter_and_route_only(self):
        self.env.update(SFT_SUCCESS_ONLY='1', SFT_ROUTE_ONLY='1')
        result = self.run_job('math')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        output = self.root / 'outputs/router-collect-math-12345/sft'
        report = json.loads((output / 'report.json').read_text())
        self.assertEqual(report['accepted_examples'], 20)
        self.assertEqual(report['accepted_episode_outcomes'], {'success': 20})
        for split in ('train', 'val'):
            for line in (output / f'{split}.jsonl').read_text().splitlines():
                self.assertEqual(json.loads(line)['messages'][-1]['content'], "call_agent(role='SOLVE', task='')")

    def test_empty_dataset_keeps_diagnostics_and_fails_job(self):
        self.env['SFT_MAX_LENGTH'] = '1'
        result = self.run_job()
        self.assertNotEqual(result.returncode, 0)
        output = self.run_dir / 'sft'
        report = json.loads((output / 'report.json').read_text())
        self.assertEqual(report['accepted_examples'], 0)
        self.assertFalse(report['ready_for_training'])
        self.assertTrue((output / 'summary.txt').is_file())
        self.assertTrue((output / 'audit.jsonl').is_file())
        self.assertIn('overlong_example', report['filter_counts'])


if __name__ == '__main__':
    unittest.main()
