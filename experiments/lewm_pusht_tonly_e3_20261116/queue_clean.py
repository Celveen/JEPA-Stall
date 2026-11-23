"""Wait for the verified epoch-3 model, then run clean CEM once on GPU 4."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
TRAIN = REPO / 'eval_outputs/lewm_pusht_tonly_e3_20261116'
QUEUE = REPO / 'eval_outputs/lewm_pusht_tonly_e3_clean_queue'
OUTPUT = REPO / 'eval_outputs/lewm_pusht_tonly_e3_clean50'
PYTHON = REPO / 'eval_outputs/rope_local/venv/bin/python'


def save(value):
    temporary = QUEUE / 'status.tmp'
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(QUEUE / 'status.json')


def run():
    status = {'status': 'waiting_for_training', 'pid': os.getpid(), 'gpu': 4,
              'model_epoch': 3, 'episodes': 50, 'started_unix': time.time()}
    save(status)
    try:
        smoke = json.loads((REPO / 'eval_outputs/lewm_pusht_prepare/clean_pipeline_smoke/complete.json').read_text())
        assert smoke['diagnostic_only'] and smoke['status'] == 'complete'
        while True:
            training = json.loads((TRAIN / 'status.json').read_text())
            assert training['status'] != 'failed', 'Training failed; evaluation must not proceed'
            if training['status'] == 'complete':
                break
            time.sleep(30)
        assert (TRAIN / 'training_result.json').exists()
        while True:
            inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used',
                                                 '--format=csv,noheader,nounits'], text=True)
            devices = {int(row.split(',')[0]): (row.split(',')[1].strip(), int(row.split(',')[2]))
                       for row in inventory.splitlines()}
            if devices[4][1] < 500:
                break
            status.update(status='waiting_for_gpu4', updated_unix=time.time()); save(status)
            time.sleep(30)
        assert not OUTPUT.exists(), 'Do not overwrite an existing evaluation'
        environment = os.environ.copy()
        environment.update(CUDA_VISIBLE_DEVICES=devices[4][0], OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                           MKL_NUM_THREADS='1', SDL_VIDEODRIVER='dummy', MPLCONFIGDIR='/tmp/lewm-pusht-matplotlib')
        command = [str(PYTHON), '-u', str(ROOT / 'evaluate_clean.py'), '--output', str(OUTPUT)]
        status.update(status='evaluating', command=command, evaluation_started_unix=time.time()); save(status)
        with (QUEUE / 'evaluate.log').open('w') as stream:
            result = subprocess.run(command, cwd=REPO, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        assert result.returncode == 0, f'Evaluation failed: {result.returncode}; inspect evaluate.log'
        receipt = json.loads((OUTPUT / 'complete.json').read_text())
        assert receipt['status'] == 'complete' and not receipt['diagnostic_only'] and receipt['episodes'] == 50
        status.update(status='complete', result=receipt, ended_unix=time.time())
    except Exception as exc:
        status.update(status='failed', error=repr(exc), ended_unix=time.time())
        save(status)
        raise
    save(status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.run:
        run()
        return
    QUEUE.mkdir(parents=True, exist_ok=False)
    with (QUEUE / 'supervisor.log').open('w') as stream:
        child = subprocess.Popen([str(PYTHON), str(Path(__file__).resolve()), '--run'], cwd=REPO,
                                 start_new_session=True, stdout=stream, stderr=subprocess.STDOUT)
    print(json.dumps({'queue_pid': child.pid, 'status': str(QUEUE / 'status.json')}))


if __name__ == '__main__':
    main()
