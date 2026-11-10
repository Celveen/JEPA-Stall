"""Durable GPU0-only queue, integration gate, paired analysis and result publication."""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

from analyze import analyze_pair, summarize

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
REPORT = REPO / 'reports/rope_ecp_20261110'
OUTPUT = REPO / 'eval_outputs/rope_ecp_20261110'
PYTHON = '/data/zjc/envs/cljepa/bin/python'
RUNTIME = Path('/data/zjc/adaJEPA/CL-JEPA-cudagraph')
BRANCH = 'codex/cvpr2027-paper-results-v14'


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def runtime_sources():
    paths = [HERE / name for name in ['rope_ecp.py', 'run_chunk.py']]
    paths += [RUNTIME / name for name in ['plan.py', 'planning/mpc.py', 'planning/cem.py',
              'planning/objectives.py', 'planning/evaluator.py', 'preprocessor.py',
              'models/visual_world_model.py', 'models/dino.py',
              'env/deformable_env/FlexEnvWrapper.py', 'env/serial_vector_env.py',
              'env/deformable_env/src/sim/sim_env/flex_env.py']]
    return {str(path): sha256(path) for path in paths}


def create_environment(gpu):
    result = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free',
                                      '--format=csv,noheader,nounits'], text=True)
    selected = next(line.split(',') for line in result.splitlines()
                    if int(line.split(',')[0]) == gpu)
    if int(selected[2]) < 16000:
        raise RuntimeError('Selected GPU has less than 16 GB free; existing jobs preserved.')
    flex = Path('/data/zjc/adaJEPA/PyFleX')
    environment = dict(os.environ)
    environment.update({'CUDA_VISIBLE_DEVICES': selected[1].strip(), 'EGL_GPU': str(gpu),
                        'PYFLEXROOT': str(flex), 'DATASET_DIR': '/data/zjc/adaJEPA/CL-JEPA/data',
                        'TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD': '1', 'WANDB_MODE': 'offline',
                        'SDL_VIDEODRIVER': 'dummy', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'})
    environment['PYTHONPATH'] = str(flex / 'bindings/build') + ':' + environment.get('PYTHONPATH', '')
    environment['LD_LIBRARY_PATH'] = str(flex / 'external/SDL2-2.0.4/lib/x64') + ':' + environment.get('LD_LIBRARY_PATH', '')
    return environment


def run_job(arm, chunk, integration, environment, state, sources):
    output = OUTPUT / ('integration' if integration else '') / arm / f'chunk_{chunk:02d}'
    if (output / 'complete.json').exists():
        return output
    if runtime_sources() != sources:
        raise RuntimeError('Frozen runtime sources changed; refusing to mix implementations.')
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f'Incomplete output preserved; inspect before restart: {output}')
    command = [PYTHON, '-u', str(HERE / 'run_chunk.py'), '--arm', arm,
               '--chunk', str(chunk), '--output', str(output)]
    if integration:
        command += ['--first-only', '--force-trigger']
    log_dir = OUTPUT / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f'{"integration_" if integration else ""}{arm}_chunk_{chunk:02d}.log'
    with log.open('w') as stream:
        process = subprocess.Popen(command, cwd=RUNTIME, env=environment,
                                   stdout=stream, stderr=subprocess.STDOUT)
        state.update({'stage': 'integration' if integration else 'evaluating',
                      'active_arm': arm, 'active_chunk': chunk, 'worker_pid': process.pid,
                      'active_log': str(log), 'active_output': str(output), 'updated_utc': timestamp()})
        write_json(REPORT / 'pipeline_status.json', state)
        while process.poll() is None:
            time.sleep(10)
            state['updated_utc'] = timestamp()
            write_json(REPORT / 'pipeline_status.json', state)
        if process.returncode != 0 or not (output / 'complete.json').exists():
            raise RuntimeError(f'{arm} chunk {chunk} failed; exit={process.returncode}; log={log}')
    return output


def package_results():
    paths = []
    for arm in ['native', 'S']:
        for chunk in range(5):
            folder = OUTPUT / arm / f'chunk_{chunk:02d}'
            paths += [folder / name for name in ['complete.json', 'run_manifest.json',
                      'frozen_model_audit.json', 'ecp_records.json', 'ecp_costs.json',
                      'rope_metrics.npz', 'final_eval_results.npz', 'logs.json', 'prefix_reuse.json']]
            if arm == 'native':
                paths.append(folder / 'native_execution_cache.npz')
    targets = [RUNTIME / f'eval_outputs/rope_cl_single/stream/chunk_{chunk}.pkl' for chunk in range(5)]
    members = [(path, str(path.relative_to(REPO))) for path in paths]
    members += [(path, 'targets/' + path.name) for path in targets]
    manifest = [{'path': name, 'sha256': sha256(path), 'bytes': path.stat().st_size}
                for path, name in members]
    write_json(REPORT / 'archive_manifest.json', {'files': manifest,
               'scope': 'both arms; original fixed targets; actions; particle CD; model invariance; no weights'})
    archive = REPORT / 'paired_rope_results.tar.gz'
    with tarfile.open(archive, 'w:gz') as bundle:
        for path, name in members:
            bundle.add(path, arcname=name)
        bundle.add(REPORT / 'archive_manifest.json', arcname='ARCHIVE_MANIFEST.json')
    return {'path': archive.name, 'sha256': sha256(archive), 'bytes': archive.stat().st_size}


def publish_report():
    # Publish just this report against the latest remote tip, preserving changes
    # made elsewhere during a long simulator run. No force pushes or main-checkout rebases.
    workspace = Path(tempfile.mkdtemp(prefix='jepa-rope-publication-')) / 'repo'
    ssh = 'ssh -i /home/zjc/.ssh/id_ed25519_jepa_stall -o IdentitiesOnly=yes'
    def git(*arguments):
        return subprocess.run(['git', '-C', str(workspace), *arguments], check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout
    subprocess.run(['git', 'clone', '--depth', '1', '--filter=blob:none', '--sparse',
                    '--single-branch', '--branch', BRANCH, '-c', f'core.sshCommand={ssh}',
                    'git@github.com:Celveen/JEPA-Stall.git', str(workspace)], check=True)
    git('sparse-checkout', 'set', '--no-cone', '/reports/rope_ecp_20261110/')
    destination = workspace / 'reports/rope_ecp_20261110'
    shutil.copytree(REPORT, destination, dirs_exist_ok=True)
    git('add', '--', 'reports/rope_ecp_20261110')
    git('-c', 'user.name=Codex', '-c', 'user.email=codex@localhost',
        'commit', '-m', 'results: publish frozen Rope native versus ECP memory pilot')
    git('push', 'origin', f'HEAD:refs/heads/{BRANCH}')
    return {'status': 'published', 'commit': git('rev-parse', 'HEAD').strip(),
            'branch': BRANCH, 'workspace': str(workspace)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if not args.run:
        raise SystemExit('Use --run to start the authorized Rope-only, frozen two-arm pilot.')
    REPORT.mkdir(parents=True, exist_ok=True)
    lock = (REPORT / 'pipeline.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'status': 'running', 'stage': 'starting', 'pid': os.getpid(), 'gpu': args.gpu,
             'started_utc': timestamp(), 'planned_primary_episodes': 100,
             'planned_paired_tasks': 50, 'integration_episodes': 2,
             'paired_tasks_completed': 0, 'training': False, 'online_updates': False}
    write_json(REPORT / 'pipeline_status.json', state)
    try:
        environment = create_environment(args.gpu)
        sources = runtime_sources()
        existing = REPORT / 'frozen_runtime_sources.json'
        if existing.exists() and json.loads(existing.read_text()) != sources:
            raise RuntimeError('Runtime differs from the previously recorded pilot.')
        write_json(existing, sources)
        validation = json.loads((REPORT / 'numerical_validation.json').read_text())
        assert validation['passed']
        native = run_job('native', 0, True, environment, state, sources)
        ecp = run_job('S', 0, True, environment, state, sources)
        integration = analyze_pair(native, ecp, 0)
        assert integration[0]['first_intervention_plan_zero_based'] == 1
        write_json(REPORT / 'integration_validation.json', {'passed': True,
                   'forced_trigger_excluded_from_primary_results': True, 'rows': integration})
        for chunk in range(5):
            run_job('native', chunk, False, environment, state, sources)
            run_job('S', chunk, False, environment, state, sources)
            summary = summarize(OUTPUT, REPORT)
            state['paired_tasks_completed'] = summary['paired_episodes']
            write_json(REPORT / 'pipeline_status.json', state)
        assert summary['status'] == 'complete'
        archive = package_results()
        state.update({'status': 'complete', 'stage': 'completed_and_packaged',
                      'ended_utc': timestamp(), 'archive': archive, 'summary': summary})
        write_json(REPORT / 'pipeline_status.json', state)
        try:
            publication = publish_report()
        except Exception as error:
            publication = {'status': 'failed', 'error': repr(error),
                           'all_local_results_preserved': True}
        write_json(REPORT / 'publication_status.json', publication)
    except Exception as error:
        state.update({'status': 'failed', 'error': repr(error), 'updated_utc': timestamp()})
        write_json(REPORT / 'pipeline_status.json', state)
        raise


if __name__ == '__main__':
    main()
