"""Preserve the already running native chunk while updating only S branch bookkeeping."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

from run import HERE, REPO, REPORT, OUTPUT, PYTHON, runtime_sources, write_json


def main():
    handoff_path = REPORT / 'native_handoff.json'
    handoff = json.loads(handoff_path.read_text())
    while True:
        state = json.loads((REPORT / 'pipeline_status.json').read_text())
        if state['pid'] != handoff['previous_pipeline_pid']:
            raise RuntimeError('A different pipeline took ownership; preserving all runs.')
        if state['status'] == 'failed':
            if 'Frozen runtime sources changed' not in state.get('error', ''):
                raise RuntimeError('Unexpected worker failure; no automatic restart.')
            if not (OUTPUT / 'native/chunk_00/complete.json').exists():
                raise RuntimeError('Native chunk did not complete; no automatic restart.')
            break
        time.sleep(10)
    # The old queue stops before starting S, after finishing its unchanged native
    # computation. The new runner will skip that complete native chunk.
    while True:
        path = Path(f"/proc/{handoff['previous_pipeline_pid']}/stat")
        if not path.exists() or path.read_text().split()[2] == 'Z':
            break
        time.sleep(1)
    old_s = OUTPUT / 'integration/S'
    archive = OUTPUT / 'integration_before_noop_reuse'
    archive.mkdir(exist_ok=True)
    old_s.rename(archive / 'S')
    validation = REPORT / 'integration_validation.json'
    if validation.exists():
        validation.rename(REPORT / 'integration_validation_before_noop_reuse.json')
    old_log = OUTPUT / 'logs/integration_S_chunk_00.log'
    if old_log.exists():
        old_log.rename(archive / 'integration_S_chunk_00.log')
    write_json(REPORT / 'native_generation_sources.json', handoff['native_generation_sources'])
    write_json(REPORT / 'frozen_runtime_sources.json', runtime_sources())
    handoff.update({'status': 'resumed', 'resumed_utc': datetime.now(timezone.utc).isoformat(),
                    'completed_native_chunk_preserved': True,
                    'native_sampling_model_targets_and_objective_unchanged': True})
    with (REPORT / 'launcher.log').open('a') as stream:
        process = subprocess.Popen([PYTHON, '-u', str(HERE / 'run.py'), '--gpu', '0', '--run'],
                                   cwd=REPO, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    handoff['new_pipeline_pid'] = process.pid
    write_json(handoff_path, handoff)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        write_json(REPORT / 'native_handoff_error.json', {'error': repr(error),
                   'local_results_preserved': True})
        raise
