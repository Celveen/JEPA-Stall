"""Independent qualitative rerun; append RGB capture after unchanged planning finishes."""
import hashlib
import json
from pathlib import Path
import numpy as np
import worker
from chunk_planner import ObservedChunkPlanner

original = ObservedChunkPlanner.plan


def capture(self, *args, **kwargs):
    result = original(self, *args, **kwargs)
    # No rendering, RNG calls or additional simulator steps. All arrays already exist.
    physical = self.preprocessor.denormalize_actions(result[0].detach().cpu()).numpy()
    np.savez_compressed('recorded_execution.npz',
                        obs_visual=self.execution_observations['visual'],
                        obs_proprio=self.execution_observations['proprio'],
                        states=self.execution_states, physical_actions=physical)
    Path('capture_provenance.json').write_text(json.dumps(dict(
        role='independent qualitative rerun; excluded from original formal-20 statistics',
        source='actual observations retained during this execution; no physics replay',
        instrumentation='save existing arrays only after original planner returns',
        capture_worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2)+'\n')
    return result


if __name__ == '__main__':
    ObservedChunkPlanner.plan = capture
    worker.main()
