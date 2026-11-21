"""Verify that the guidance loss has correct physical gradients and prefix scope."""

from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source"))
from planning.lewm_contact_objective import LeWMContactObjectiveMPCPlanner
from planning.lewm_proprio import pusher_rollout_map

p = object.__new__(LeWMContactObjectiveMPCPlanner)
p.n_taken_actions, p.executed_controls, p.action_scale = 1, 5, 100.
p._action_mean = torch.tensor([-.0087, .0068], dtype=torch.float64)
p._action_std = torch.tensor([.2019, .2002], dtype=torch.float64)
p._position_std = torch.tensor([100., 100.], dtype=torch.float64)
p._contact_indices = torch.tensor([1])
p._contact_current = torch.tensor([[200., 250., 5., -3.]], dtype=torch.float64)
p._contact_target = torch.tensor([[240., 250.]], dtype=torch.float64)
p._contact_map = torch.tensor(pusher_rollout_map(5, 100, 20, 100, 10)[1:], dtype=torch.float64)

actions = torch.randn(3, 5, 10, generator=torch.Generator().manual_seed(17), dtype=torch.float64,
                      requires_grad=True)
cost = p._contact_cost(actions).sum()
gradient, = torch.autograd.grad(cost, actions)
assert gradient[1, :1].norm() > 0
assert torch.count_nonzero(gradient[[0, 2]]) == 0
assert torch.count_nonzero(gradient[:, 1:]) == 0
assert torch.autograd.gradcheck(p._contact_cost, (actions,), eps=1e-6, atol=1e-5)

# Compare the differentiable map with direct PD integration at every control.
commands = (actions.detach()[1, 0].reshape(5, 2) * p._action_std + p._action_mean).numpy() * 100
position = p._contact_current[0, :2].numpy().copy()
velocity = p._contact_current[0, 2:].numpy().copy()
expected = []
for command in commands:
    target = position + command
    for _ in range(10):
        velocity += (100 * (target - position) - 20 * velocity) * .01
        position += velocity * .01
    expected.append(position.copy())
np.testing.assert_allclose(p._contact_positions(actions)[0].detach().numpy(), expected, atol=1e-10)

# With only this objective, optimized controls approach the target without a kick.
zero = (-p._action_mean / p._action_std).repeat(3, 5, 5).reshape(3, 5, 10)
optimized = zero.clone().requires_grad_(True)
initial = float(p._contact_cost(optimized))
optimizer = torch.optim.Adam([optimized], lr=.1)
for _ in range(100):
    optimizer.zero_grad()
    loss = 10 * p._contact_cost(optimized).sum()
    loss.backward()
    optimizer.step()
final = float(p._contact_cost(optimized))
assert final < .15 * initial, (initial, final)
assert float(p._contact_positions(optimized)[0, -1, 0]) > 230
np.testing.assert_array_equal(optimized.detach()[[0, 2]], zero[[0, 2]])
np.testing.assert_array_equal(optimized.detach()[:, 1:], zero[:, 1:])
print("PASS: finite-difference gradients, selected-task/prefix scope, exact PD positions.")
print(f"PASS: optimization alone reduces contact loss {initial:.6f} -> {final:.6f}.")
