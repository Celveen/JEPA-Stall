# Method prose edits

Apply selectively; preserve equations and all symbol definitions. The existing equations are already logically ordered. These edits make the register more concise and reduce experimental implementation detail in the method without weakening reproducibility.

1. Subsection title: `World models and the intervention interface` -> `Latent planning objective`.

2. First paragraph, final two sentences: replace `Here $\theta$ denotes model parameters. Each block contains five control calls; $H=5$, and MPC executes only the first block before replanning.` with `Here $\theta$ denotes model parameters and $H$ the planning horizon. MPC executes the first action block and replans from the resulting observation.` Move the precise five-calls/five-blocks setting to the experimental setup/appendix. If experimental setup already describes these values, no duplication is needed.

3. After the main objective, use: `Here $\mathcal R_t^{\rm stall}$ is the stagnation regularizer, $\kappa\geq0$ controls its strength, $g_t\in\{0,1\}$ gates correction, and $\alpha_t$ controls the relative emphasis on proprioception. ECP optimizes this objective with either a frozen world model or a model updated by AdaJEPA.` This replaces both the current definition paragraph's last two sentences and its first sentence; all symbols remain defined.

4. Trigger subsection: keep $W$, $\epsilon$, and both thresholds defined, but consolidate the experimental values into the appendix parameter table if space is needed. Retain the re-encoding rule, net-displacement interpretation, and detection warm-up definition. Use `Detection begins after a task-specific warm-up period (\cref{sec:parameters}).` instead of the PushObj=10/Maze=4 sentence. This setting is protocol, not the scientific mechanism.

5. After the regularizer equation, replace `Memories are summed and retained until the episode ends, including after motion resumes.` with `Memory persists throughout the episode, so previously stalled neighborhoods continue to influence subsequent plans.` This preserves mechanism and explains why retention matters.

6. Kernel paragraph: replace `The kernel therefore supplies a local repulsive cost while preserving the goal as its destination.` with `The kernel therefore adds a local repulsive cost without penalizing the goal.` The latter is the precise property established by the preceding support condition. Do not claim that the regularizer guarantees progress or escape.

7. Reweighting subsection, opening: `The regularizer encodes the spatial history of stagnation. Proprioceptive reweighting adjusts the goal component emphasized during recovery.` More formal than repeated italicized where/which framing; intro already uses the memorable interpretation.

8. Scheduling subsection, final three sentences: `Memory accumulates before activation. Once opened, the gate remains active and enables both the regularizer and trigger-dependent reweighting. Model updates continue throughout, allowing prediction learning and planning correction to follow different schedules.` This is clearer about the persistent gate versus transient boost. Ensure actual onset uses next plan, as in existing text.

9. Data subsection: retain the attribution to AdaJEPA and the bundled recipe description. Do not call the entire change simply an action-bias effect: initialization/placement, retries, jitter, and filtering also change. Move the 16,000/4,000 counts into setup/appendix and end with `We retain the latent-prediction and trajectory-straightening objective~\cite{wang2026straightening}, isolating the collection recipe as the change to the training procedure.` This is acceptable only if the implementation truly fixes all other architecture/training hyperparameters; otherwise use the safer `The predictive architecture and learning objective follow the original implementation~\cite{wang2026straightening}; implementation details appear in \cref{sec:implementation}.`

10. `Directional action correction` is auxiliary and not part of the main S+P method. Move its full subsection to the appendix if root removes the corresponding main-text comparison. Keep it in the ablations only if needed to answer a distinct mechanism question. Do not leave it appearing to be a mandatory ECP component while the headline results use S+P only.

# Evidence/wording constraints

- The abstract and intro refer to the released-checkpoint baseline re-evaluated under this paper's common protocol, not numbers copied from the official AdaJEPA paper.
- The headline full-method row is S+P: 253/350 frozen (72.3%) and 269/350 adaptive (76.9%). The 270/350 adaptive result (77.1%) belongs to P-only. Component ablation must retain this distinction.
- Combined overall gains are 100/350 = 28.6 percentage points for frozen and 26/350 = 7.4 points for AdaJEPA, if root chooses to report deltas.
- Do not relocate the fact that evaluation tasks were used for controller selection entirely out of main text. A concise central protocol disclosure is scientifically material. Numeric seed identifiers and overlap bookkeeping belong in the appendix.
- The local potential is a planning-time regularizer. The training interface currently changes collection, not a newly validated stagnation loss in training.
