# Cross-model PushObj comparison audit

Checked 2026-11-06. Read-only research; no training or evaluation launched.

## Decision for the proposed first-page chart

The six existing Frozen/AdaJEPA bars are supported by paired local evaluation: 43.7/61.1/72.3% and 69.4/74.9/76.9%. They describe two deployment regimes of the project's model family. They do not demonstrate transfer to DINO-WM, LeWM, or DDP-WM architectures. A separate cross-method panel currently has no verified, protocol-matched values for those three methods. Do not insert published PushT scores or unmatched PushObj scores as if they share the local 350-task, seven-shape, 30-replan evaluation.

## Verified published results and protocol

| Method | Exact verified published number relevant to the request | Protocol identity | Comparable with local 76.9%? |
|---|---|---|---|
| DINO-WM | PushObj 34% (Table 3) | Four training shapes; evaluation on two unseen shapes. Appendix Table 11 lists 20,000 training trajectories and frameskip 5. This is an unseen-shape aggregate, not the local seven-shape aggregate. | No. A matched reevaluation is needed. |
| LeWM | Original paper contains no PushObj success result. | Original control tasks: PushT, TwoRoom, OGBench-Cube, Reacher. Appendix D uses CEM (300 samples; PushT 30 iterations), horizon 25 controls and executes all 25 before replanning. Appendix F gives a 50-action budget, with goal 25 steps away. | No. No original-paper PushObj value exists to use. |
| DDP-WM | PushT 98% (Table 1), compared with DINO-WM 90% in that paper | Five evaluated environments: PointMaze, PushT, Wall, Rope, Granular; no PushObj. Official method name: Disentangled Dynamics Prediction for Efficient World Models, Yin et al., ICML 2026. | No. PushT is a different benchmark from the seven-shape PushObj suite. |

Primary sources:

- DINO-WM, Table 3 and Appendix A.2/A.9: https://arxiv.org/html/2411.04983v2 ; published record: https://proceedings.mlr.press/v267/zhou25t.html
- LeWM, §4.1 and Appendices D/F: https://arxiv.org/html/2603.19312v3
- DDP-WM, Table 1 and §4.1: https://arxiv.org/html/2602.01780v2 ; published record: https://proceedings.mlr.press/v306/yin26l.html

The publisher PDF links returned unsupported-content errors in the browsing tool; the paper HTML and publisher landing pages were accessible. The DDP-WM publisher abstract independently confirms the 98% PushT number.

## Later LeWM PushObj result is also a different protocol

TEMPO reports a LeWM PushObj baseline of 53.6 ± 2.2% for unseen shapes at a one-plan goal offset (Table 3), and LeWM+TEMPO 54.8 ± 2.0%. Appendix G trains on T/Z/square/small tee and holds out L/I/+, using 20,000 episodes and ten training epochs. Our train/test shape assignment differs, and our headline number averages all seven shapes. Thus 53.6% is a real LeWM result, but is not a valid bar in the proposed matched local ranking.

Source: https://arxiv.org/html/2610.04988v1 , Table 3 and Appendix G.

D-JEPA separately reports 53.67% mean on I/small-T/square (Table 4), using 100 starts per shape, selection from 63 candidates, and one complete 25-control execution. This is another relevant PushObj result whose execution protocol differs from our up-to-30-replan MPC evaluation. Source: https://arxiv.org/html/2609.24749v1 , Table 4 and Appendix A.1.

## Official available code/checkpoints

- DINO-WM official README lists public pretrained checkpoints for PushT, PointMaze and Wall, with no listed PushObj checkpoint: https://github.com/gaoyuezhou/dino_wm
- LeWM official README lists PushT, Cube, TwoRoom and Reacher checkpoints; no listed PushObj checkpoint: https://github.com/lucas-maes/le-wm . PushT checkpoint: https://huggingface.co/quentinll/lewm-pusht
- DDP-WM official README lists PushT, PointMaze, Wall, Rope and Granular checkpoints, no listed PushObj checkpoint: https://github.com/HCPLab-SYSU/DDP-WM . Release: https://huggingface.co/yinkx3/ddp-wm

This establishes an implementation route, not permission to start runs and not evidence that an unlisted asset is unavailable everywhere. A local text search of reports/docs/provenance found no matching three-method PushObj comparison; existing local DINO-WM results concern Wall.

## Concrete evidence needed for the two proposed claims

1. **Cross-model applicability:** evaluate each backbone with its native planning objective and with our intervention on identical starts/goals and execution budgets. This requires at least three paired baseline/intervention comparisons. A rank chart of native baselines versus our own final model alone does not show that our intervention transfers to those models.
2. **Best performance among compared models:** evaluate native DINO-WM, LeWM, DDP-WM and our final system on one declared PushObj protocol. Match the train/test shape split and training-data access; distinguish matched-training experiments from released-checkpoint transfer. A PushT-only checkpoint versus our four-shape-trained model cannot isolate algorithm quality. Freeze the protocol and selection rule before executing a fresh test set.
3. **Intervention interface:** the memory-based stagnation penalty can operate on each model's latent. Explicit proprioceptive goal weighting requires a corresponding proprioceptive feature stream; LeWM is described as pixel-only. Do not silently add privileged state or a new proprioceptive head and still label it an unchanged LeWM deployment. A useful portability study can first test the shared S component, with S+P reported where the backbone already supports P.

Current supportable manuscript language: “The method improves both frozen and test-time-adaptive deployment, reaching 76.9% on our matched PushObj evaluation.” Cross-architecture deployment and a SOTA claim require the paired studies above; they are objectives for the next experiments, not completed findings.
