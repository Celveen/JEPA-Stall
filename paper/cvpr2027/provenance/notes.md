# Literature verification for JEPA-Stall

Checked against primary sources on **2026-11-05, Asia/Shanghai**. This file is an editorial evidence note, not manuscript text. `references.bib` contains 20 entries. Metadata were checked with arXiv, PMLR, publisher pages, CVF, and author-maintained project pages. References are not experimental comparisons unless actually run locally.

## Closest work and boundaries

| Citation key | Verified publication status and mechanism | Implication for our framing |
|---|---|---|
| `maes2026lewm` | [LeWorldModel](https://arxiv.org/abs/2603.19312), 2026 preprint. Joint encoder/predictor learning with prediction and isotropic-Gaussian SIGReg objectives. | Cite as a compact latent WM reference and visual style reference. Do not call our TS-based implementation LeWM. |
| `wang2026straightening` | [Temporal Straightening](https://arxiv.org/abs/2603.12231), ICML 2026 camera-ready; PDF says PMLR 306. Curvature regularization improves latent geometry for gradient planning. | Identify TS as the base representation objective; distinguish temporal curvature in training from execution-history repulsion at deployment. |
| `wang2026adajepa` | [AdaJEPA](https://arxiv.org/abs/2606.32026), 2026 preprint. Self-supervised model updates using executed transitions within MPC. **Section A.2 already introduces contact bias in PushObj data.** | Our narrow contribution is deciding how to alter planning behavior after repeated nonprogress, with frozen-model evidence. Contact-biased collection alone is not a first claim. |
| `liu2026djepa` | [D-JEPA](https://arxiv.org/abs/2609.24749), 2026 preprint. Learns decision-relevant relations among candidate futures using executed outcomes; includes visual/proprioceptive rank-fusion controls. | Avoid claiming discovery that prediction or distance rankings need not produce successful execution. Distinguish online episode history and temporary correction from learned outcome alignment. |
| `kang2026anisowm` | [AnisoWM](https://arxiv.org/abs/2609.37441), 2026 preprint. Learns diagonal covariance under trace/anisotropy constraints during WM training, inducing a different Euclidean planning geometry. | Merely reweighting latent components is not the conceptual novelty. Our potential contribution is when and for how long weights change during execution. |
| `mete2026optta` | [OpTTA](https://arxiv.org/abs/2610.35636), 2026 preprint accepted to **CoRL 2026 Workshop on Pretrain to Adapt**, not the main conference. Its model update combines observed prediction loss with imagined planning cost. | Do not claim the first task-aware online adaptation. Frozen-model planner correction is a different intervention site. |

**Recommended claim:** execution history can expose a persistent mismatch between the planner's current preferences and task progress; bounded, temporary changes to action proposals, goal-component weighting, and visitation costs can restore progress without retraining the world model. Match this language to precisely the configurations that yielded improvements. Treat the role of interaction-rich training data as an empirical dimension, unless factorial controls isolate a mechanism.

**Claims to avoid:** universal stalled-versus-unlearnable classification; proving frozen models already contain all required dynamics; universal cross-dataset thresholds; first use of contact bias, action priors, potential-based repulsion, objective reweighting, or prediction/planning mismatch; state-of-the-art against papers that were not evaluated in the same protocol. A negative auxiliary-loss result shows that the tested objectives did not improve the tested configuration; it does not establish impossibility or a causal explanation by itself.

## Publication metadata checks

- [DINO-WM](https://proceedings.mlr.press/v267/zhou25t.html): ICML 2025, Gaoyue Zhou, Hengkai Pan, Yann LeCun, Lerrel Pinto; PMLR 267, pp. 79115–79135.
- [V-JEPA 2 official repository citation](https://github.com/facebookresearch/vjepa2/blob/main/README.md): arXiv 2506.09985 (2025). Use **Mahmoud Assran**, **Mojtaba Komeili** as one author, and **Francois Robert Hogan**, following the official citation. The arXiv abstract page incorrectly splits Komeili's name in its HTML metadata. Full authors are retained in the supplied BibTeX.
- [I-JEPA CVF record](https://openaccess.thecvf.com/content/CVPR2023/html/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.html): **CVPR 2023**, pp. 15619–15629. The arXiv comments currently say ICCV; prefer CVF's proceedings record. CVF direct fetch returned 403, but its indexed original record provides full BibTeX and the primary PDF is readable.
- [Dreamer author project](https://dreamrl.github.io/): ICLR 2020; title *Dream to Control: Learning Behaviors by Latent Imagination*. Do not mix this with DreamerV3's later publication.
- [SPiRL PMLR record](https://proceedings.mlr.press/v155/pertsch21a.html): conference is **CoRL 2020**, proceedings year **2021**. Both are represented correctly in BibTeX.
- [World Models](https://arxiv.org/abs/1803.10122): this entry is the *World Models* preprint. Do not give it the distinct NeurIPS paper's *Recurrent World Models Facilitate Policy Evolution* title/venue.
- [D4RL](https://arxiv.org/abs/2004.07219): bibliography uses the original 2020 arXiv record, whose latest revision is 2021. Use for environment/dataset lineage, not an experimental baseline.

## Suggested related-work prose (adapt to final method details)

```tex
\paragraph{Latent world models for control.}
World Models established a compact learned simulator as a basis for behavior learning~\cite{ha2018worldmodels}.
PlaNet optimizes actions through learned latent dynamics, while Dreamer learns behavior through imagined trajectories~\cite{hafner2019planet,hafner2020dreamer}.
TD-MPC2 combines latent trajectory optimization with reward and value learning~\cite{hansen2024tdmpc2}.
Joint-embedding predictive architectures offer a complementary route based on feature prediction~\cite{assran2023ijepa}.
DINO-WM learns action-conditioned dynamics over pretrained spatial features, and V-JEPA~2 extends predictive representation learning to image-goal robot control~\cite{zhou2025dinowm,assran2025vjepa2}.
LeWorldModel trains a compact encoder and predictor jointly with Gaussian regularization~\cite{maes2026lewm}.
We study the execution of latent planners: repeated replanning can continue to select behaviors that produce little task progress.

\paragraph{Geometry and decision alignment.}
The distinction between fitting a model and supporting decisions predates modern visual world models; value-aware model learning explicitly accounts for downstream value structure~\cite{farahmand2017valueaware}.
Temporal Straightening encourages locally straight latent trajectories, improving the geometry encountered by gradient-based planning~\cite{wang2026straightening}.
AnisoWM learns an anisotropic regularization target during training to change the metric induced by latent distance~\cite{kang2026anisowm}.
D-JEPA learns relations among candidate outcomes to align latent action selection with realized outcomes~\cite{liu2026djepa}.
These methods establish representation geometry and candidate ordering as important design choices.
Our focus is their temporal use within an episode: execution history determines when the planner's current action preferences should be adjusted and when its default objective should resume.

\paragraph{Closed-loop adaptation and action priors.}
Model predictive control repeatedly optimizes, executes, and replans from the observed state~\cite{garcia1989mpc}.
AdaJEPA adds self-supervised model updates using executed transitions, and OpTTA incorporates imagined goal-reaching costs into those updates~\cite{wang2026adajepa,mete2026optta}.
Our planner-side intervention instead modifies action proposals and costs while retaining the world model, making its effect directly measurable with frozen dynamics.
Behavior and skill priors guide exploration and reuse movement structure across tasks~\cite{tirumala2022behaviorpriors,pertsch2021spirl}; artificial potential fields provide a classical mechanism for locally shaping motion~\cite{khatib1986potential}.
We instantiate a temporary interaction prior from recent nonprogress: directional action proposals, adjusted proprioceptive goal weighting, and a penalty on revisiting stalled neighborhoods.
The experimental analysis separates this execution-time correction from contact-biased data collection, which is already part of AdaJEPA's PushObj construction~\cite{wang2026adajepa}.
Inverse-dynamics representation learning also has a long history~\cite{pathak2017curiosity}; our auxiliary-objective ablations test whether such additional training signals improve the same downstream planner.
```

The final paragraph describes the requested design direction; the manuscript must align it with actual code and tables. In particular, do not imply all three correction components were enabled in every reported run, or that repulsion caused the full combined gain.

## Figure references (visually inspected)

- **LeWM Figure 1**, PDF p.1: white background; light blue encoder/predictor blocks; gray-blue arrows; warm red reserved for the new regularizer; small actual environment frames; aligned panels and restrained typography. Local preview: `lewm_page1.png`.
- **TS Figure 1**, PDF p.1: three clean panels (original environment, old embedding, new embedding), consistent trajectory colors; compact axis labels; one visual contrast per panel. Local preview: `ts_page1.png`.
- **AdaJEPA Figure 1**, PDF p.2: left closed-loop mechanism, right latent-planning detail; observation images anchor the diagram; orange identifies updates. Local preview: `adajepa_page2.png`. Prefer its layout clarity without copying decorative icons.

Suggested original figures: (1) matched, real simulator frames showing baseline nonprogress and recovery, labeled with task, step, configuration, and success criterion; (2) a white-background closed-loop overview with blue-gray native MPC and one warm accent for the history-triggered correction. An optional third figure can plot only verified paired results. Use the references for visual grammar, not copied artwork or imagined simulator output.
