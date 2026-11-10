Final integration: the delivered abstract is four sentences and 89 words; the initial independent proposal below was shortened. See REVISION_V7.md for the final edit summary.

# Proposed v7 opening

Files: `abstract.tex` (4 sentences, approximately 98 words) and `introduction.tex` (approximately 555 words). These are independent proposals; the shared manuscript has not been edited.

## Story

**Execution can remain repetitive even when an agent keeps predicting and replanning. The repeated trajectory is useful evidence for changing action preferences.** This is the publishable organizing idea. ECP converts observed stagnation into (i) local trajectory regularization and (ii) temporary emphasis on goal components. Contact-biased augmentation is presented as the complementary training-side improvement, with its two checkpoint settings explicitly measured.

The introduction now has a causal argument rather than an inventory of interfaces:

1. Promise of latent planning → specific execution problem.
2. Concrete pusher / corridor observations → what information execution history supplies.
3. ECP → distinct jobs of memory and reweighting.
4. Why the regularizer is local and goal-directed → correct TS / AdaJEPA distinction.
5. Complementary training-data intervention, preserving attribution to AdaJEPA.
6. Quantified improvements → mechanism evidence → contributions.

## Primary-paper structure consulted

- [Temporal Straightening](https://arxiv.org/html/2603.12231v3): the opening narrows quickly from latent planning to one actionable geometric problem, then explains the regularizer's purpose before giving results. We borrow the tight problem–mechanism–evidence progression; no claims about straighter representations or improved optimization conditioning are imported.
- [LeWorldModel](https://arxiv.org/html/2603.19312v3): the abstract names a specific bottleneck, states a compact design, then reports concrete practical consequences. This motivates removing broad interface language and giving the ECP mechanism before the numeric gains.
- [AdaJEPA](https://arxiv.org/html/2606.32026v1): the introduction links an observed deployment problem to a precise modification of the MPC loop. Our corresponding modification is execution-conditioned action optimization; online parameter adaptation remains attributed to AdaJEPA.

No phrasing copied. The anti-defensive-writing skill was read and applied to the organization; unsupported priority, generalization, or SOTA claims are not added.

## Accuracy notes for integration

- The headline 72.3 / 76.9 values remain the combined S+P variant, not the 77.1 P-only maximum.
- The four-sentence abstract gives the combined approach's baseline-to-final gains; the introduction and main table retain the intermediate 61.1 / 74.9 values.
- 'Data augmentation' denotes the full contact-biased collection recipe. The main method / experimental protocol must continue to explain source placement, perturbations, retries, and filtering; the causal effect of the directional bias alone has not been isolated.
- 'Without modifying the predictive architecture' is justified by the existing pipeline; it does not claim no runtime cost, no additional data collection, or no model training.
- 'Both frozen and adaptive' describes the evaluated parameter-update settings, not validated transfer across DINO-WM, LeWM, and DDP-WM.
- The Maze figure of 66% belongs to gated S-only regularization. It is deliberately named that way in the abstract.
- The support excludes the goal away from the numerical floor; retain the exact qualifier in the method / mathematical appendix.
- 'Four concise sentences' is a reasonable implementation of the user's request for a brief abstract. Literal four printed CVPR column lines would allow only around 40 words and omit most requested content. At 98 words, this version is substantially shorter than v6.
