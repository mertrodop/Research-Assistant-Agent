---
type: claim
paper: cutout
claim_id: c10plus_resnet18_cutout
metric: test_error
status: verified
fidelity: inferred
verdict: within_tolerance
generated_by: reproducer
---

<!-- reproducer:begin -->
# c10plus_resnet18_cutout

Paper: [[Papers/cutout|cutout]]

> ResNet18 + cutout 9.31 ± 0.18 3.99 ± 0.13 34.98 ± 0.29 21.96 ± 0.24 -

- Location: Table 1
- Metric: `test_error` %; dataset CIFAR-10, split test
- Setting: ResNet18 + cutout, standard augmentation (C10+)

## Result

- Paper: 3.99 % ± 0.26
- Ours: 4.10 % (raw test_acc=0.959; run_result `fa3e0d70fc71`)
- |diff|: 0.11 <= 0.26
- Verdict: within tolerance
- Status: verified
- Fidelity: inferred

## Tolerance

- abs=0.26 (bound ± 0.26), pre-registered `4fb891da5f52`
- Rule: 2 x paper std (± 0.13 after 3.99 in the claim quote)

## Runs

1 completed / 1 attempt(s) (0 aborted), 1.613 GPU h

- [[Runs/run-c10plus_resnet18_cutout-20261124-1|run-c10plus_resnet18_cutout-20261124-1]] (best): completed, 4.10 %, exit 0

## Inferred because

- [env_deviation] torch 0.4.0 -> 1.12.1 (no CUDA build for sm_89) [`e96c2aafb37d`]
- [open_unknown] eval_checkpoint not stated by the paper; assumed last (which=last) [`9fcb79927e62`]
- [open_unknown] seed not stated by the paper; assumed 0 (repo default --seed 0) [`9fcb79927e62`]
- [env_assumption] python_version = 3.7 (python of the image) [`e96c2aafb37d`]
- [env_assumption] cuda_version = 11.3 (CUDA build of torch) [`e96c2aafb37d`]
- [protocol] single run vs paper mean of 5 [`eacd02e90d73`]

## Info

- tqdm unpinned (no version stated; pip resolves at build time) [`e96c2aafb37d`]

## Unknowns

- [[Unknowns/cutout__cuda_version|cuda_version]]: open, best run assumed 11.3
- [[Unknowns/cutout__dataset_source|dataset_source]]: resolved: torchvision:CIFAR10
- [[Unknowns/cutout__eval_checkpoint|eval_checkpoint]]: open, best run assumed last
- [[Unknowns/cutout__python_version|python_version]]: open, best run assumed 3.7
- [[Unknowns/cutout__seed|seed]]: open, best run assumed 0
<!-- reproducer:end -->
