---
type: claim
paper: cutout
claim_id: c100plus_resnet18_cutout
metric: test_error
status: verified
fidelity: inferred
verdict: within_tolerance
plan_failed: "metric line 'test_acc: %.3f' not found in train.py"
generated_by: reproducer
---

<!-- reproducer:begin -->
# c100plus_resnet18_cutout

Paper: [[Papers/cutout|cutout]]

> ResNet18 + cutout 9.31 ± 0.18 3.99 ± 0.13 34.98 ± 0.29 21.96 ± 0.24 -

- Location: Table 1
- Metric: `test_error` %; dataset CIFAR-100, split test
- Setting: ResNet18 + cutout, standard augmentation (C100+)

## Result

- Paper: 21.96 % ± 0.48
- Ours: 21.60 % (raw test_acc=0.784; run_result `bd38fcf9783e`)
- |diff|: 0.36 <= 0.48
- Verdict: within tolerance
- Status: verified
- Plan: plan_failed: metric line 'test_acc: %.3f' not found in train.py (no current run plan)
- Fidelity: inferred

## Tolerance

- abs=0.48 (bound ± 0.48), pre-registered `529fb175f61f`
- Rule: 2 x paper std (± 0.24 after 21.96 in the claim quote)

## Runs

1 completed / 1 attempt(s) (0 aborted), 1.687 GPU h

- [[Runs/run-c100plus_resnet18_cutout-20261124-1|run-c100plus_resnet18_cutout-20261124-1]] (best): completed, 21.60 %, exit 0

## Inferred because

- [env_deviation] torch 0.4.0 -> 1.12.1 (no CUDA build for sm_89) [`e96c2aafb37d`]
- [open_unknown] eval_checkpoint not stated by the paper; assumed last (which=last) [`dc2b793d7f37`]
- [open_unknown] seed not stated by the paper; assumed 0 (repo default --seed 0) [`dc2b793d7f37`]
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
