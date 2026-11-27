---
type: paper
paper: cutout
repo: uoguelph-mlrg/Cutout
commit: "287f934ea5fa00d4345c2cccecf3552e2b1c33e3"
claims_total: 2
claims_verified: 2
gpu_hours: 3.2998
generated_by: reproducer
---

<!-- reproducer:begin -->
# cutout

- Repository: [uoguelph-mlrg/Cutout](https://github.com/uoguelph-mlrg/Cutout/tree/287f934ea5fa00d4345c2cccecf3552e2b1c33e3)
- Commit: `287f934ea5fa00d4345c2cccecf3552e2b1c33e3`
- Paper source: `paper:cutout`

## Environment

- Image: `reproducer/cutout:424da3bbafa8`
- Base image: `pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime`
- Versions: python 3.7, torch 1.12.1, torchvision 0.13.1, CUDA 11.3
- Env plan: `e96c2aafb37d`
- Deviations:
  - torch 0.4.0 -> 1.12.1 (no CUDA build for sm_89)
- Assumptions:
  - python_version = 3.7 (python of the image)
  - cuda_version = 11.3 (CUDA build of torch)

## Claims

| Claim | Paper ± tol | Ours | \|diff\| | Verdict | Status | Fidelity |
| --- | --- | --- | --- | --- | --- | --- |
| [[Claims/cutout__c10plus_resnet18_cutout\|c10plus_resnet18_cutout]] | 3.99 % ± 0.26 | 4.10 % | 0.11 | within tolerance | verified | inferred |
| [[Claims/cutout__c100plus_resnet18_cutout\|c100plus_resnet18_cutout]] | 21.96 % ± 0.48 | 21.60 % | 0.36 | within tolerance | verified | inferred |

## Unknowns

| Unknown | Origin | Status | Value |
| --- | --- | --- | --- |
| [[Unknowns/cutout__cuda_version\|cuda_version]] | repo | open | assumed 11.3 |
| [[Unknowns/cutout__dataset_source\|dataset_source]] | repo | resolved | torchvision:CIFAR10 |
| [[Unknowns/cutout__eval_checkpoint\|eval_checkpoint]] | paper | open | assumed last |
| [[Unknowns/cutout__python_version\|python_version]] | repo | open | assumed 3.7 |
| [[Unknowns/cutout__seed\|seed]] | paper | open | assumed 0 |

## Runs

- [[Runs/run-c10plus_resnet18_cutout-20261124-1|run-c10plus_resnet18_cutout-20261124-1]]: [[Claims/cutout__c10plus_resnet18_cutout|c10plus_resnet18_cutout]], completed, 4.10 %
- [[Runs/run-c100plus_resnet18_cutout-20261124-1|run-c100plus_resnet18_cutout-20261124-1]]: [[Claims/cutout__c100plus_resnet18_cutout|c100plus_resnet18_cutout]], completed, 21.60 %

## Totals

- Claims: 2 (verified 2)
- GPU hours: 3.300 (2 run_result(s))
- Tokens: api=1000 local=450 calls=2
<!-- reproducer:end -->
