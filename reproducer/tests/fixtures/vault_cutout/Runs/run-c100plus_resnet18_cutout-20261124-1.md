---
type: run
paper: cutout
claim_id: c100plus_resnet18_cutout
run_id: run-c100plus_resnet18_cutout-20261124-1
status: completed
spec_faithful: false
generated_by: reproducer
---

<!-- reproducer:begin -->
# run-c100plus_resnet18_cutout-20261124-1

Claim: [[Claims/cutout__c100plus_resnet18_cutout|c100plus_resnet18_cutout]] (paper [[Papers/cutout|cutout]])

Best run for this claim.

## Command

```bash
python train.py --dataset cifar100 --model resnet18 --data_augmentation --cutout --length 8
```

## Execution

- Image: `reproducer/cutout:424da3bbafa8`
- Base image: `pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime`
- Status: completed (exit 0)
- Duration: 6074.4 s (1h 41m)
- GPU hours: 1.687
- Spec-faithful: no (assumptions used)

## Metrics

- Converted `test_error` (claim precision): 21.60 %
- Raw `raw:test_acc`: 0.784

## References

- Log: `runs/run-c100plus_resnet18_cutout-20261124-1/run.log`
- Artifact: `runs/run-c100plus_resnet18_cutout-20261124-1/artifacts/logs/cifar100_resnet18.csv`
- run_result: `bd38fcf9783e`
- run_plan: `dc2b793d7f37`
- env plan: `e96c2aafb37d`
- preflight: `bc5d42a8009f`
- config digest: `bbbbbbbbbbbb`
<!-- reproducer:end -->
