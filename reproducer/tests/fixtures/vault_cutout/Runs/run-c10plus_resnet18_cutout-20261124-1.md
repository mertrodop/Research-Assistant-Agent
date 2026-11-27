---
type: run
paper: cutout
claim_id: c10plus_resnet18_cutout
run_id: run-c10plus_resnet18_cutout-20261124-1
status: completed
spec_faithful: false
generated_by: reproducer
---

<!-- reproducer:begin -->
# run-c10plus_resnet18_cutout-20261124-1

Claim: [[Claims/cutout__c10plus_resnet18_cutout|c10plus_resnet18_cutout]] (paper [[Papers/cutout|cutout]])

Best run for this claim.

## Command

```bash
python train.py --dataset cifar10 --model resnet18 --data_augmentation --cutout --length 16
```

## Execution

- Image: `reproducer/cutout:424da3bbafa8`
- Base image: `pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime`
- Status: completed (exit 0)
- Duration: 5804.9 s (1h 36m)
- GPU hours: 1.613
- Spec-faithful: no (assumptions used)

## Metrics

- Converted `test_error` (claim precision): 4.10 %
- Raw `raw:test_acc`: 0.959

## References

- Log: `runs/run-c10plus_resnet18_cutout-20261124-1/run.log`
- Artifact: `runs/run-c10plus_resnet18_cutout-20261124-1/artifacts/logs/cifar10_resnet18.csv`
- run_result: `fa3e0d70fc71`
- run_plan: `9fcb79927e62`
- env plan: `e96c2aafb37d`
- preflight: `94d40bbc0ea9`
- config digest: `aaaaaaaaaaaa`
<!-- reproducer:end -->
