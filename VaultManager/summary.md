# Uncertainty-Aware Skin Lesion Classification with Evidential Error-Correcting Output Codes

## Metadata
- **Authors:** Emre Tuygan, Erchan Aptoula, Berrin Yanikoglu
- **Year:** 2026
- **Venue:** IEEE
- **DOI:** Not reported

## TL;DR
This paper proposes EDL-ECOC, a framework combining evidential deep learning (EDL) with error-correcting output codes (ECOC) for skin lesion classification. It decomposes multiclass classification into binary evidential tasks and introduces out-of-dichotomy uncertainty regularization. On the ISIC 2019 dataset, EDL-ECOC achieves a mean balanced accuracy of 0.619 and an OOD detection AUROC of 0.696, outperforming base classifiers in accuracy while maintaining strong OOD detection capabilities.

## Motivation
Reliable skin lesion classification requires accurate predictions together with meaningful uncertainty estimates. Deep classifiers commonly use softmax probabilities which may be high even when incorrect, and while EDL and ECOC address reliability, they have largely been investigated as separate approaches. The objective is to bridge this gap by combining the properties of ECOC with the explicit uncertainty representation of EDL.

## Proposed Methodology
- Decompose the multiclass skin lesion classification problem into multiple binary classification tasks using an ECOC matrix (M) with ternary codes (-1, 0, +1).
- Utilize a shared EfficientNetV2-M backbone to map input images to embeddings.
- Pass embeddings to L independent binary evidential ECOC heads (28 OvO and 8 OvR) that produce raw outputs for positive and negative class partitions.
- Map raw outputs to non-negative evidence using a softplus activation function and convert them into Dirichlet parameters.
- Apply an out-of-dichotomy uncertainty regularization strategy that exploits the zero entries of the ternary codebook to encourage low evidence for samples outside each head’s assigned class partition.
- Decode the final class label and uncertainty using an EDL-ECOC decoder.

## Training Details
- Dataset: ISIC 2019 skin lesion classification dataset
- Backbone: ImageNet pretrained EfficientNetV2-M
- Epochs: 35
- Optimizer: AdamW
- Learning Rate: 1e-4
- Weight Decay: 1e-2
- Batch Size: 32
- Head Dropout: 0.4
- Backbone Drop Path: 0.2
- Regularization Coefficients: lambda_kl=1.0, lambda_dic=0.4 (linearly annealed over first 10 epochs)
- Random Seeds: 3
- Checkpoint Selection: Highest validation macro F1

## Evaluation Metrics
- Balanced Accuracy
- Macro F1
- Out-of-Distribution (OOD) Detection AUROC

## Limitations
- The paper does not explicitly list limitations in a dedicated section, but notes that future work will investigate alternative codebook designs and generalization to other multiclass classification tasks.
