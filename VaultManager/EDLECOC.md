
# Uncertainty-Aware Skin Lesion Classification with Evidential Error-Correcting Output Codes

**Emre Tuygan, Erchan Aptoula and Berrin Yanikoglu**  
Sabanci University, Istanbul, 34956 Türkiye  
{emre.tuygan,erchan.aptoula,berrin}@sabanciuniv.edu

## Abstract

Reliable skin lesion classification requires accurate predictions together with meaningful uncertainty estimates. This work proposes EDL-ECOC, a framework that combines evidential deep learning (EDL) with error-correcting output codes (ECOC) for multiclass classification. The framework decomposes the multiclass problem into multiple evidential binary tasks, enabling uncertainty to be estimated at the level of individual ECOC dichotomies. An out-of-dichotomy uncertainty regularization strategy is introduced that exploits the zero entries of the ternary codebook as an additional regularization signal, encouraging low evidence for samples outside each head’s assigned class partition. On the ISIC 2019 dataset, across three random seeds, EDL-ECOC achieves a mean balanced accuracy of 0.619 and a mean out-of-distribution (OOD) detection AUROC of 0.696. The corresponding base classifier results are 0.582 and 0.668, respectively. These results support the utility of combining ECOC decomposition with evidential uncertainty estimation and of exploiting out-of-dichotomy samples for uncertainty-aware regularization.

**Keywords—** evidential deep learning, error-correcting output codes, skin lesion classification

## I. Introduction

Deep learning has become a dominant approach for automated skin lesion classification, but diagnostically distinct lesions can appear visually similar while lesions within the same class vary in appearance. Clinically useful systems must therefore provide not only accurate predictions but also reliable indications of when a prediction should not be trusted.

Deep classifiers commonly represent prediction confidence using the maximum softmax probability. However, softmax probabilities are normalized point estimates and may be high even when a prediction is incorrect [1]. This limitation has motivated the development of Evidential Deep Learning (EDL) as an uncertainty-aware learning method [2]. Beyond uncertainty estimation, robustness to errors made by individual classifiers is a further important property, motivating error-correcting output codes (ECOC) as a complementary line of work [3].

Although EDL and ECOC have both been applied to different classification tasks, they have largely been investigated as separate approaches addressing different aspects of model reliability [4]–[8]. The objective of this work is to bridge this gap by combining the properties of ECOC with the explicit uncertainty representation of EDL. The proposed formulation preserves the redundant codeword structure while providing a separate uncertainty estimate for each binary task. In this way, the framework produces localized uncertainty information associated with individual ECOC dichotomies while retaining the benefits of ECOC-based decoding.

The main contributions of this work are summarized as follows:

- An evidential ECOC architecture and training paradigm are introduced by combining the ECOC codebook structure with the explicit uncertainty representation provided by EDL.
- An uncertainty-oriented regularization strategy is proposed for the out-of-dichotomy samples of each ECOC head, encouraging the corresponding heads to produce low evidence and high uncertainty when an input falls outside their assigned dichotomy.

Code: https://github.com/EmreTuygan/EDLECOC

## II. Related Work

### A. Error-Correcting Output Codes

ECOC is an ensemble-based framework for multiclass classification that decomposes a $K$-class problem into a collection of binary classification tasks, a strategy originally motivated by error-correcting principles from coding theory [3]. Each binary classifier is assigned according to a predefined codebook, where every row corresponds to a class codeword and every column defines the positive and negative class partitions for a particular binary classifier. The predictions of all binary classifiers are then combined to infer the final class label [3]. Distributing the decision across multiple classifiers in this way can compensate for individual classifier errors and improve robustness [8].

Common codebook designs include One-vs-Rest (OvR), where each classifier distinguishes one class from all others, and One-vs-One (OvO), where a separate classifier is trained for every class pair, as shown in Table I [9], [10]. At inference, the binary outputs are compared with each class codeword, and the predicted label is assigned to the class whose codeword is closest under a chosen decoding metric, such as Hamming or Euclidean distance [11]. Beyond nearest-distance decoding, weighted and class-balanced decoding schemes have also been proposed to improve performance [11].


### B. Evidential Deep Learning

Unlike classifiers that use a softmax layer to produce a single point estimate of the class probabilities, EDL models the prediction as a Dirichlet distribution over class probabilities [2]. Instead of estimating class probabilities, the network predicts $\mathbf{e}\in\mathbb{R}_{\geq 0}^{K}$ for a $K$-class classification problem, where each element represents the amount of evidence supporting a particular class. The corresponding Dirichlet parameters $\boldsymbol{\alpha}=(\alpha_1,\ldots,\alpha_K)$ and Dirichlet strength $S$ are obtained by adding one unit of base evidence to each class [2],

$$
\alpha_k=e_k+1,\qquad k\in\{1,\ldots,K\},\qquad S=\sum_{k=1}^{K}\alpha_k.\tag{1}
$$

The expected class probabilities $\hat{\mathbf{p}}$ and predictive uncertainty $u$ are then computed as [2]:

$$
\hat{\mathbf{p}}=\frac{\boldsymbol{\alpha}}{S},\qquad u=\frac{K}{S}.\tag{2}
$$

Since the predictive distribution is explicitly modeled, the uncertainty is inversely proportional to the total accumulated evidence. Consequently, higher accumulated evidence corresponds to lower uncertainty, whereas low accumulated evidence corresponds to higher uncertainty [2].

## III. Proposed Method

### A. Problem Formulation

Skin lesion classification is considered with $C$ diagnostic classes. Each lesion is represented by an image $\mathbf{x}_i\in\mathbb{R}^{H\times W\times 3}$ with label $y_i\in\{1,\ldots,C\}$. Instead of training a $C$-way softmax classifier, the multiclass task is decomposed into a set of binary classification tasks using an ECOC matrix

$$
M\in\{-1,0,+1\}^{C\times L},\tag{3}
$$

where $L$ is the number of ECOC heads. For class $c\in\{1,\ldots,C\}$ and head $h\in\{1,\ldots,L\}$, $M_{c,h}=+1$ and $M_{c,h}=-1$ respectively indicate that class $c$ has been labeled as positive or negative for the $h$-th base classifier, while $M_{c,h}=0$ denotes a “don’t-care” label for that class [11]. The positive and negative class partitions of each head’s binary task are denoted by

$$
C_h^{+}=\{c:M_{c,h}=+1\},\qquad C_h^{-}=\{c:M_{c,h}=-1\}.\tag{4}
$$

### B. Proposed Architecture

The overall architecture of the proposed EDL-ECOC framework is illustrated in Fig. 1. A shared backbone maps each input $\mathbf{x}_i$ to an embedding $\mathbf{z}_i\in\mathbb{R}^{d}$, which is passed to $L$ independent binary evidential ECOC heads constructed according to the OvO and OvR dichotomies. Head $h$ discriminates $C_h^{+}$ from $C_h^{-}$, producing raw outputs $r_{i,h,-}$ and $r_{i,h,+}$ associated with the negative and positive class partitions of the $h$-th dichotomy, respectively.

**Fig. 1. Overview of the proposed EDL-ECOC architecture.**

The figure illustrates the following processing pipeline:

1. Input RGB image
2. Shared EfficientNetV2 backbone
3. Multiple independent binary evidential heads:
   - Head $h_1$: $(\hat{p}_1,u_1)$
   - Head $h_2$: $(\hat{p}_2,u_2)$
   - ...
   - Head $h_L$: $(\hat{p}_L,u_L)$
4. EDL-ECOC decoder
5. Final predicted class $\hat{y}_i$ and uncertainty $u_i$

Following EDL [2], the outputs are mapped to non-negative evidence via a softplus activation function, and converted into Dirichlet parameters:

$$
e_{i,h,k}=\operatorname{softplus}(r_{i,h,k}),\qquad \alpha_{i,h,k}=1+e_{i,h,k},\qquad k\in\{-,+\}.\tag{5}
$$

## IV. Experiments and Results

### A. Dataset and Training Protocol

The proposed approach is evaluated on the ISIC 2019 skin lesion classification dataset using the official train-test split, whose test set includes 2,047 held-out images from the UNK class, absent from the eight training classes and reserved for OOD evaluation [13]–[15].

All experiments use the same data preprocessing pipeline, data augmentation policy, and ImageNet pretrained EfficientNetV2-M backbone [16], [17]. All models are trained for 35 epochs using AdamW (learning rate = $10^{-4}$, weight decay = $10^{-2}$) with a batch size of 32, head dropout of 0.4, and backbone drop path of 0.2. The regularization coefficients are set to $\lambda_{\mathrm{kl}}=1.0$ and $\lambda_{\mathrm{dic}}=0.4$, both linearly annealed over the first 10 epochs. Experiments are repeated across three random seeds, and results are reported as mean ± standard deviation. For each run, the checkpoint with the highest validation macro F1 is selected for evaluation. Model configurations and regularization coefficients are selected based on validation macro F1, while all reported results are computed on the official held-out test set.

### B. Compared Models and Ablations

To isolate the contributions of EDL, ECOC decomposition, and the proposed out-of-dichotomy regularization, five configurations using the same backbone architecture are compared:

- **Base Classifier.** The classifier is configured for eight-class classification and trained using focal loss [18].
- **EDL.** The conventional classification head is replaced by an 8-way evidential classification head, providing an EDL baseline without ECOC decomposition.
- **ECOC.** The multiclass problem is decomposed using the same 36-head ECOC codebook as the proposed approach, but without evidential modeling.
- **EDL-ECOC w/o Reg.** The 36 ECOC heads are modeled evidentially, but out-of-dichotomy regularization is disabled by setting $\lambda_{\mathrm{dic}}=0$.
- **EDL-ECOC.** The complete proposed model combines the evidential ECOC formulation with out-of-dichotomy uncertainty regularization.

**Computational overhead.** All ECOC variants use 36 binary heads (28 OvO and 8 OvR) operating on the same shared backbone embedding. The backbone is evaluated only once per image, so the additional computation is limited to 36 linear projections corresponding to approximately 0.07% of the backbone parameter count.

### C. Classification

Balanced accuracy and macro F1 are reported as the primary classification metrics to account for class imbalance. Results averaged over three random seeds are summarized in Table II.

**Table II. Classification performance on the ISIC 2019 test set (mean ± standard deviation over three seeds).**

| Model | Balanced Acc. ↑ | F1 ↑ |
|:--|--:|--:|
| Base Classifier | 0.582 ± 0.013 | 0.614 ± 0.011 |
| ECOC | 0.585 ± 0.021 | 0.615 ± 0.018 |
| EDL | 0.593 ± 0.019 | 0.608 ± 0.020 |
| EDL-ECOC w/o Reg. | 0.604 ± 0.019 | 0.627 ± 0.017 |
| EDL-ECOC | 0.619 ± 0.013 | 0.632 ± 0.008 |

The Base and ECOC classifiers exhibit comparable classification performance, which is consistent with the motivation of ECOC: decomposing the multiclass decision into redundant binary tasks to improve robustness rather than directly increasing raw classification performance. The EDL baseline achieves a higher balanced accuracy than the Base Classifier, although this improvement does not translate to macro F1. This pattern is compatible with previously reported sensitivity of EDL models to class imbalance [19]. Taken together, the ablation results suggest that the observed improvement does not arise from ECOC decomposition or EDL in isolation, but from their combination. Importantly, the proposed out-of-dichotomy regularization provides a further improvement over the unregularized EDL-ECOC model, supporting the contribution of the proposed regularization strategy to classification performance.

### D. Uncertainty-Based Out-of-Distribution Detection

OOD detection is evaluated using area under the receiver operating characteristic curve (AUROC), with the held-out UNK class treated as positive and the eight training classes as negative. EDL-based models use evidential uncertainty as the detection score. For the non-evidential baselines, uncertainty is derived from softmax confidence: the Base Classifier uses one minus the maximum softmax probability (MSP) [20], while the ECOC baseline aggregates the corresponding head-level $1-\mathrm{MSP}$ uncertainty scores using contribution weighting.

**Table III. OOD detection AUROC on the ISIC 2019 test set (mean ± standard deviation over three seeds).**

| Model | OOD AUROC ↑ |
|:--|--:|
| Base Classifier | 0.668 ± 0.019 |
| ECOC | 0.653 ± 0.008 |
| EDL | 0.697 ± 0.022 |
| EDL-ECOC w/o Reg. | 0.689 ± 0.012 |
| EDL-ECOC | 0.696 ± 0.022 |

The EDL baseline achieves the highest mean OOD AUROC, highlighting the effectiveness of evidential uncertainty for separating held-out UNK samples from the training classes. The proposed EDL-ECOC model nearly matches this performance, achieving a mean OOD AUROC of 0.696 compared with 0.697 for EDL. Both evidential models achieve higher OOD AUROC than the non-evidential Base Classifier and ECOC baselines. EDL-ECOC also achieves a higher mean OOD AUROC than its unregularized variant and higher classification performance than EDL, indicating that the proposed formulation largely preserves the OOD detection capability of standalone EDL while improving discriminative performance.


## V. Conclusion

This work presented EDL-ECOC, a framework that integrates evidential deep learning with error-correcting output codes for uncertainty-aware multiclass classification. Rather than discarding the zero entries of the ECOC codebook during binary-head training, the proposed formulation exploits the corresponding out-of-dichotomy samples as a task-relative OOD signal to regularize evidential uncertainty. Experiments on ISIC 2019 across three random seeds show the benefits of this framework. EDL-ECOC achieves the highest balanced accuracy and macro F1 among the compared models, while its OOD detection performance nearly matches that of standalone EDL. The ablation results show that combining EDL with ECOC improves classification performance over either component alone, while out-of-dichotomy regularization provides a further improvement. Under a fixed 10% rejection budget, EDL-ECOC also achieves the highest post-rejection classification performance and captures the largest fraction of held-out UNK samples, demonstrating the utility of the learned uncertainty for selective prediction. Overall, the results show that EDL and ECOC decomposition can be effectively combined to improve multiclass classification while maintaining informative uncertainty estimates. Future work will investigate alternative codebook designs and generalization to other multiclass classification tasks.

## References

[1] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration of modern neural networks,” in *Proceedings of the 34th International Conference on Machine Learning*, 2017, pp. 1321–1330.

[2] M. Sensoy, L. Kaplan, and M. Kandemir, “Evidential deep learning to quantify classification uncertainty,” in *Advances in Neural Information Processing Systems*, Dec. 2018, pp. 3179–3189.

[3] T. G. Dietterich and G. Bakiri, “Solving multiclass learning problems via error-correcting output codes,” *Journal of Artificial Intelligence Research*, vol. 2, pp. 263–286, 1995.

[4] M. Chen, J. Gao, and C. Xu, “R-EDL: Relaxing nonessential settings of evidential deep learning,” in *International Conference on Learning Representations*, 2024.

[5] P. Wang, S. Zhang, R. Chen, J. Chen, Y. Fan, L. Ning, and Y. Cao, “Regularized evidential neural networks for deep active learning,” *Pattern Recognition*, vol. 173, p. 112836, 2026.

[6] W. Fu, Y. Chen, Y. Liu, and X. Yue, “D-EDL: Differential evidential deep learning for robust medical out-of-distribution detection,” *Medical Image Analysis*, vol. 109, p. 103888, 2026.

[7] S. A. A. Ahmed, C. Zor, M. Awais, B. Yanikoglu, and J. Kittler, “Deep convolutional neural network ensembles using ECOC,” *IEEE Access*, vol. 9, pp. 86 083–86 095, 2021.

[8] Y. Song, Q. Kang, and W. P. Tay, “Error-correcting output codes with ensemble diversity for robust learning in neural networks,” in *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 35, no. 11, 2021, pp. 9722–9729.

[9] N. J. Nilsson, *Learning Machines: Foundations of Trainable Pattern-Classifying Systems*. McGraw-Hill, 1965.

[10] T. Hastie and R. Tibshirani, “Classification by pairwise coupling,” in *Advances in Neural Information Processing Systems*, vol. 10, 1997, pp. 507–513.

[11] S. Escalera, O. Pujol, and P. Radeva, “On the decoding process in ternary error-correcting output codes,” *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 32, no. 1, pp. 120–134, Jan. 2010.

[12] M. Sensoy, L. Kaplan, F. Cerutti, and M. Saleki, “Uncertainty-aware deep classifiers using generative models,” in *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 34, no. 4, 2020, pp. 5620–5627.

[13] P. Tschandl, C. Rosendahl, and H. Kittler, “The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions,” *Scientific Data*, vol. 5, p. 180161, 2018.

[14] N. C. F. Codella, D. Gutman, M. E. Celebi, B. Helba, M. A. Marchetti, S. W. Dusza, A. Kalloo, K. Liopyris, N. Mishra, H. Kittler, and A. Halpern, “Skin lesion analysis toward melanoma detection: A challenge at the 2017 international symposium on biomedical imaging, hosted by the international skin imaging collaboration,” *arXiv preprint* arXiv:1710.05006, 2017.

[15] C. Hernández-Pérez, M. Combalia, S. Podlipnik, N. C. F. Codella, V. Rotemberg, A. C. Halpern, O. Reiter, C. Carrera, A. Barreiro, B. Helba, S. Puig, V. Vilaplana, and J. Malvehy, “BCN20000: Dermoscopic lesions in the wild,” *Scientific Data*, vol. 11, no. 1, p. 641, 2024.

[16] M. Tan and Q. V. Le, “EfficientNetV2: Smaller models and faster training,” in *Proceedings of the 38th International Conference on Machine Learning*, 2021, pp. 10 096–10 106.

[17] R. Wightman, “PyTorch image models,” GitHub repository, 2019.

[18] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, “Focal loss for dense object detection,” in *Proceedings of the IEEE International Conference on Computer Vision*, 2017, pp. 2980–2988.

[19] T. Xia, T. Dang, J. Han, L. Qendro, and C. Mascolo, “Uncertainty-aware health diagnostics via class-balanced evidential deep learning,” *IEEE Journal of Biomedical and Health Informatics*, vol. 28, no. 11, pp. 6417–6428, 2024.

[20] D. Hendrycks and K. Gimpel, “A baseline for detecting misclassified and out-of-distribution examples in neural networks,” in *International Conference on Learning Representations*, 2017.

---

*This study was supported by the Scientific and Technological Research Council of Türkiye (TUBITAK) under Grant No. 124N417. The authors thank TUBITAK for its support.*

*979-8-3195-2681-6/26/$31.00 ©2026 IEEE*
