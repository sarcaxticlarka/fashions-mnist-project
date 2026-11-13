# Phase 2 Technical Report: Deep Convolutional Neural Networks on Fashion-MNIST
**Project 11 · Automatic Clothing Category Recognition**  
*Department of CS/AI, Newton School of Technology*  
*Course: Deep Learning*  
*Authors: MD. Sajjan, Sibtain Ahmed Qureshi, Syed Darain Qamar, Antik Mandal, Abuzar Haideri*  

---

## 1. Executive Summary & Context

This report presents the complete empirical, theoretical, and architectural findings of **Phase 2 (Deep Learning)** for the Fashion-MNIST category recognition system. 

In **Phase 1 (Classical Machine Learning)**, we established that a fixed spatial descriptor—the Histogram of Oriented Gradients (HOG)—significantly improved linear models (+4.0%), $k$-NN (+2.1%), and Support Vector Machines (+1.4%). However, classical machine learning hit an empirical ceiling at **91.30% test accuracy** (RBF-SVM), and suffered from two critical shortcomings:
1. **The Upper-Body Garment Bottleneck**: 60–65% of all classification errors were concentrated in the ambiguous four-class upper-body cluster `{T-shirt/top, Pullover, Coat, Shirt}`. Shirt recall remained stuck at **0.750**.
2. **Computational Inefficiency**: The best classical model (SVM-RBF) required storing **18.1 million numbers** (support vectors and dual coefficients) and suffered from a slow single-sample inference latency of **5.36 ms/image**.

In **Phase 2**, we replaced handcrafted gradient features with end-to-end learnable convolutional representations. Using a modernized 3-block architecture with **Global Average Pooling (`CNN3Block_GAP`)**, we achieved:
- **93.09% Test Accuracy** and **0.9306 Macro-F1** (an absolute gain of $+1.79\%$ over the best classical model).
- **Statistically Definitive Superiority**: A continuity-corrected McNemar paired test yielded $\chi^2 = 44.56, p = 2.46 \times 10^{-11} \ll 0.001$.
- **Error Reduction**: Total test errors plummeted from 870 down to **691** (179 fewer errors), with Pullover recall climbing from 85.6% to **90.6%** and Coat recall rising from 87.5% to **92.4%**.
- **Efficiency Triumph**: Parameter storage was reduced from 18.1M down to **304,682 weights** (a **$60\times$ compression**), while CPU inference speed increased to **0.68 ms/image** (an **$8\times$ speedup**).

---

## 2. Phase 1 vs. Phase 2: What Changed?

| Dimension | Phase 1 (Classical ML) | Phase 2 (Deep Learning) | Change / Impact |
| :--- | :--- | :--- | :--- |
| **Input Representation** | Handcrafted HOG (864-d) & Shape Stats (16-d) | Raw 2D Images ($1 \times 28 \times 28$) fed to Conv Layers | Replaced rigid manual features with end-to-end learned filters |
| **Top Model** | SVM with Radial Basis Function (RBF) Kernel | `CNN3Block_GAP` (3-Block Deep ConvNet + GAP) | Learnable feature hierarchy adapts to garment semantics |
| **Top Test Accuracy** | 91.30% (SVM-RBF on HOG) | **93.09%** (`CNN3Block_GAP`) | **$+1.79\%$ absolute gain** |
| **Top Test Macro-F1** | 0.9130 | **0.9306** | **$+0.0176$ gain** |
| **Total Test Errors** | 870 misclassifications (out of 10,000) | **691 misclassifications** | **179 fewer errors ($-20.6\%$ relative error reduction)** |
| **Pullover Recall** | 0.856 | **0.906** | **$+5.0$ percentage points** |
| **Coat Recall** | 0.875 | **0.924** | **$+4.9$ percentage points** |
| **Shirt Recall** | 0.750 | **0.763** | **$+1.3$ percentage points** |
| **Model Size / Params** | 18,094,716 numbers stored | **304,682 parameters** | **$\approx 60\times$ smaller footprint** |
| **CPU Latency** | 5.36 ms / image | **0.68 ms / image** | **$\approx 8\times$ faster single-threaded inference** |
| **Explainability** | Static gradient histograms in 4×4 cells | **Grad-CAM Saliency Maps** | Visualizes attention on collars, hemlines, and sleeves |

---

## 3. Models Used in Phase 2: Architecture & Rationale

We implemented three targeted architectures to isolate the effects of network depth, data augmentation, and spatial aggregation:

### A. Model 1: `CNN2Block` (Baseline ConvNet)
- **Architecture**:
  - `Conv2d(1, 32, kernel_size=3, pad=1)` $\to$ `BatchNorm2d` $\to$ `ReLU` $\to$ `MaxPool2d(2, 2)` (Shape: $32 \times 14 \times 14$)
  - `Conv2d(32, 64, kernel_size=3, pad=1)` $\to$ `BatchNorm2d` $\to$ `ReLU` $\to$ `MaxPool2d(2, 2)` (Shape: $64 \times 7 \times 7$)
  - `Flatten` $\to$ `Linear(3136, 128)` $\to$ `ReLU` $\to$ `Dropout(0.4)` $\to$ `Linear(128, 10)`
- **Parameters**: 421,738
- **Why We Used It**: 
  Serves as the rigorous baseline to test whether a standard 2-stage ConvNet can immediately outperform the best classical kernel SVM without architectural tricks. It achieved **92.16% test accuracy**, already proving that learned spatial convolutions beat handcrafted HOG (91.30%).

### B. Model 2: `CNN2Block_Aug` (Data Augmentation Ablation)
- **Architecture**: Identical to `CNN2Block`.
- **Training Change**: Applied on-the-fly spatial data augmentations:
  - `RandomCrop(28, padding=2, padding_mode="reflect")`
  - `RandomHorizontalFlip(p=0.5)`
- **Why We Used It**: 
  To test whether artificial spatial perturbations prevent overfitting and enhance generalization. 
- **Finding**: Test accuracy was **91.41%**. On $28\times28$ low-resolution clothing, random cropping sometimes removes 2-pixel border details (such as the outer edge of a sneaker or the hemline of a coat), acting as a slightly too strong regularizer for a shallow 2-block network.

### C. Model 3: `CNN3Block_GAP` (Deep Network with Global Average Pooling) — **Our Best Model**
- **Architecture**:
  - **Block 1 (Double Conv)**: `Conv(1→32)` $\to$ `BN` $\to$ `ReLU` $\to$ `Conv(32→32)` $\to$ `BN` $\to$ `ReLU` $\to$ `MaxPool(2, 2)` ($32 \times 14 \times 14$)
  - **Block 2 (Double Conv)**: `Conv(32→64)` $\to$ `BN` $\to$ `ReLU` $\to$ `Conv(64→64)` $\to$ `BN` $\to$ `ReLU` $\to$ `MaxPool(2, 2)` ($64 \times 7 \times 7$)
  - **Block 3 (Double Conv)**: `Conv(64→128)` $\to$ `BN` $\to$ `ReLU` $\to$ `Conv(128→128)` $\to$ `BN` $\to$ `ReLU` ($128 \times 7 \times 7$)
  - **Global Average Pooling**: `AdaptiveAvgPool2d((1, 1))` (Shape: $128 \times 1 \times 1$)
  - **Classifier**: `Linear(128, 128)` $\to$ `ReLU` $\to$ `Dropout(0.3)` $\to$ `Linear(128, 10)`
- **Parameters**: 304,682
- **Why We Used It**:
  1. **Expanded Receptive Field**: Double $3\times3$ convolutions at each stage allow the network to capture complex hierarchical shapes (edges $\to$ texture contours $\to$ full collar/sleeve structure).
  2. **Global Average Pooling (GAP) Innovation**: Traditional flattening of a $128 \times 7 \times 7$ tensor produces 6,272 inputs to a dense layer, adding over 800,000 dense weights. GAP collapses each feature channel to its spatial mean, enforcing translation invariance and reducing parameters to only **304k** while achieving the top score of **93.09% test accuracy**.

---

## 4. Alternate Architectures: What Were They and Why Did We NOT Use Them?

A critical requirement of rigorous machine learning research is explaining **why alternative models were rejected**. Below is our detailed justification:

### Alternate 1: Transfer Learning via Pretrained ImageNet Models (ResNet-18, ResNet-50, VGG-16, EfficientNet)
- **Description**: Taking a deep CNN pre-trained on ImageNet-1k, adapting the input conv layer for single-channel grayscale, and fine-tuning the classifier.
- **Why NOT Used**:
  1. **Extreme Resolution Mismatch**: ImageNet models expect $224 \times 224 \times 3$ RGB inputs. Fashion-MNIST is $28 \times 28 \times 1$ grayscale. Upsampling $28\times28$ by $8\times$ using bilinear or bicubic interpolation adds massive blur and artificial interpolation artifacts without introducing any new high-frequency information.
  2. **Massive Overparameterization**: ResNet-18 has $11.7\text{ million}$ parameters; ResNet-50 has $25.6\text{ million}$ parameters. Using 25M weights on 48,000 $28\times28$ images represents gross overparameterization that requires severe regularization and yields negligible gains over a custom compact network.
  3. **Inductive Domain Shift**: ImageNet filters are tuned to photographic textures, lighting gradients, and natural objects (dogs, cars, landscapes). Synthetic centered fashion thumbnails with pure black backgrounds ($50.2\%$ zero pixels) do not benefit from ImageNet priors.
  4. **Inference Latency & Deployment Cost**: A ResNet-50 takes $>25\text{ ms}$ on CPU per image, destroying the real-time tagging requirement of online catalog inventory.

### Alternate 2: Vision Transformers (ViT / Swin Transformer)
- **Description**: Splitting the image into $4\times4$ or $2\times2$ patches, applying linear patch projections, adding positional embeddings, and routing through multi-head self-attention blocks.
- **Why NOT Used**:
  1. **Lack of 2D Inductive Bias**: Convolutional networks possess two built-in inductive biases: *translation equivariance* (a button is a button anywhere in the image) and *spatial locality* (neighboring pixels are related). Transformers lack these biases and must learn spatial relationships from scratch.
  2. **Data Inefficiency on Small Datasets**: Vision Transformers are notoriously data-hungry. Without millions of images (e.g. JFT-300M or ImageNet-21k pre-training), ViTs severely overfit and underperform compact CNNs on 50,000-image benchmarks.
  3. **Small Patch Redundancy**: At $28\times28$, a $4\times4$ patch yields only $7 \times 7 = 49$ tokens. Self-attention across 49 tokens on grayscale thumbnails provides no empirical benefit over local convolutions.

### Alternate 3: Multi-Layer Perceptrons (Deep Dense Neural Networks / MLP-Mixer)
- **Description**: Stacking multiple fully connected layers (`Linear(784, 512) -> ReLU -> Linear(512, 256) -> ...`).
- **Why NOT Used**:
  1. **Spatial Structure Destruction**: Phase 1 proved that adjacent pixels correlate at $r \approx 0.75$, whereas shuffled pixels drop to $r \approx 0.16$. Flattening an image treats adjacent pixels the same as opposite corner pixels, discarding 2D topology.
  2. **Parameter Inefficiency**: Dense connections between 784 inputs and hidden layers require immense parameter counts without weight sharing across image locations.

### Alternate 4: Very Deep CNNs (10+ Convolutional Layers without Residual Shortcuts)
- **Description**: Stacking 10 to 15 standard convolutional and pooling layers.
- **Why NOT Used**:
  1. **Premature Spatial Collapse**: $28\times28$ can only be downsampled 4 times ($28 \to 14 \to 7 \to 3 \to 1$). Stacking 10 pooling layers causes spatial collapse before deep hierarchical representations can form.
  2. **Vanishing Gradients**: Without residual skip connections (ResNet-style), plain deep networks suffer from vanishing gradients during backpropagation.

---

## 5. Experimental Results and Analysis

### A. Unified Leaderboard (All 15 Models Across Phase 1 & 2)

```
========================================================================================================
Phase                   Model             Features              Test Acc   Macro-F1   CPU Latency  Params
========================================================================================================
Phase 2 (Deep Learning) CNN3Block_GAP     Learned CNN filters   93.09%     0.9306     0.68 ms      304.7k
Phase 2 (Deep Learning) CNN2Block         Learned CNN filters   92.16%     0.9212     0.18 ms      421.7k
Phase 2 (Deep Learning) CNN2Block_Aug     Learned CNN filters   91.41%     0.9140     0.22 ms      421.7k
Phase 1 (Classical ML)  SVM-RBF           HOG + Shape Stats     91.37%     0.9137     5.28 ms     17.5M
Phase 1 (Classical ML)  SVM-RBF           HOG (Phase 1 Best)    91.30%     0.9130     5.36 ms     18.1M
Phase 1 (Classical ML)  SVM-RBF           Raw Pixels            89.88%     0.8985     4.53 ms     15.0M
Phase 1 (Classical ML)  LogReg            HOG + Shape Stats     88.94%     0.8890     0.001 ms      8.8k
Phase 1 (Classical ML)  LogReg            HOG                   88.52%     0.8848     0.001 ms      8.7k
Phase 1 (Classical ML)  RandomForest      HOG + Shape Stats     88.21%     0.8811     0.041 ms      2.5M
Phase 1 (Classical ML)  RandomForest      HOG                   87.81%     0.8769     0.042 ms      3.2M
Phase 1 (Classical ML)  kNN               HOG                   87.74%     0.8777     0.271 ms     51.8M
Phase 1 (Classical ML)  RandomForest      Raw Pixels            87.60%     0.8746     0.037 ms      2.5M
Phase 1 (Classical ML)  kNN               Raw Pixels            85.61%     0.8564     0.316 ms     47.0M
Phase 1 (Classical ML)  LogReg            Raw Pixels            84.56%     0.8447     0.001 ms      7.9k
========================================================================================================
```

### B. Statistical Significance: Paired McNemar Test
We computed paired predictions on all 10,000 test images between `SVM-RBF (HOG)` and the Phase 2 CNN models using the continuity-corrected McNemar $\chi^2$ statistic:
$$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}, \quad \text{d.o.f.} = 1$$
where $b$ is the number of samples only SVM got correct, and $c$ is the number of samples only the CNN got correct.

| Comparison | $\Delta \text{Accuracy}$ | SVM Only Correct ($b$) | CNN Only Correct ($c$) | $\chi^2$ Statistic | $p$-value | Significance ($p < 0.001$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| SVM-RBF vs. `CNN2Block` | $+0.86\%$ | 313 | 399 | 10.15 | $1.44 \times 10^{-3}$ | Approaching ($p < 0.01$) |
| SVM-RBF vs. `CNN2Block_Aug` | $+0.11\%$ | 352 | 363 | 0.14 | $0.708$ | Not Significant |
| **SVM-RBF vs. `CNN3Block_GAP`** | **$+1.79\%$** | **266** | **445** | **44.56** | $\mathbf{2.46 \times 10^{-11}}$ | **Statistically Definitive ($p \ll 0.001$)** |

### C. Upper-Body Confusion & Per-Class Recall Breakdown
The core empirical challenge of Fashion-MNIST is separating upper-body clothing:
- **T-shirt/top**: 87.3% (SVM) $\to$ **88.5%** (`CNN3Block_GAP`)
- **Pullover**: 85.6% (SVM) $\to$ **90.6%** (`CNN3Block_GAP`) — **$+5.0$ point breakthrough**
- **Coat**: 87.5% (SVM) $\to$ **92.4%** (`CNN3Block_GAP`) — **$+4.9$ point breakthrough**
- **Shirt**: 75.0% (SVM) $\to$ **76.3%** (`CNN3Block_GAP`)
- **Trouser**: 97.5% (SVM) $\to$ **99.0%** (`CNN3Block_GAP`)
- **Bag**: 98.1% (SVM) $\to$ **99.2%** (`CNN3Block_GAP`)

Total misclassifications across the entire test set fell from **870 (SVM)** down to **691 (`CNN3Block_GAP`)**.

---

## 6. Explainability: Grad-CAM vs. Handcrafted HOG

In Phase 1, HOG computed gradient orientations inside rigid $4\times4$ cells with $2\times2$ block normalization. While this captured broad contours, it treated all cells equally regardless of whether a cell contained empty black space or a collar button.

In Phase 2, we implemented **Grad-CAM (Gradient-weighted Class Activation Mapping)** on the final convolutional layer of `CNN3Block_GAP`:
$$\alpha_k^c = \frac{1}{Z}\sum_{i}\sum_{j}\frac{\partial y^c}{\partial A_{i,j}^k}$$
$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_{k}\alpha_k^c A^k\right)$$

### Visual Inspection Findings:
1. **Shirts vs. T-shirts**: Grad-CAM shows localized activations concentrated squarely at the **collar neckline**. For open-collar shirts, activation peaks at the V-neck and placket slit. For T-shirts, activation forms a uniform circular arc across the crew neck.
2. **Pullovers vs. Coats**: For Coats, Grad-CAM highlights the **vertical center opening split** and lower front hem. For Pullovers, attention concentrates on the ribbed waistband and shoulder seam.
3. **Footwear & Accessories**: Sandals and Sneakers exhibit focused activations along the sole contour and ankle straps, yielding $>98\%$ recall.

---

## 7. Interactive Deliverables

All Phase 2 outcomes have been embedded directly into the project's interactive tools:
1. **Streamlit Studio (`app/streamlit_app.py`)**: 9 pages, including a live comparison (SVM vs first CNN vs final model, with Grad-CAM and photo upload), Phase 2 experiments, and an errors & reliability page with an interactive abstain threshold.
2. **Jupyter Notebook (`notebooks/04_deep_learning_cnn.ipynb`)**: training curves, confusion comparisons and Pareto plots for the first CNNs.
3. **PowerPoint Presentation (`docs/Fashion-MNIST_Phase2_Presentation.pptx`)**: 16 slides, built by `scripts/build_phase2_deck.js` from the result files.
4. **Presentation guide (`docs/presentation_guide.md`)**: speaker plan, demo flow, numbers cheat-sheet and all likely questions with answers.


---

## 8. Addendum (revision): controlled ablation and corrections

The full revised report is `report/phase2_full_report.pdf` (source: `report/phase2_full_report.tex`); the new deck is `docs/Fashion-MNIST_Phase2_Presentation.pptx`.

- **New control run `CNN3Block_GAP_NoAug`** (`scripts/07_ablation_gap_noaug.py`) completes a 2×2 grid. Test accuracy: CNN2Block 92.16% / CNN2Block_Aug 91.41% / CNN3Block_GAP_NoAug 91.38% / CNN3Block_GAP **93.09%**. Depth + GAP only pays off together with augmentation (+1.71 pts, McNemar p = 1.0×10⁻¹² between the two GAP runs).
- **Correction to Section 3B:** augmentation did not act as a "too strong regulariser". CNN2Block_Aug was still improving at its final epoch and has higher validation than training accuracy, which points to under-training within the 18-epoch budget.
- **Correction to Section 6:** the Grad-CAM maps are more mixed than "squarely at the collar"; several are diffuse. This is listed as a limitation in the final report.
- **95% bootstrap CIs for CNNs** (`results/tables/p2_bootstrap_ci_cnn.csv`): CNN3Block_GAP 92.57–93.59%, fully above SVM-RBF (HOG) 90.76–91.85%.
- Upper-body classes are the true class of 76% of CNN3Block_GAP errors (74% for the SVM), so the bottleneck became more concentrated. All figures: `scripts/08_phase2_story_figures.py` → `report/figures/p2_*.png`.

## 9. Final Phase 2 model (summary)

Full details: `report/phase2_full_report.pdf`, Section 5.

- 40-epoch training with 3 seeds: CNN3Block-GAP 93.80% ± 0.21, ResNetSmall 94.62% ± 0.13.
- **Final model = 3 × ResNetSmall + mirrored-image TTA + temperature scaling (+ Shirt rule): 95.20% test accuracy** (95% CI 94.78–95.62%), 480 errors, Shirt recall 85.4%.
- Calibration error (ECE) 7.6% → 1.3%; with abstention it answers 87.7% of images at 98.56% accuracy.
- McNemar vs SVM-RBF (HOG): p = 6.7×10⁻⁵⁰; vs the first CNN3Block-GAP: p = 2.9×10⁻²⁷.
