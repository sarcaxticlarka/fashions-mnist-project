# Viva voce — likely questions with answers

**Why is accuracy fine here?** Classes are exactly balanced (6,000/1,000 each). We still report macro-F1 and per-class recall.

**Why validation only from train?** The test set must stay unseen for tuning, else the reported accuracy is optimistic. Selection on val, one test evaluation.

**Why refit on train+val?** More data with already-chosen hyper-parameters; standard practice, no test leakage.

**What is HOG?** Per-pixel gradients → magnitude/orientation → per-cell orientation histograms (weighted by magnitude) → L2-Hys normalisation over 2×2-cell blocks → concatenation. Our 864-d: 6 bins × 4 cells × 6×6 blocks.

**Why cell 4×4 and not the classic 8×8?** 28×28 is tiny; 8×8 cells would cover a third of the image. Ablation showed smaller cells are better; the 2,000-d budget picked 4×4, 6 bins.

**Why did Random Forest not benefit?** Trees threshold single pixels hierarchically and build their own interactions; McNemar p = 0.49.

**Why does Logistic Regression benefit most?** It is linear in its inputs; HOG supplies non-linear, neighbourhood features.

**Why SVM RBF best?** Kernel creates smooth non-linear boundaries; works well with 864-d dense features. Cost: 5 ms/image and 18M stored numbers.

**What does gamma='scale' mean?** γ = 1/(d·Var(X)).

**Why standardise for LogReg only?** Regularised gradient-based fit is scale-sensitive; HOG is already normalised, SVM/kNN/RF use the same raw scale for pixels. Shape stats are divided by fixed constants so they don't dominate distances (before: SVM hog_stats 88.7 %, after: 91.4 %).

**What is McNemar's test?** Paired test on discordant predictions b (only A right) and c (only B right): χ²=(|b−c|−1)²/(b+c), 1 d.o.f.

**What are the confidence intervals?** 95 % bootstrap over the 10,000 test images (1,000 resamples): best model [0.908, 0.919].

**Why is Shirt hardest?** Near-identical silhouette to T-shirt/Pullover/Coat; distinguishing collar/buttons are 1–2 px at 28×28. EDA mean-image distance already predicted this.

**How do you count parameters for kNN/RF/SVM?** Table in report §V: stored matrix / 2 per internal node / support vectors + dual coefs.

**Limitations?** HOG parameters chosen by an LR probe; one split; SVM tuned on a 15k subsample; 28×28 grayscale.

---

## Phase 2 (Deep Learning / CNN) Viva Questions

**Why did you implement CNNs for Phase 2 instead of sticking with classical ML?**  
In Phase 1, the best classical model (SVM-RBF on HOG) reached an accuracy ceiling of 91.3%, with 60–65% of all errors trapped inside the ambiguous upper-body garment cluster (`{T-shirt, Pullover, Coat, Shirt}`). At $28\times28$ resolution, discriminating features (collars, plackets, buttons) are 1–2 pixels wide. Handcrafted HOG descriptors bin edge gradients rigidly into predefined $4\times4$ cells, whereas CNNs learn hierarchical spatial filters end-to-end that adapt specifically to subtle neckline and hemline contours.

**What CNN architectures did you design and compare?**  
We implemented two primary architectures:
1. **CNN2Block (Baseline)**: A 2-stage convolutional network ($32 \to 64$ filters, $3\times3$ kernels, BatchNorm, ReLU, MaxPool $2\times2$) followed by `Flatten`, `Linear(3136, 128)`, `Dropout(0.4)`, and `Linear(128, 10)`. Achieves 92.16% test accuracy with 421.7k parameters.
2. **CNN3Block-GAP (Modernized Deep ConvNet)**: A 3-stage double-conv network ($32 \to 64 \to 128$ filters) with **Global Average Pooling (GAP)** (`AdaptiveAvgPool2d((1, 1))`) replacing the high-dimensional flattening layer. Achieves **93.09% test accuracy** with only **304.7k parameters**.

**Why use Global Average Pooling (GAP) instead of Flatten + Dense?**  
A traditional flattening layer after a $128\times7\times7$ feature map creates $128 \times 49 = 6,272$ inputs to a dense layer, adding over 800,000 parameters that are prone to severe overfitting. GAP reduces each feature map channel to a single scalar average ($128\times1\times1$), acting as a strong structural regularizer, enforcing translation invariance, and reducing parameter count from ~1.2M down to only 304k.

**Is the CNN improvement over SVM-RBF statistically significant?**  
Yes. We evaluated paired test predictions using the continuity-corrected McNemar $\chi^2$ test on the 10,000 test images. `CNN3Block-GAP` vs. `SVM-RBF (HOG)` yields $\chi^2 = 44.56$ and $p = 2.46 \times 10^{-11} \ll 0.001$. The CNN was uniquely correct on 445 samples where the SVM failed, whereas the SVM was uniquely correct on only 266 samples.

**Did the CNN resolve the upper-body garment bottleneck?**  
Significantly:
- **Pullover recall** increased from 85.6% (SVM) $\to$ **90.6%** (CNN) (+5.0 points).
- **Coat recall** increased from 87.5% (SVM) $\to$ **92.4%** (CNN) (+4.9 points).
- **Shirt recall** improved from 75.0% (SVM) $\to$ **76.3%** (CNN).
- **Total test errors** fell from 870 down to **691** (179 fewer errors).

**How does inference latency and storage compare between CNN and SVM?**  
- **Storage/Parameters**: SVM-RBF stores **18.1 million numbers** (support vectors and dual coefficients). `CNN3Block-GAP` has only **304,682 weights and biases** (a $\approx 60\times$ reduction in model footprint).
- **Latency**: Single-sample CPU inference takes **0.68 ms** for `CNN3Block-GAP` versus **5.36 ms** for SVM-RBF (an $\approx 8\times$ speedup).

**What is Grad-CAM and how did you use it?**  
Grad-CAM (Gradient-weighted Class Activation Mapping) calculates the gradient of the predicted class score with respect to feature activation maps of the final convolutional layer: $\alpha_k^c = \frac{1}{Z}\sum_i \sum_j \frac{\partial y^c}{\partial A_{i,j}^k}$. We take a rectified linear combination $L^c = \text{ReLU}(\sum_k \alpha_k^c A^k)$ to produce a $28\times28$ heatmap. Visualizing Grad-CAM on ambiguous shirts and pullovers proves that the CNN concentrates attention on the collar opening, sleeve transitions, and hemlines.

**Is there data leakage?** No: scalers/PCA fitted on training data; no duplicates between train and test (checked by hashing).

---

## Data Pipeline, Leakage Prevention & Split Verification Viva Questions

**What was your specific contribution to this project?**  
I led the **Data Pipeline Integrity, Split Verification, and Leakage Prevention** engineering (`src/data_loader.py` `check_splits()` and `scripts/00_check_splits.py`):
1. **Strict Stratification Verification**: Guaranteed that the 10-class proportions in train and validation partitions match to a margin of $< 0.002$ using `np.bincount`, preventing class distribution shift between tuning and evaluation sets.
2. **Byte-Level Data Leakage Prevention**: Built automated hashing verification using `hash(im.tobytes())` across all $70,000$ images, programmatically asserting that train, validation, and test sets are strictly disjoint ($0$ duplicate or leaked samples).
3. **Range & Dimensionality Assertions**: Implemented automated checks asserting all pixel values fall strictly in $[0, 1]$, and verified that the test partition strictly retains the official $10,000$ Zalando benchmark images used only once for final evaluation.

**How does Streamlit handle caching, and how did you optimize frontend performance?**  
We utilized Streamlit's two primary caching primitives:
- `@st.cache_data`: Applied to immutable tabular and matrix datasets (`get_splits()`, `get_results()`). It caches serializable data structures in memory, ensuring that re-rendering sliders or dropdowns takes $<5$ ms without re-reading the 70,000 IDX files.
- `@st.cache_resource`: Applied to heavy model objects (`_load_model()`). Trained estimators (such as the 48 MB SVM-RBF model) are deserialized via `joblib` once and held in memory across user sessions, preventing costly disk I/O on every interaction.

**How is inference latency measured in the live interface?**  
When an image is passed to `model.predict()`, we extract the feature representation (either 784-d raw pixels or 864-d HOG via `compute_hog(img[None], n_jobs=1)`), benchmark execution time with high-resolution monotonic clocks (`time.perf_counter()`), and report both feature extraction time and model inference time. This allows the evaluator to see why Logistic Regression runs in ~0.05 ms while SVM-RBF takes ~5 ms per image.
