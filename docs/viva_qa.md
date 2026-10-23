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

**What changes in Phase 2?** CNN (2–3 conv blocks, pooling, dropout, early stopping) on the same splits, same metrics + parameter count + latency.

**Is there data leakage?** No: scalers/PCA fitted on training data; no duplicates between train and test (checked by hashing).
