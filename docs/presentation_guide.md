# Presentation guide — Phase 2 (final)

Everything the team needs for the Phase 2 presentation and viva in one place: who presents what,
how to run the live demo, the numbers to remember, and likely questions with answers.

- **Slides:** `docs/Fashion-MNIST_Phase2_Presentation.pptx` (16 slides, also as PDF)
- **Report:** `report/phase2_full_report.pdf`
- **Live demo:** `.venv/bin/streamlit run app/streamlit_app.py`
- Phase 1 slides and their English + Hinglish script: `docs/Fashion-MNIST_Phase1_Presentation.pptx`, `docs/presentation_script.md`

---

## 1. Speaker plan (≈ 15 min + demo)

| Member | Slides | Topic | Time |
|---|---|---|---|
| Member 1 | 1–3 | Title, the story, Phase 1 recap | 3 min |
| Member 2 | 4–6 | Architectures, training curves, 2×2 ablation | 3 min |
| Member 3 | 7–9 | Per-class results, Grad-CAM, from good to reliable | 3 min |
| Member 4 | 10–12 | Seeds, step-by-step improvements, calibration | 3 min |
| Member 5 | 13–16 + demo | Abstain, scorecard, limitations, close, live demo | 3 min + 3 min demo |

Rules: one message per slide, point at the axis titles when you show a chart ("x-axis is…, y-axis is…"), hand over with the next speaker's topic.

## 2. Live demo flow (Member 5, ≈ 3 min)

1. **Page 1 · Overview** — the four KPI cards and the accuracy journey chart.
2. **Page 2 · Try the Models** — click *Hard case #26 Shirt*: SVM wrong, final model right; show calibrated probabilities and the Grad-CAM heat-map. Optionally upload a clothing photo.
3. **Page 7 · Errors & Reliability → Calibration & abstain** — move the confidence slider: fewer images answered, higher accuracy.
4. **Page 9 · Leaderboard** — hover the accuracy-vs-speed chart; download buttons for the report and slides.

If the app is slow on first load, it is loading the models once (cached afterwards).

## 3. Numbers cheat-sheet

**Data & protocol:** 70,000 images, 28×28 grayscale, 10 classes · split 48,000 train / 12,000 validation / 10,000 test · seed 42 · all choices on validation · test used once.

**Phase 1:** best = SVM-RBF on HOG (864 features) **91.30%**, 870 errors, Shirt recall 75.0%, 18.1M stored numbers, 5.36 ms/image · 74% of SVM errors in T-shirt / Pullover / Coat / Shirt.

**Phase 2, first CNNs:** CNN2Block 92.16% · CNN2Block-Aug 91.41% · CNN3Block-GAP **93.09%** (305k weights, 0.68 ms) · CNN3Block-GAP without augmentation 91.38% · McNemar vs SVM p = 2.5×10⁻¹¹.

**Phase 2, final:**
| | Value |
|---|---|
| CNN3Block-GAP, 40 epochs, 3 seeds | 93.80% ± 0.21 |
| ResNetSmall, 3 seeds | 94.62% ± 0.13 (696k weights) |
| **Final model** (3 × ResNetSmall + mirrored-image TTA + temperature scaling) | **95.20%**, CI 94.78–95.62%, macro-F1 0.952 |
| Errors | **480** (SVM 870, first CNN 691) |
| Shirt / Pullover / Coat recall | 85.4% / 92.8% / 93.5% |
| Calibration error (ECE) | 7.6% → **1.3%** (temperature T = 0.65) |
| Abstain | answers 87.7% of images at **98.56%** accuracy; 26% of handed-off images are Shirts |
| McNemar | vs SVM p = 6.7×10⁻⁵⁰ · vs first CNN p = 2.9×10⁻²⁷ |
| Cost | 2.09M weights, 11.3 ms/image on CPU (3 nets × 2 views) |

---

## 4. Questions and answers

### Data and protocol
**Why is accuracy a fair metric here?** Classes are exactly balanced (6,000 train / 1,000 test each). We still report macro-F1 and per-class recall.

**Why take validation only from the training set?** The test set must stay unseen for every choice, otherwise the reported accuracy is optimistic. Selection on validation, one final test evaluation.

**Is there data leakage?** No. Scalers/PCA are fitted on training data only, and a byte-level hash check across all 70,000 images shows train, validation and test are disjoint (0 duplicates). Class proportions in train and validation match within 0.002.

**Were any final-model choices made on the test set?** No. The ensemble, the temperature, the Shirt bias and the abstain threshold were all chosen on the 12,000 validation images.

### Phase 1 (classical ML)
**What is HOG?** Per-pixel gradients → magnitude and direction → histogram of directions per small cell → normalisation over 2×2-cell blocks → one long vector. Ours: 6 direction bins, 4×4-pixel cells, 864 numbers.

**Why 4×4 cells, not the usual 8×8?** At 28×28, an 8×8 cell covers a third of the image. Smaller cells were better in the ablation; a 2,000-feature budget picked 4×4 with 6 bins.

**Why did Random Forest not benefit from HOG?** Trees split on single features and build their own interactions; McNemar p = 0.49 (not significant).

**Why did Logistic Regression benefit most (+4.0 pts)?** It is linear in its inputs; HOG gives it non-linear, neighbourhood information.

**Why was SVM-RBF the best classical model?** The RBF kernel gives smooth non-linear boundaries on 864 dense features. Cost: 5.36 ms/image and 18.1M stored numbers (support vectors).

**What does gamma = 'scale' mean?** γ = 1 / (number of features × variance of X).

**Why is Shirt the hardest class?** Its outline is almost the same as T-shirt, Pullover and Coat; the collar and buttons are only 1–2 pixels at 28×28. The class-mean distances in EDA already predicted this.

**How do you count "parameters" for kNN / RF / SVM?** Stored training matrix (kNN), 2 numbers per tree node (RF), support vectors + dual coefficients (SVM).

### Phase 2 — first CNNs
**Why move from classical ML to CNNs?** Phase 1 stopped at 91.3% and most errors were upper-body garments. HOG uses fixed cells; a CNN learns its own filters from the data.

**What is global average pooling (GAP) and why use it?** It averages each 7×7 feature map into one number. Flattening 128×7×7 would give 6,272 inputs and 800k+ dense weights; GAP needs 128 inputs, so the network over-fits less and has fewer weights (305k).

**What did the 2×2 ablation show?** Depth + GAP alone: 91.38% (over-fits after 5 epochs). Augmentation alone on the shallow net: 91.41%. Both together: 93.09%. Neither ingredient works alone.

**Why did augmentation first seem to hurt?** The augmented models were still improving at their last epoch — they were under-trained, not over-regularised. With 40 epochs the GAP network reaches 93.80% on average.

**What augmentation did you use?** Random 2-pixel shift (crop with reflect padding) and left–right flip; for ResNetSmall also Random Erasing (blanks a small random rectangle).

### Phase 2 — final model
**What is ResNetSmall?** A small residual network: a 3×3 stem and three stages of two residual blocks (32 → 64 → 128 channels, 28 → 14 → 7 pixels), then GAP and one dense layer. 696k weights.

**Why residual connections?** The shortcut lets gradients flow directly, so a deeper network (13 conv layers) trains without vanishing gradients.

**What is label smoothing?** Training targets become 0.9 for the right class instead of 1.0, spreading 0.1 over the others. It discourages over-confident predictions and helps generalisation.

**Why three seeds?** To show the result is not luck. The spread is small: ±0.13 points for ResNetSmall.

**What is test-time augmentation (TTA)?** The model also classifies a mirrored copy of each image and averages the two answers. It added about +0.2 points.

**Why an ensemble, and how was it chosen?** Averaging three independently trained networks cancels some of their random mistakes. Three candidates (3 × GAP, 3 × ResNet, all 6) were compared on validation; 3 × ResNetSmall won.

**What is calibration / ECE?** A model is calibrated if, among images where it says 80%, about 80% are correct. ECE is the average gap between stated confidence and actual accuracy. Ours: 7.6% → 1.3%.

**What is temperature scaling and why is T below 1?** All scores are divided by one number T, chosen on validation; it never changes the predicted class. The ensemble was under-confident (label smoothing + averaging soften probabilities), so T = 0.65 sharpens them.

**Is calibration perfect now?** No. About 88% of test images are in the 95–100% confidence group, where stated and actual accuracy agree closely. The small middle groups are still somewhat over-confident.

**What is the abstain option?** If confidence is below a threshold (chosen on validation for 99% accuracy), the image goes to a human. Result: 87.7% answered automatically at 98.56% accuracy.

**What is the Shirt rule?** A small constant (+0.2) is added to the Shirt score before choosing the answer; its size maximised validation macro-F1. On this test set it did not change overall accuracy (95.20% either way).

**Is the final model significantly better?** Yes. McNemar vs SVM: 538 vs 148 disagreements won, p = 6.7×10⁻⁵⁰. Vs the first CNN: 294 vs 83, p = 2.9×10⁻²⁷.

**What does it cost?** 2.09M weights and 11.3 ms per image on one CPU (3 networks × 2 views). Slower than the first CNN (0.68 ms) but still real-time. If speed matters more, one ResNetSmall gives 94.6% at about 1.9 ms.

**What is McNemar's test?** A paired test on the images where exactly one model is right: χ² = (|b − c| − 1)² / (b + c), 1 degree of freedom.

**How were the confidence intervals computed?** 2,000 bootstrap resamples of the 10,000 test images; 2.5th and 97.5th percentiles of accuracy.

### Explainability
**What is Grad-CAM?** It takes the gradient of the predicted class score with respect to the last convolutional layer, averages it per channel to weight the feature maps, and keeps the positive part. The result is a heat-map of the regions that pushed the network towards its answer.

**What did Grad-CAM show?** Attention is on the garment, never the background, often on cuffs, hems and collars. Some maps (patterned shirts, printed T-shirts) are diffuse. It is qualitative evidence only.

### Why not other models?
**Why not a pre-trained ResNet-50 / ImageNet model?** Built for 224×224 colour photos; upsampling 28×28 adds blur, not information; 11–25M weights for 48k tiny images.

**Why not a Vision Transformer?** No built-in locality bias; needs far more data than 48k images to compete.

**Why not a deep MLP?** Flattening throws away which pixels are neighbours (adjacent-pixel correlation r ≈ 0.75).

### Limitations
Shirt is still the hardest class; Grad-CAM is qualitative; only clean, centred thumbnails were tested (no real shop photos); limited hyper-parameter search; the final runs used Apple MPS, so single runs can move by ±0.1–0.3 points (hence 3 seeds).

### Engineering and the app
**How does the app stay fast?** `st.cache_data` caches tables and arrays, `st.cache_resource` keeps loaded models in memory. Cache keys include the file's modification time, so re-running the scripts refreshes the app automatically.

**How is latency measured?** Single image, CPU, 100 warm-up calls then 2,000 timed calls with `time.perf_counter()`.

**Are the numbers in the slides and report typed by hand?** No. `scripts/10_ensemble_calibration.py` writes the result files; the slides (`scripts/build_phase2_deck.js`), report (`scripts/11_report_numbers.py`) and app all read them.

**Data-pipeline contribution (split verification and leakage prevention):** `check_splits()` in `src/data_loader.py` and `scripts/00_check_splits.py` verify stratification (class proportions within 0.002), assert zero overlap between train/validation/test by hashing every image, check pixel range [0, 1], and confirm the official 10,000 test images are used once.

---

## 5. Rubric mapping

| Criterion | Where |
|---|---|
| Dataset quality & EDA | notebooks/01, report/main.pdf, app page 4 |
| Feature engineering | notebooks/02, report/main.pdf, app pages 3–4 |
| Model application (classical & CNN) | `src/models.py`, `src/cnn_models.py`, scripts 03, 05, 07, 09 |
| Experimental rigour | 2×2 ablation, 3 seeds, bootstrap CIs, McNemar — report §3 and §5 |
| Explainability & reliability | Grad-CAM (`src/cnn_explain.py`), calibration and abstain — app pages 2, 3, 7 |
| Code quality | modular `src/`, numbered scripts, fixed seeds, `scripts/run_all.sh` |
| Presentation | 16-slide deck + 9-page Streamlit app |
| Report | `report/phase2_full_report.pdf` (Phase 2), `report/main.pdf` (Phase 1) |
| Viva | this guide, section 4 |
