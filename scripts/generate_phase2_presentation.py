"""Generate the Phase 2 Deep Learning PowerPoint presentation (docs/Fashion-MNIST_Phase2_Presentation.pptx)."""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "docs" / "Fashion-MNIST_Phase2_Presentation.pptx"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

# Color Palette
DARK_NAVY = RGBColor(26, 36, 56)
ACCENT_BLUE = RGBColor(41, 128, 185)
ACCENT_GREEN = RGBColor(39, 174, 96)
TEXT_DARK = RGBColor(44, 62, 80)
TEXT_MUTED = RGBColor(127, 140, 141)
CARD_BG = RGBColor(245, 247, 250)
WHITE = RGBColor(255, 255, 255)

def add_header(slide, title_text, subtitle_text):
    tb = slide.shapes.add_textbox(Inches(0.8), Inches(0.5), Inches(11.7), Inches(1.2))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title_text
    p.font.name = "Arial"
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = DARK_NAVY
    
    p2 = tf.add_paragraph()
    p2.text = subtitle_text
    p2.font.name = "Arial"
    p2.font.size = Pt(14)
    p2.font.color.rgb = ACCENT_BLUE

def add_card(slide, left, top, width, height, title, body_bullets, accent_color=ACCENT_BLUE):
    card = slide.shapes.add_shape(1, left, top, width, height) # 1 = MSO_SHAPE.RECTANGLE
    card.fill.solid()
    card.fill.fore_color.rgb = CARD_BG
    card.line.color.rgb = accent_color
    card.line.width = Pt(2)
    
    tf = card.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title
    p.font.name = "Arial"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = DARK_NAVY
    p.alignment = PP_ALIGN.LEFT
    
    for bullet in body_bullets:
        pb = tf.add_paragraph()
        pb.text = "• " + bullet
        pb.font.name = "Arial"
        pb.font.size = Pt(13)
        pb.font.color.rgb = TEXT_DARK
        pb.alignment = PP_ALIGN.LEFT

blank_layout = prs.slide_layouts[6]

# SLIDE 1: Title
s1 = prs.slides.add_slide(blank_layout)
tb1 = s1.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.3), Inches(3.5))
tf1 = tb1.text_frame
tf1.word_wrap = True
p = tf1.paragraphs[0]
p.text = "Automatic Clothing Category Recognition"
p.font.size = Pt(36)
p.font.bold = True
p.font.color.rgb = DARK_NAVY

p = tf1.add_paragraph()
p.text = "Phase 2: Deep Learning Evaluation (Convolutional Neural Networks)"
p.font.size = Pt(22)
p.font.color.rgb = ACCENT_GREEN
p.font.bold = True

p = tf1.add_paragraph()
p.text = "Learned Hierarchical Representations vs. Handcrafted HOG · Global Average Pooling · Grad-CAM Interpretability"
p.font.size = Pt(14)
p.font.color.rgb = TEXT_MUTED

tb_meta = s1.shapes.add_textbox(Inches(1.0), Inches(5.0), Inches(11.3), Inches(1.5))
tf_m = tb_meta.text_frame
p = tf_m.paragraphs[0]
p.text = "Team T002 · Newton School of Technology | Advanced Machine Learning"
p.font.size = Pt(13)
p.font.bold = True
p.font.color.rgb = DARK_NAVY
p = tf_m.add_paragraph()
p.text = "Authors: MD. Sajjan, Sibtain Ahmed Qureshi, Syed Darain Qamar, Antik Mandal, Abuzar Haideri"
p.font.size = Pt(12)
p.font.color.rgb = TEXT_MUTED

# SLIDE 2: Problem & Phase 1 Bottleneck
s2 = prs.slides.add_slide(blank_layout)
add_header(s2, "Phase 1 Retrospective & The Motivation for Phase 2", "Where handcrafted features succeeded and where they reached an empirical ceiling")
add_card(s2, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8), "Phase 1 Accomplishments", [
    "HOG descriptors significantly improved 3 of 4 classical learners over raw pixels.",
    "Logistic Regression jumped +4.0% (84.6% → 88.5%, p < 10⁻⁶).",
    "Best Classical Model: SVM-RBF on HOG reached 91.30% test accuracy.",
    "Data pipeline was completely leak-free and verified by automated byte-hashing.",
], ACCENT_BLUE)
add_card(s2, Inches(6.8), Inches(1.8), Inches(5.6), Inches(4.8), "The Phase 1 Bottleneck", [
    "Upper-Body garments ({Shirt, T-shirt, Pullover, Coat}) caused 60–65% of ALL errors.",
    "Shirt recall reached only 0.750 in the best classical model.",
    "At 28x28 grayscale, collars, buttons and cuffs are only 1–2 pixels wide.",
    "Rigid HOG cell grids cannot adaptively align to subtle non-linear contours.",
    "SVM-RBF stored 18.1M numbers and had high latency (5.36 ms/image).",
], RGBColor(231, 76, 60))

# SLIDE 3: Architectures
s3 = prs.slides.add_slide(blank_layout)
add_header(s3, "Phase 2 CNN Architectures Evaluated", "Designing compact, expressive convolutional networks for 28x28 clothing items")
add_card(s3, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8), "1. Baseline: CNN2Block", [
    "Block 1: Conv(1→32, 3x3, pad 1) + BN + ReLU + MaxPool(2x2)",
    "Block 2: Conv(32→64, 3x3, pad 1) + BN + ReLU + MaxPool(2x2)",
    "Classifier: Flatten → Linear(3136, 128) + ReLU + Dropout(0.4) → Linear(10)",
    "Trainable Parameters: 421,738",
    "Evaluated both without and with spatial data augmentation (flip + crop).",
], ACCENT_BLUE)
add_card(s3, Inches(6.8), Inches(1.8), Inches(5.6), Inches(4.8), "2. Modernized: CNN3Block-GAP", [
    "Stage 1: [Conv(32) + BN + ReLU + Conv(32) + BN + ReLU] + MaxPool",
    "Stage 2: [Conv(64) + BN + ReLU + Conv(64) + BN + ReLU] + MaxPool",
    "Stage 3: [Conv(128) + BN + ReLU + Conv(128) + BN + ReLU]",
    "Global Average Pooling (GAP): AdaptiveAvgPool2d((1, 1))",
    "Classifier: Linear(128, 128) + ReLU + Dropout(0.3) → Linear(10)",
    "Trainable Parameters: 304,682 (cuts 117k dense weights!).",
], ACCENT_GREEN)

# SLIDE 4: Leaderboard
s4 = prs.slides.add_slide(blank_layout)
add_header(s4, "Unified Empirical Results (10,000 Test Images)", "Comparing Classical ML (Phase 1) vs Deep Learning (Phase 2)")
add_card(s4, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8), "Top Performers Leaderboard", [
    "1. CNN3Block-GAP (Phase 2): 93.09% Acc | 0.9306 Macro-F1",
    "2. CNN2Block (Phase 2): 92.16% Acc | 0.9212 Macro-F1",
    "3. CNN2Block + Aug (Phase 2): 91.41% Acc | 0.9140 Macro-F1",
    "4. SVM-RBF + HOG+Stats (Phase 1): 91.37% Acc | 0.9137 Macro-F1",
    "5. SVM-RBF + HOG (Phase 1): 91.30% Acc | 0.9130 Macro-F1",
    "6. SVM-RBF + Pixels (Phase 1): 89.88% Acc | 0.8985 Macro-F1",
    "7. LogReg + HOG (Phase 1): 88.52% Acc | 0.8848 Macro-F1",
    "8. Random Forest (Phase 1): 87.81% Acc | 0.8769 Macro-F1",
], ACCENT_GREEN)
add_card(s4, Inches(6.8), Inches(1.8), Inches(5.6), Inches(4.8), "Key Performance Milestones", [
    "Accuracy Ceiling Broken: CNN3Block-GAP exceeds SVM-RBF by +1.79 percentage points.",
    "Validation Convergence: Reached 94.33% val accuracy with AdamW & Cosine Annealing.",
    "Total Test Errors: Dropped from 870 (SVM) down to 691 (CNN) — 179 fewer errors!",
    "Zero Test Leakage: Maintained identical 48k/12k/10k splits (SEED=42).",
], ACCENT_BLUE)

# SLIDE 5: Statistical Significance
s5 = prs.slides.add_slide(blank_layout)
add_header(s5, "Statistical Validation: McNemar Paired Significance Test", "Confirming that deep learning gains are genuine and not random test variance")
add_card(s5, Inches(0.8), Inches(1.8), Inches(11.7), Inches(4.8), "McNemar Test: SVM-RBF (HOG) vs CNN3Block-GAP", [
    "Evaluation on all 10,000 paired test set predictions:",
    "• Test Accuracy Delta: +1.79% (91.30% → 93.09%)",
    "• Samples where CNN was correct but SVM failed: 445 images",
    "• Samples where SVM was correct but CNN failed: 266 images",
    "• McNemar Chi-Square Statistic: χ² = 44.56 (continuity-corrected, 1 d.o.f.)",
    "• p-value: p = 2.46 × 10⁻¹¹  (« 0.001)",
    "Conclusion: CNN3Block-GAP is statistically superior to the best Phase 1 classifier at the highest level of empirical significance.",
], ACCENT_GREEN)

# SLIDE 6: Error Analysis & Recall
s6 = prs.slides.add_slide(blank_layout)
add_header(s6, "Resolving the Upper-Body Garment Ambiguity", "How hierarchical convolutional filters unlocked the hardest clothing categories")
add_card(s6, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8), "Per-Class Recall Comparison", [
    "Pullover Recall: 85.6% (SVM) → 90.6% (CNN3Block)  [+5.0%]",
    "Coat Recall: 87.5% (SVM) → 92.4% (CNN3Block)  [+4.9%]",
    "Shirt Recall: 75.0% (SVM) → 76.3% (CNN3Block)  [+1.3%]",
    "T-shirt Recall: 87.3% (SVM) → 88.5% (CNN3Block)  [+1.2%]",
    "Trouser & Bag: Exceed 99.0% recall (99.0% and 99.2%).",
], ACCENT_GREEN)
add_card(s6, Inches(6.8), Inches(1.8), Inches(5.6), Inches(4.8), "Impact on Error Distribution", [
    "Phase 1 Total Errors: 1,544 (LogReg) → 870 (SVM-RBF).",
    "Phase 2 Total Errors: Reduced to 691 (CNN3Block-GAP).",
    "Upper-body errors dropped from 552 (SVM) to 460 (CNN).",
    "The CNN successfully separated open outerwear (Coats) from crew-neck knitwear (Pullovers) using learned spatial filters.",
], ACCENT_BLUE)

# SLIDE 7: Grad-CAM Explainability
s7 = prs.slides.add_slide(blank_layout)
add_header(s7, "Explainability: Grad-CAM Saliency Maps", "Visualizing what the CNN's final convolutional layer focuses on to make predictions")
add_card(s7, Inches(0.8), Inches(1.8), Inches(11.7), Inches(4.8), "How Learned CNN Filters Differ from Fixed HOG Grids", [
    "• Fixed HOG Limitation: Rigidly counts edge angles in fixed 4x4 cells regardless of object semantics.",
    "• Grad-CAM Mechanism: Computes gradients of the target class score with respect to feature maps in Conv Block 3.",
    "• Collar & Neckline Focus: CNN actively focuses on collar opening depth to separate open Shirts from T-shirts.",
    "• Sleeve & Cuff Transitions: Attention maps highlight sleeve edges to distinguish Pullovers from short tops.",
    "• Hemline Splitting: Grad-CAM highlights the open vertical split at the bottom of Coats versus closed Pullover waists.",
    "• Interactive Studio: Evaluators can run real-time Grad-CAM on any test index in Streamlit Page 8.",
], ACCENT_BLUE)

# SLIDE 8: Efficiency & Pareto
s8 = prs.slides.add_slide(blank_layout)
add_header(s8, "Computational Efficiency & Pareto Frontier", "Why CNN3Block-GAP dominates the cost-performance trade-off")
add_card(s8, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8), "Storage Footprint (Parameters)", [
    "SVM-RBF (HOG): 18.1 Million numbers (Support Vectors)",
    "k-NN (Pixels): 47.0 Million numbers (Full Dataset)",
    "CNN2Block: 421,738 parameters",
    "CNN3Block-GAP: 304,682 parameters",
    "Result: ~60x reduction in memory compared to SVM-RBF!",
], ACCENT_GREEN)
add_card(s8, Inches(6.8), Inches(1.8), Inches(5.6), Inches(4.8), "Inference Latency (CPU Single-Thread)", [
    "Logistic Regression: 0.001 ms/image (ultrafast linear)",
    "SVM-RBF (HOG): 5.36 ms/image (slow kernel distances)",
    "CNN3Block-GAP: 0.68 ms/image",
    "CNN2Block: 0.18 ms/image",
    "Result: ~8x faster inference than the best Phase 1 SVM!",
], ACCENT_BLUE)

# SLIDE 9: Conclusions
s9 = prs.slides.add_slide(blank_layout)
add_header(s9, "Project Conclusions & Takeaways", "Summary of the complete Phase 1 + Phase 2 investigation")
add_card(s9, Inches(0.8), Inches(1.8), Inches(11.7), Inches(4.8), "Executive Takeaways", [
    "1. Handcrafted features (HOG) provide strong gains for linear models (+4.0%), but hit a ceiling at 91.3%.",
    "2. Tree ensembles (Random Forest) do not benefit from HOG (p = 0.49), creating their own pixel splits.",
    "3. Convolutional Neural Networks break the classical ceiling, reaching 93.09% test accuracy (p = 2.46 × 10⁻¹¹).",
    "4. Global Average Pooling (GAP) is key: it expands feature depth while cutting parameters from 1.2M to 304k.",
    "5. CNN3Block-GAP establishes a new Pareto optimum: +1.79% accuracy, 60x smaller model, and 8x faster CPU inference.",
    "6. Reproducibility: All code, models, notebooks, and the 8-page Streamlit studio are fully containerized and verified.",
], DARK_NAVY)

# SLIDE 10: Thank You
s10 = prs.slides.add_slide(blank_layout)
tb10 = s10.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(11.3), Inches(3.0))
tf10 = tb10.text_frame
p = tf10.paragraphs[0]
p.text = "Thank You!"
p.font.size = Pt(44)
p.font.bold = True
p.font.color.rgb = DARK_NAVY

p = tf10.add_paragraph()
p.text = "Questions & Live Demonstration"
p.font.size = Pt(22)
p.font.color.rgb = ACCENT_GREEN

p = tf10.add_paragraph()
p.text = "Interactive Studio: streamlit run app/streamlit_app.py (Pages 1–8)"
p.font.size = Pt(16)
p.font.color.rgb = TEXT_MUTED

prs.save(str(OUT_PATH))
print(f"Presentation saved to: {OUT_PATH}")

