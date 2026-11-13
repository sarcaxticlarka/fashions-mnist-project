/*
 * Builds docs/Fashion-MNIST_Phase2_Presentation.pptx
 *
 *   python scripts/08_phase2_story_figures.py      # writes results/tables/p2_deck_data.json
 *   NODE_PATH=<dir with pptxgenjs, react-icons, react, react-dom, sharp> node scripts/build_phase2_deck.js
 *
 * All charts are native PowerPoint charts (editable, with axis titles) built
 * from the JSON above, so the deck always matches the saved results.
 */
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");

const ROOT = path.resolve(__dirname, "..");
const D = JSON.parse(fs.readFileSync(path.join(ROOT, "results/tables/p2_deck_data.json"), "utf8"));
const OUT = path.join(ROOT, "docs/Fashion-MNIST_Phase2_Presentation.pptx");
const FS = JSON.parse(fs.readFileSync(path.join(ROOT, "results/tables/p2_final_summary.json"), "utf8"));
function csv(name) {
  const [h, ...rows] = fs.readFileSync(path.join(ROOT, "results/tables", name), "utf8").trim().split("\n");
  const keys = h.split(",");
  return rows.map((r) => Object.fromEntries(r.split(",").map((v, i) => [keys[i], isNaN(+v) || v === "" ? v : +v])));
}
const PER_RUN = csv("p2_final_per_run.csv"), BIAS = csv("p2_final_shirt_bias.csv");
const pc = (v, d = 2) => (v * 100).toFixed(d) + "%";
const F = { acc: pc(FS.test_acc), accN: FS.test_acc * 100, err: FS.errors, shirt: pc(FS.per_class_recall.Shirt, 1),
  eceB: pc(FS.ece_before), eceA: pc(FS.ece_after), cov: pc(FS.abstain.test_coverage, 1), kept: pc(FS.abstain.test_acc_on_kept),
  ms: FS.ms_per_image.toFixed(1), params: FS.params };
const APPLY_THEME = process.env.PPTX_APPLY_THEME; // optional path to the pptx skill's apply_theme.js
const FULL = process.env.FULL_DECK === "1"; // FULL_DECK=1 adds 8 detail slides (24 total)

// ---------------------------------------------------------------- theme
const HEX = {
  ink: "1D1B2E", ink2: "2A2742", violet: "6250D6", violetDark: "3E2DA8", lav: "F1EFFC",
  lav2: "E4E0FA", orange: "EB6834", aqua: "1BAF7A", grey: "B4B8C4", muted: "5B5A6B",
  white: "FFFFFF", line: "E6E6EE",
};
const THEME = {
  name: "Fashion-MNIST Violet",
  headFontFace: "Arial",
  bodyFontFace: "Calibri",
  colors: {
    dk1: HEX.ink, lt1: HEX.white, dk2: HEX.ink2, lt2: HEX.lav,
    accent1: HEX.violet, accent2: HEX.orange, accent3: HEX.aqua, accent4: HEX.grey,
    accent5: HEX.violetDark, accent6: HEX.muted, hlink: HEX.violet, folHlink: HEX.violetDark,
  },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.title = "Fashion-MNIST Phase 2: Deep Learning";
pres.author = "Team T002";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const W = 13.333, H = 7.5, MX = 0.6;

// ---------------------------------------------------------------- icons
async function icon(Comp, color, size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + color, size: String(size) }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

// ---------------------------------------------------------------- layouts
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: HEX.white },
  objects: [
    { placeholder: { options: { name: "kicker", type: "body", x: MX, y: 0.32, w: 9, h: 0.32, fontSize: 12, bold: true,
        color: HEX.violet, charSpacing: 2, margin: 0, valign: "middle", align: "left" }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: MX, y: 0.62, w: W - 2 * MX, h: 0.8, fontSize: 30,
        bold: true, color: HEX.ink, fontFace: "Arial", margin: 0, valign: "top", align: "left" }, text: "" } },
    { text: { text: "Fashion-MNIST  ·  Phase 2 Deep Learning", options: { x: MX, y: 7.05, w: 6, h: 0.3, fontSize: 10,
        color: HEX.muted, margin: 0 } } },
  ],
  slideNumber: { x: W - MX - 0.5, y: 7.05, w: 0.5, h: 0.3, fontSize: 10, color: HEX.muted, align: "right" },
});
pres.defineSlideMaster({
  title: "DARK",
  background: { color: HEX.ink },
  objects: [],
});

const content = (kicker, title, section) => {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(kicker.toUpperCase(), { placeholder: "kicker" });
  s.addText(title, { placeholder: "title" });
  return s;
};

// ---------------------------------------------------------------- helpers
const shadow = () => ({ type: "outer", color: "1D1B2E", blur: 10, offset: 2, angle: 90, opacity: 0.10 });
function card(s, x, y, w, h, fill = HEX.lav, name = "card") {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.14, fill: { color: fill },
    line: { color: fill }, shadow: fill === HEX.white ? shadow() : undefined, objectName: name });
}
function iconDot(s, data, x, y, d = 0.62, fill = HEX.violet) {
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill } });
  const p = d * 0.26;
  s.addImage({ data, x: x + p, y: y + p, w: d - 2 * p, h: d - 2 * p });
}
function txt(s, text, o) { s.addText(text, Object.assign({ isTextBox: true, margin: 0, color: HEX.ink, fontSize: 15, valign: "top" }, o)); }
function takeaway(s, text, y = 6.35) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y, w: W - 2 * MX, h: 0.55, rectRadius: 0.12,
    fill: { color: HEX.lav }, line: { color: HEX.lav } });
  s.addText([{ text: "Takeaway  ", options: { bold: true, color: HEX.violet } }, { text, options: { color: HEX.ink } }],
    { isTextBox: true, x: MX + 0.25, y, w: W - 2 * MX - 0.5, h: 0.55, fontSize: 15, valign: "middle", margin: 0 });
}
function chartOpts(extra) {
  return Object.assign({
    catAxisLabelColor: HEX.muted, valAxisLabelColor: HEX.muted, catAxisLabelFontSize: 12, valAxisLabelFontSize: 11,
    catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
    valGridLine: { color: HEX.line, size: 0.75 }, catGridLine: { style: "none" },
    catAxisLineShow: true, catAxisLineColor: "C9C9D6", valAxisLineShow: false,
    showCatAxisTitle: true, showValAxisTitle: true,
    catAxisTitleColor: HEX.ink, valAxisTitleColor: HEX.ink, catAxisTitleFontSize: 13, valAxisTitleFontSize: 13,
    catAxisTitleFontFace: "+mn-lt", valAxisTitleFontFace: "+mn-lt",
    dataLabelColor: HEX.ink, dataLabelFontSize: 11, dataLabelFontFace: "+mn-lt",
    legendFontSize: 12, legendFontFace: "+mn-lt", legendColor: HEX.ink, legendPos: "t",
  }, extra);
}
const r1 = (v) => Math.round(v * 10) / 10;
const r2 = (v) => Math.round(v * 100) / 100;

(async () => {
  const I = {};
  const want = { shirt: fa.FaTshirt, chart: fa.FaChartLine, flask: fa.FaFlask, rocket: fa.FaRocket,
    scale: fa.FaBalanceScale, search: fa.FaSearch, target: fa.FaBullseye, bolt: fa.FaBolt, compress: fa.FaCompressArrowsAlt,
    layers: fa.FaLayerGroup, shield: fa.FaShieldAlt, check: fa.FaCheckCircle, warn: fa.FaExclamationTriangle,
    eye: fa.FaEye, cogs: fa.FaCogs, redo: fa.FaRedo, cubes: fa.FaCubes, db: fa.FaDatabase, brain: fa.FaBrain,
    flag: fa.FaFlagCheckered, arrow: fa.FaArrowRight };
  for (const [k, c] of Object.entries(want)) {
    I[k] = await icon(c, HEX.white);
    I[k + "V"] = await icon(c, HEX.violet);
    I[k + "O"] = await icon(c, HEX.orange);
  }
  const best = D.leaderboard.find((r) => r.label === "CNN3Block GAP");
  const svm = D.leaderboard.find((r) => r.label === "SVM-RBF (HOG)");

  // ============================================================ 1. TITLE
  pres.addSection({ title: "Introduction" });
  let s = pres.addSlide({ masterName: "DARK", sectionTitle: "Introduction" });
  s.addShape(pres.shapes.OVAL, { x: 9.3, y: -1.6, w: 5.8, h: 5.8, fill: { color: HEX.violet, transparency: 15 }, line: { color: HEX.violet, transparency: 100 } });
  s.addShape(pres.shapes.OVAL, { x: 11.2, y: 3.6, w: 3.2, h: 3.2, fill: { color: HEX.orange, transparency: 10 }, line: { color: HEX.orange, transparency: 100 } });
  s.addImage({ data: I.shirt, x: 10.9, y: 0.9, w: 1.9, h: 1.9 });
  txt(s, "PROJECT 11  ·  AUTOMATIC CLOTHING CATEGORY RECOGNITION", { x: MX + 0.1, y: 0.9, w: 9, h: 0.35, fontSize: 13, bold: true, color: "B9B0F5", charSpacing: 2 });
  txt(s, "Phase 2: Teaching a CNN\nto See Clothes", { x: MX + 0.1, y: 1.45, w: 9.2, h: 1.9, fontSize: 48, bold: true, color: HEX.white, fontFace: "Arial" });
  txt(s, "From handcrafted features to a calibrated, reliable CNN", { x: MX + 0.1, y: 3.4, w: 8.5, h: 0.5, fontSize: 18, color: "D6D3F0" });
  const stats = [[F.acc, "final test accuracy"], [`${F.err}`, "errors (SVM: 870)"], [F.eceA, "calibration error"], [F.kept, "when confident"]];
  stats.forEach(([n, l], i) => {
    const x = MX + 0.1 + i * 2.2;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 4.45, w: 2.0, h: 1.25, rectRadius: 0.14, fill: { color: HEX.ink2 }, line: { color: "3A3660" } });
    txt(s, n, { x, y: 4.55, w: 2.0, h: 0.65, fontSize: 30, bold: true, color: i === 0 ? "FF9A6B" : HEX.white, align: "center", valign: "middle" });
    txt(s, l, { x, y: 5.18, w: 2.0, h: 0.35, fontSize: 13, color: "B9B6CF", align: "center" });
  });
  txt(s, "Team T002  ·  Newton School of Technology  ·  Deep Learning", { x: MX + 0.1, y: 6.35, w: 10, h: 0.3, fontSize: 13, bold: true, color: HEX.white });
  txt(s, "MD. Sajjan, Sibtain Ahmed Qureshi, Syed Darain Qamar, Antik Mandal, Abuzar Haideri", { x: MX + 0.1, y: 6.7, w: 10, h: 0.3, fontSize: 12, color: "B9B6CF" });
  s.addNotes("Open with the headline: a compact CNN beats every classical model from Phase 1, with a statistically definitive margin, while being 60 times smaller and 8 times faster. Then promise the second half: how we turned a good model into a reliable final model.");

  // ============================================================ 2. JOURNEY
  s = content("The story in one slide", "From handcrafted features to learned features", "Introduction");
  const steps = [
    [I.db, "Phase 1", "Classical ML + HOG features", "Best: SVM-RBF 91.30%", HEX.grey],
    [I.warn, "The ceiling", "74% of errors in 4 upper-body classes", "Shirt recall only 75%", HEX.orange],
    [I.brain, "Phase 2", "CNNs learn features from raw pixels", "Best: CNN3Block-GAP 93.09%", HEX.violet],
    [I.rocket, "Phase 2 final", "Ensemble + calibration + abstain", `Final: ${F.acc}`, HEX.aqua],
  ];
  steps.forEach(([ic, h, b, k, col], i) => {
    const x = MX + i * 3.08, y = 2.0;
    card(s, x, y, 2.78, 3.6, HEX.white, "step" + i);
    s.addShape(pres.shapes.OVAL, { x: x + 0.3, y: y + 0.35, w: 0.8, h: 0.8, fill: { color: col }, line: { color: col } });
    s.addImage({ data: ic, x: x + 0.5, y: y + 0.55, w: 0.4, h: 0.4 });
    txt(s, h, { x: x + 0.3, y: y + 1.35, w: 2.3, h: 0.45, fontSize: 20, bold: true });
    txt(s, b, { x: x + 0.3, y: y + 1.85, w: 2.25, h: 0.9, fontSize: 15, color: HEX.muted });
    txt(s, k, { x: x + 0.3, y: y + 2.8, w: 2.3, h: 0.6, fontSize: 14, bold: true, color: col === HEX.grey ? HEX.muted : col });
    if (i < 3) s.addImage({ data: I.arrowV, x: x + 2.83, y: y + 1.65, w: 0.22, h: 0.22 });
  });
  takeaway(s, "Phase 2 broke the 91.3% ceiling, explained why, and finished with a model whose confidence can be trusted.");
  s.addNotes("Walk left to right. Phase 1 established the ceiling and showed where errors concentrate. Phase 2 replaces handcrafted HOG with learned convolutional filters. The final model is the last part of the talk.");

  // ============================================================ 3. PHASE 1 RECAP
  pres.addSection({ title: "Phase 1 recap" });
  s = content("Phase 1 recap", "Classical ML hit a ceiling at 91.3%", "Phase 1 recap");
  const fam = ["LogReg (HOG+Stats)", "kNN (HOG+Stats)", "RandomForest (HOG+Stats)", "SVM-RBF (HOG)"];
  const famRows = fam.map((l) => D.leaderboard.find((r) => r.label === l));
  s.addChart(pres.charts.BAR, [{ name: "Best test accuracy", labels: ["Logistic Reg.", "k-NN", "Random Forest", "SVM-RBF"], values: famRows.map((r) => r2(r.acc)) }],
    chartOpts({ x: MX, y: 1.6, w: 7.4, h: 4.55, barDir: "col", chartColors: [HEX.grey, HEX.grey, HEX.grey, HEX.orange],
      valAxisMinVal: 84, valAxisMaxVal: 93, valAxisMajorUnit: 1, valAxisLabelFormatCode: '0"%"', showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: '0.0"%"',
      valAxisTitle: "Test accuracy (%) — best feature set per model", catAxisTitle: "Classical model (Phase 1)", showLegend: false, barGapWidthPct: 60 }));
  const p1c = [[I.target, "91.30%", "best classical accuracy (SVM-RBF on HOG)"], [I.shirt, "74%", "of the SVM's errors are T-shirt, Pullover, Coat or Shirt"], [I.warn, "75.0%", "Shirt recall — 1 in 4 shirts mislabelled"]];
  p1c.forEach(([ic, n, l], i) => {
    const y = 1.65 + i * 1.5;
    card(s, 8.4, y, 4.33, 1.3, HEX.lav);
    iconDot(s, ic, 8.62, y + 0.33, 0.62, i === 0 ? HEX.orange : HEX.violet);
    txt(s, n, { x: 9.45, y: y + 0.17, w: 3.1, h: 0.55, fontSize: 28, bold: true, color: i === 0 ? HEX.orange : HEX.violet });
    txt(s, l, { x: 9.45, y: y + 0.7, w: 3.2, h: 0.55, fontSize: 13, color: HEX.muted });
  });
  takeaway(s, "Handcrafted HOG features cannot see the subtle collar and hem details that separate a Shirt from a T-shirt.");
  s.addNotes("Each bar is the best result for one classical model family on the 10,000 test images. Every family stays below 91.4%. The cards on the right show why: errors are concentrated in four visually similar upper-body classes.");

  // ============================================================ 4. SETUP (optional detail slide, off in the short deck)
  pres.addSection({ title: "Phase 2 approach" });
  if (FULL) {
  s = content("Fair comparison", "Same data, same rules as Phase 1", "Phase 2 approach");
  txt(s, "70,000 images split once (seed 42) — identical to Phase 1", { x: MX, y: 1.65, w: 12, h: 0.4, fontSize: 16, bold: true });
  const split = [["Train", "48,000", 48, HEX.violet], ["Validation", "12,000", 12, "9C8FEA"], ["Test (used once)", "10,000", 10, HEX.orange]];
  let sx = MX; const tot = 70, bw = W - 2 * MX;
  split.forEach(([n, c, v, col]) => {
    const w = (bw * v) / tot;
    s.addShape(pres.shapes.RECTANGLE, { x: sx, y: 2.15, w: w - 0.04, h: 0.85, fill: { color: col }, line: { color: col } });
    txt(s, [{ text: n + "  ", options: { bold: true } }, { text: c + " images" }], { x: sx + 0.15, y: 2.15, w: w - 0.3, h: 0.85, fontSize: 15, color: HEX.white, valign: "middle" });
    sx += w;
  });
  const proto = [
    [I.layers, "Input", "Raw 28×28 grayscale pixels — no handcrafted features"],
    [I.cogs, "Training", "AdamW, cosine learning-rate schedule, batch 128, 15–20 epochs"],
    [I.check, "Model selection", "Early stopping on validation macro-F1 — test set never touched"],
    [I.redo, "Augmentation", "Random 2-px shift (crop) + left-right flip, when switched on"],
  ];
  proto.forEach(([ic, h, b], i) => {
    const x = MX + (i % 2) * 6.17, y = 3.45 + Math.floor(i / 2) * 1.3;
    iconDot(s, ic, x, y + 0.05, 0.62);
    txt(s, h, { x: x + 0.85, y, w: 5.1, h: 0.4, fontSize: 17, bold: true });
    txt(s, b, { x: x + 0.85, y: y + 0.42, w: 5.1, h: 0.6, fontSize: 15, color: HEX.muted });
  });
  takeaway(s, "Any accuracy difference between Phase 1 and Phase 2 comes from the model, not from a different data split.");
  s.addNotes("Stress fairness: exactly the same split and seed as Phase 1, test images used once at the very end, model choices made on validation only.");
  }

  // ============================================================ 5. MODELS
  s = content("Architectures", "Four CNNs in a 2×2 experiment", "Phase 2 approach");
  txt(s, "Our best network — CNN3Block-GAP", { x: MX, y: 1.6, w: 8, h: 0.4, fontSize: 16, bold: true });
  const arch = [["Input", "1×28×28"], ["Block 1", "32 filters\n14×14"], ["Block 2", "64 filters\n7×7"], ["Block 3", "128 filters\n7×7"], ["Global avg\npool", "128 numbers"], ["Output", "10 classes"]];
  arch.forEach(([a, b], i) => {
    const x = MX + i * 2.05, col = i === 4 ? HEX.orange : (i === 0 || i === 5 ? HEX.ink2 : HEX.violet);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 2.1, w: 1.75, h: 1.3, rectRadius: 0.12, fill: { color: col }, line: { color: col } });
    txt(s, a, { x, y: 2.17, w: 1.75, h: 0.6, fontSize: 15, bold: true, color: HEX.white, align: "center", valign: "middle" });
    txt(s, b, { x, y: 2.75, w: 1.75, h: 0.55, fontSize: 12, color: "E9E6FF", align: "center", valign: "middle" });
    if (i < 5) s.addImage({ data: I.arrowV, x: x + 1.8, y: 2.65, w: 0.2, h: 0.2 });
  });
  txt(s, "Each block = two 3×3 convolutions + batch-norm + ReLU. Global average pooling replaces the big flatten layer → fewer weights, less overfitting.", { x: MX, y: 3.5, w: 12.1, h: 0.4, fontSize: 13, color: HEX.muted, italic: true });
  // 2x2 grid
  const gx = MX + 2.6, gy = 4.05, cw = 4.5, ch = 0.8;
  txt(s, "No augmentation", { x: gx, y: gy - 0.05, w: cw, h: 0.35, fontSize: 14, bold: true, align: "center", color: HEX.muted });
  txt(s, "Crop + flip augmentation", { x: gx + cw + 0.15, y: gy - 0.05, w: cw, h: 0.35, fontSize: 14, bold: true, align: "center", color: HEX.muted });
  const grid = [["CNN2Block\n(flatten, 422k params)", "CNN2Block", "CNN2Block-Aug"], ["CNN3Block-GAP\n(305k params)", "CNN3Block-GAP-NoAug  (new run)", "CNN3Block-GAP  ★ best"]];
  grid.forEach(([rl, a, b], r) => {
    const y = gy + 0.38 + r * (ch + 0.12);
    txt(s, rl, { x: MX, y, w: 2.45, h: ch, fontSize: 14, bold: true, valign: "middle" });
    [a, b].forEach((t, c) => {
      const isBest = r === 1 && c === 1;
      card(s, gx + c * (cw + 0.15), y, cw, ch, isBest ? HEX.violet : HEX.lav);
      txt(s, t, { x: gx + c * (cw + 0.15), y, w: cw, h: ch, fontSize: 16, bold: true, align: "center", valign: "middle", color: isBest ? HEX.white : HEX.ink });
    });
  });
  takeaway(s, "Same 48k / 12k / 10k split as Phase 1 · every choice made on validation · test set used once.");
  s.addNotes("The top row is the network architecture. The grid below is the experiment: two architectures times augmentation on or off. The bottom-left cell is new in this revision - without it we could not tell whether depth or augmentation caused the improvement.");

  // ============================================================ 6. LEADERBOARD (optional detail slide, off in the short deck)
  pres.addSection({ title: "Results" });
  if (FULL) {
  s = content("Results · leaderboard", "Every CNN matches or beats the best classical model", "Results");
  const lbl = ["LogReg (Pixels)", "kNN (HOG+Stats)", "RandomForest (HOG+Stats)", "LogReg (HOG+Stats)", "SVM-RBF (HOG)", "SVM-RBF (HOG+Stats)", "CNN3Block GAP NoAug", "CNN2Block Aug", "CNN2Block", "CNN3Block GAP"];
  const lrows = lbl.map((l) => D.leaderboard.find((r) => r.label === l));
  const disp = lrows.map((r) => r.label.replace("CNN3Block GAP NoAug", "CNN3Block-GAP (no aug)").replace("CNN3Block GAP", "CNN3Block-GAP").replace("CNN2Block Aug", "CNN2Block-Aug"));
  const isCnn = (r) => r.phase === 2, isSvm = (r) => r.label === "SVM-RBF (HOG)";
  s.addChart(pres.charts.BAR, [
    { name: "Phase 2 — CNN", labels: disp, values: lrows.map((r) => (isCnn(r) ? r2(r.acc) : 0)) },
    { name: "Phase 1 best — SVM-RBF (HOG)", labels: disp, values: lrows.map((r) => (isSvm(r) ? r2(r.acc) : 0)) },
    { name: "Other Phase 1 models", labels: disp, values: lrows.map((r) => (!isCnn(r) && !isSvm(r) ? r2(r.acc) : 0)) },
  ], chartOpts({ x: MX, y: 1.55, w: 8.3, h: 4.7, barDir: "bar", barGrouping: "stacked", chartColors: [HEX.violet, HEX.orange, HEX.grey],
    valAxisMinVal: 80, valAxisMaxVal: 94, valAxisMajorUnit: 2, valAxisLabelFormatCode: '0"%"', showValue: true, dataLabelPosition: "inEnd", dataLabelFormatCode: '0.00"%";;;',
    dataLabelColor: HEX.white, dataLabelFontBold: true, barGapWidthPct: 45, catAxisLabelFontSize: 12,
    valAxisTitle: "Test accuracy on 10,000 unseen images (%)", catAxisTitle: "Model (feature set)", showLegend: true, legendPos: "b" }));
  const lbCards = [["+1.79 pts", "over SVM-RBF (91.30% → 93.09%)", HEX.violet], ["92.57–93.59%", "95% confidence interval of the best CNN — fully above the SVM's 90.76–91.85%", HEX.violet], ["≈ tie", "Augmented CNN2Block and un-augmented GAP net are inside the SVM's interval", HEX.orange]];
  lbCards.forEach(([n, l, c], i) => {
    const y = 1.6 + i * 1.55;
    card(s, 9.2, y, 3.53, 1.38, HEX.lav);
    txt(s, n, { x: 9.42, y: y + 0.15, w: 3.2, h: 0.5, fontSize: 24, bold: true, color: c });
    txt(s, l, { x: 9.42, y: y + 0.65, w: 3.15, h: 0.7, fontSize: 13, color: HEX.muted });
  });
  takeaway(s, "Only the deep GAP network trained with augmentation is clearly better than Phase 1.");
  s.addNotes("Read the x-axis: test accuracy on the 10,000 test images. Violet bars are CNNs, orange is the Phase 1 champion. Confidence intervals come from 2,000 bootstrap resamples of the test set.");
  }

  // ============================================================ 7. VALIDATION CURVES
  s = content("Results · training", "The GAP network was still improving when training stopped", "Results");
  const hist = D.histories, maxEp = 20, eps = Array.from({ length: maxEp }, (_, i) => String(i + 1));
  const ser = (m, nm) => ({ name: nm, labels: eps, values: eps.map((_, i) => (i < hist[m].val_acc.length ? r2(hist[m].val_acc[i] * 100) : null)) });
  s.addChart(pres.charts.LINE, [ser("CNN3Block_GAP", "CNN3Block-GAP"), ser("CNN2Block", "CNN2Block"), ser("CNN2Block_Aug", "CNN2Block-Aug"), ser("CNN3Block_GAP_NoAug", "CNN3Block-GAP (no aug)")],
    chartOpts({ x: MX, y: 1.55, w: 8.4, h: 4.7, chartColors: [HEX.violet, HEX.grey, HEX.aqua, HEX.orange], lineSize: 2.5, lineDataSymbol: "circle", lineDataSymbolSize: 6,
      valAxisMinVal: 84, valAxisMaxVal: 95, valAxisMajorUnit: 2, valAxisLabelFormatCode: '0"%"', displayBlanksAs: "gap",
      valAxisTitle: "Validation accuracy on 12,000 held-out images (%)", catAxisTitle: "Training epoch (1 epoch = one pass over 48,000 images)", showLegend: true, legendPos: "b" }));
  const vc = [[I.chartV, "Still climbing", "CNN3Block-GAP's best epoch was its last (20). More epochs will likely help."], [I.warnV, "Under-trained", "CNN2Block-Aug scores higher on validation than on training — it needed more epochs."], [I.searchV, "Overfits without augmentation", "The GAP net without augmentation peaked at epoch 5 and was stopped at 10."]];
  vc.forEach(([ic, h, b], i) => {
    const y = 1.6 + i * 1.55;
    card(s, 9.2, y, 3.53, 1.38, HEX.lav);
    s.addImage({ data: ic, x: 9.4, y: y + 0.2, w: 0.3, h: 0.3 });
    txt(s, h, { x: 9.82, y: y + 0.15, w: 2.8, h: 0.4, fontSize: 15, bold: true });
    txt(s, b, { x: 9.42, y: y + 0.6, w: 3.15, h: 0.75, fontSize: 13, color: HEX.muted });
  });
  takeaway(s, "Training budgets — not model limits — ended the best runs. So the final models get 40 epochs.");
  s.addNotes("x-axis: epochs; y-axis: accuracy on the 12,000 validation images. Violet keeps rising until the end. The orange dashed-style run (no augmentation) rises fast and then stalls - overfitting.");

  // ============================================================ 8. OVERFITTING (optional detail slide, off in the short deck)
  if (FULL) {
  s = content("Results · overfitting check", "Train vs validation loss: who memorises?", "Results");
  [["CNN2Block", "CNN2Block — gap opens after epoch 6"], ["CNN3Block_GAP", "CNN3Block-GAP — curves stay together"]].forEach(([m, t], i) => {
    const h = hist[m], e = h.train_loss.map((_, k) => String(k + 1));
    const x = MX + i * 6.2;
    txt(s, t, { x, y: 1.6, w: 5.9, h: 0.4, fontSize: 16, bold: true, color: i === 0 ? HEX.orange : HEX.violet });
    s.addChart(pres.charts.LINE, [
      { name: "Training loss", labels: e, values: h.train_loss.map(r2) },
      { name: "Validation loss", labels: e, values: h.val_loss.map(r2) },
    ], chartOpts({ x, y: 2.0, w: 5.9, h: 4.2, chartColors: [HEX.grey, HEX.violet], lineSize: 2.5, lineDataSymbol: "none",
      valAxisMinVal: 0, valAxisMaxVal: 0.7, valAxisMajorUnit: 0.1, valAxisLabelFormatCode: "0.0",
      valAxisTitle: "Cross-entropy loss (lower = better)", catAxisTitle: "Training epoch", showLegend: true, legendPos: "t" }));
  });
  takeaway(s, "When validation loss stays flat while training loss keeps falling, the model is memorising — GAP + augmentation prevents this.");
  s.addNotes("Loss measures how wrong the model's probabilities are; lower is better. In CNN2Block the grey training loss keeps dropping to 0.11 while validation stays at 0.20 - memorisation. In CNN3Block-GAP the two curves end close together (0.14 vs 0.16).");
  }

  // ============================================================ 9. ABLATION
  s = content("Results · what caused the gain?", "Depth + GAP only pays off together with augmentation", "Results");
  const ab = D.ablation.acc;
  s.addChart(pres.charts.BAR, [
    { name: "No augmentation", labels: ["CNN2Block (flatten)", "CNN3Block-GAP"], values: [r2(ab[0][0]), r2(ab[1][0])] },
    { name: "Crop + flip augmentation", labels: ["CNN2Block (flatten)", "CNN3Block-GAP"], values: [r2(ab[0][1]), r2(ab[1][1])] },
  ], chartOpts({ x: MX, y: 1.55, w: 7.2, h: 4.7, barDir: "col", chartColors: [HEX.grey, HEX.violet], barGapWidthPct: 70,
    valAxisMinVal: 90, valAxisMaxVal: 94, valAxisMajorUnit: 1, valAxisLabelFormatCode: '0"%"', showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: '0.00"%"',
    valAxisTitle: "Test accuracy (%)", catAxisTitle: "Architecture", showLegend: true, legendPos: "t" }));
  card(s, 8.15, 1.6, 4.58, 4.55, HEX.lav);
  txt(s, "How to read the 2×2", { x: 8.45, y: 1.85, w: 4, h: 0.4, fontSize: 17, bold: true, color: HEX.violet });
  txt(s, [
    { text: "Augmentation on the shallow net: ", options: { bold: true } }, { text: "92.16% → 91.41% (under-trained in 18 epochs)", options: { breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 6 } },
    { text: "Depth + GAP alone: ", options: { bold: true } }, { text: "91.38% — overfits after 5 epochs", options: { breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 6 } },
    { text: "Both together: ", options: { bold: true, color: HEX.violet } }, { text: "93.09% — augmentation adds +1.71 pts to the GAP net", options: { breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 6 } },
    { text: "McNemar between the two GAP runs: p = 1.0×10⁻¹²", options: { color: HEX.muted, italic: true } },
  ], { x: 8.45, y: 2.35, w: 4.05, h: 3.6, fontSize: 15 });
  takeaway(s, "Neither ingredient works alone: the deeper network needs augmentation to use its extra capacity.");
  s.addNotes("This is the new controlled experiment. Before it, depth, GAP and augmentation changed at the same time, so the gain could not be explained. The final models were then repeated with 3 seeds each.");

  // ============================================================ 10. MCNEMAR (optional detail slide, off in the short deck)
  if (FULL) {
  s = content("Results · is it luck?", "McNemar test: the best CNN wins 445 to 266", "Results");
  const mc = D.mcnemar, mcl = mc.map((r) => r.model_2.replace("CNN3Block_GAP", "CNN3Block-GAP").replace("CNN2Block_Aug", "CNN2Block-Aug"));
  s.addChart(pres.charts.BAR, [
    { name: "Only SVM-RBF (HOG) correct", labels: mcl, values: mc.map((r) => r.svm_only_correct) },
    { name: "Only the CNN correct", labels: mcl, values: mc.map((r) => r.cnn_only_correct) },
  ], chartOpts({ x: MX, y: 1.55, w: 7.6, h: 4.7, barDir: "col", chartColors: [HEX.orange, HEX.violet], barGapWidthPct: 60,
    valAxisMinVal: 0, valAxisMaxVal: 500, valAxisMajorUnit: 100, showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0",
    valAxisTitle: "Test images where exactly one model is right", catAxisTitle: "CNN compared against SVM-RBF (HOG)", showLegend: true, legendPos: "t" }));
  const pv = [["CNN2Block", "p = 0.0014", "significant at 1%, not at 0.1%", HEX.muted], ["CNN2Block-Aug", "p = 0.71", "a tie", HEX.muted], ["CNN3Block-GAP", "p = 2.5×10⁻¹¹", "statistically definitive", HEX.violet]];
  txt(s, "If both models were equally good, each pair of bars would be about the same height.", { x: 8.55, y: 1.6, w: 4.2, h: 0.85, fontSize: 14, color: HEX.muted, italic: true });
  pv.forEach(([m, p, l, c], i) => {
    const y = 2.55 + i * 1.2;
    card(s, 8.55, y, 4.18, 1.05, i === 2 ? HEX.violet : HEX.lav);
    txt(s, m, { x: 8.78, y: y + 0.12, w: 3.8, h: 0.35, fontSize: 14, bold: true, color: i === 2 ? HEX.white : HEX.ink });
    txt(s, [{ text: p + "  ", options: { bold: true } }, { text: l }], { x: 8.78, y: y + 0.5, w: 3.8, h: 0.4, fontSize: 14, color: i === 2 ? "E9E6FF" : HEX.muted });
  });
  takeaway(s, "A p-value of 2.5×10⁻¹¹ means this gap would essentially never happen by chance.");
  s.addNotes("McNemar ignores images both models get right or both get wrong, and only counts disagreements. 445 versus 266 is a very lopsided split.");
  }

  // ============================================================ 11. PER-CLASS
  pres.addSection({ title: "Error analysis" });
  const rc = D.recall, fr = rc.classes.map((c) => r1(FS.per_class_recall[c] * 100));
  const ord = fr.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]).map((x) => x[1]);
  const gain = (c) => fr[rc.classes.indexOf(c)] - rc.svm[rc.classes.indexOf(c)];
  s = content("Error analysis · per class", `Every class improved — Shirt gained the most (+${gain("Shirt").toFixed(1)} pts)`, "Error analysis");
  s.addChart(pres.charts.BAR, [
    { name: "SVM-RBF (HOG) — Phase 1", labels: ord.map((i) => rc.classes[i]), values: ord.map((i) => rc.svm[i]) },
    { name: "First CNN3Block-GAP", labels: ord.map((i) => rc.classes[i]), values: ord.map((i) => rc.cnn[i]) },
    { name: "Final Phase 2 model", labels: ord.map((i) => rc.classes[i]), values: ord.map((i) => fr[i]) },
  ], chartOpts({ x: MX, y: 1.55, w: 8.2, h: 4.7, barDir: "bar", chartColors: [HEX.orange, HEX.grey, HEX.violet], barGapWidthPct: 35,
    valAxisMinVal: 70, valAxisMaxVal: 100, valAxisMajorUnit: 5, valAxisLabelFormatCode: '0"%"', showValue: false,
    valAxisTitle: "Recall: % of the class's 1,000 test images labelled correctly", catAxisTitle: "Clothing class (hardest at the bottom)", showLegend: true, legendPos: "t" }));
  [["Shirt", "75.0%", HEX.orange], ["Pullover", "85.6%", HEX.violet], ["Coat", "87.5%", HEX.violet]].forEach(([c, was, col], i) => {
    const y = 1.6 + i * 1.55, now = fr[rc.classes.indexOf(c)];
    card(s, 9.1, y, 3.63, 1.38, HEX.lav);
    txt(s, c, { x: 9.32, y: y + 0.15, w: 1.6, h: 0.45, fontSize: 17, bold: true });
    txt(s, `+${gain(c).toFixed(1)} pts`, { x: 10.6, y: y + 0.12, w: 1.95, h: 0.5, fontSize: 22, bold: true, color: col, align: "right" });
    txt(s, `${was} → ${now.toFixed(1)}%${c === "Shirt" ? "  — still the hardest" : ""}`, { x: 9.32, y: y + 0.72, w: 3.3, h: 0.5, fontSize: 14, color: HEX.muted });
  });
  takeaway(s, "The residual ensemble finally closes most of the Shirt gap that handcrafted features and the first CNN left open.");
  s.addNotes("Classes are sorted from hardest (bottom) to easiest (top). Recall is the share of each class's 1,000 test images that the model labels correctly. Gains are versus the Phase 1 SVM.");

  // ============================================================ 12. CONFUSION (optional detail slide, off in the short deck)
  if (FULL) {
  s = content("Error analysis · mix-ups", "Upper-body confusion: who gets mistaken for whom?", "Error analysis");
  const names = D.upper_names;
  const shade = (v) => {
    const t = Math.min(v / 100, 1), a = [0xf4, 0xf2, 0xfd], b = [0x3e, 0x2d, 0xa8];
    return a.map((c, k) => Math.round(c + (b[k] - c) * t).toString(16).padStart(2, "0")).join("").toUpperCase();
  };
  Object.entries(D.upper_conf).forEach(([nm, M], ti) => {
    const x = MX + ti * 6.2;
    txt(s, nm.replace("CNN3Block GAP", "CNN3Block-GAP (Phase 2)").replace("SVM-RBF (HOG)", "SVM-RBF (HOG) (Phase 1)"), { x, y: 1.55, w: 5.9, h: 0.4, fontSize: 16, bold: true, color: ti ? HEX.violet : HEX.orange });
    const hdr = [{ text: "True ↓ / Predicted →", options: { bold: true, fontSize: 11, color: HEX.muted, fill: { color: HEX.white } } }]
      .concat(names.map((n) => ({ text: n, options: { bold: true, fontSize: 12, color: HEX.ink, align: "center", fill: { color: HEX.white } } })));
    const rows = [hdr].concat(M.map((row, i) => [{ text: names[i], options: { bold: true, fontSize: 12, color: HEX.ink, fill: { color: HEX.white } } }]
      .concat(row.map((v, j) => ({ text: v.toFixed(1) + "%", options: { align: "center", valign: "middle", fontSize: 15, bold: i === j,
        color: v > 50 ? HEX.white : HEX.ink, fill: { color: shade(v) } } })))));
    s.addTable(rows, { x, y: 2.0, w: 5.9, colW: [1.5, 1.1, 1.1, 1.1, 1.1], rowH: [0.5, 0.8, 0.8, 0.8, 0.8], border: { type: "solid", pt: 2, color: HEX.white }, fontFace: "Calibri", valign: "middle" });
  });
  txt(s, "Each row is one true class and adds up to 100%. Dark diagonal = correct. Example: the CNN calls 10.4% of real Shirts a T-shirt.", { x: MX, y: 5.75, w: 12.1, h: 0.4, fontSize: 13, color: HEX.muted, italic: true });
  takeaway(s, "Pullover↔Coat confusion roughly halved, but Shirt→T-shirt got slightly worse (9.8% → 10.4%).");
  s.addNotes("Top CNN confusions overall: Shirt to T-shirt 104 images, T-shirt to Shirt 77, Shirt to Coat 64. These three pairs drove the Shirt decision rule in the final model.");
  }

  // ============================================================ 13. ERROR BREAKDOWN (optional detail slide, off in the short deck)
  if (FULL) {
  s = content("Error analysis · where errors live", "Errors fell 55% — but the hard core got more concentrated", "Error analysis");
  const er = D.errors, erl = ["LogReg (Pixels)\nPhase 1 start", "SVM-RBF (HOG)\nPhase 1 best", "CNN2Block\nPhase 2", "CNN3Block-GAP\nPhase 2 best"];
  s.addChart(pres.charts.BAR, [
    { name: "Upper-body classes (T-shirt, Pullover, Coat, Shirt)", labels: erl, values: er.map((r) => r.upper) },
    { name: "Other 6 classes", labels: erl, values: er.map((r) => r.other) },
  ], chartOpts({ x: MX, y: 1.55, w: 8.2, h: 4.7, barDir: "col", barGrouping: "stacked", chartColors: [HEX.violet, HEX.grey], barGapWidthPct: 55,
    valAxisMinVal: 0, valAxisMaxVal: 1600, valAxisMajorUnit: 400, showValue: true, dataLabelPosition: "ctr", dataLabelFormatCode: "0", dataLabelColor: HEX.white, dataLabelFontBold: true,
    valAxisTitle: "Misclassified test images (out of 10,000)", catAxisTitle: "Model", showLegend: true, legendPos: "t", catAxisLabelFontSize: 11 }));
  const eb = [["1,544 → 691", "total test errors, first Phase 1 model to best CNN"], ["74% → 76%", "share of errors in upper-body classes (SVM → CNN)"], ["Shirt", "is the true class of 34% of all remaining CNN errors (237 of 691)"]];
  eb.forEach(([n, l], i) => {
    const y = 1.6 + i * 1.55;
    card(s, 9.1, y, 3.63, 1.38, HEX.lav);
    txt(s, n, { x: 9.32, y: y + 0.15, w: 3.3, h: 0.5, fontSize: 24, bold: true, color: i === 1 ? HEX.orange : HEX.violet });
    txt(s, l, { x: 9.32, y: y + 0.68, w: 3.3, h: 0.65, fontSize: 13, color: HEX.muted });
  });
  takeaway(s, "Easy classes improved faster than hard ones — the remaining problem is almost entirely upper-body garments.");
  s.addNotes("Violet = errors on images whose true class is an upper-body garment. Even as total errors shrink, the violet share grows. Shirt: 237 of the CNN's 691 errors.");
  }

  // ============================================================ 14. EFFICIENCY (optional detail slide, off in the short deck)
  pres.addSection({ title: "Cost & explainability" });
  if (FULL) {
  s = content("Cost", "More accurate and cheaper to run", "Cost & explainability");
  const eff = [["SVM-RBF (HOG)", svm], ["CNN2Block", D.leaderboard.find((r) => r.label === "CNN2Block")], ["CNN3Block-GAP", best]];
  s.addChart(pres.charts.BAR, [{ name: "Stored numbers (thousands)", labels: eff.map((e) => e[0]), values: eff.map((e) => Math.round(e[1].params / 1000)) }],
    chartOpts({ x: MX, y: 1.55, w: 5.9, h: 3.6, barDir: "bar", chartColors: [HEX.orange, HEX.grey, HEX.violet], barGapWidthPct: 45,
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: '#,##0"k"', valAxisMinVal: 0, valAxisMaxVal: 21000, valAxisMajorUnit: 5000, valAxisLabelFormatCode: '#,##0"k"',
      valAxisTitle: "Model size: numbers stored (thousands)", catAxisTitle: "Model", showLegend: false }));
  s.addChart(pres.charts.BAR, [{ name: "ms per image", labels: eff.map((e) => e[0]), values: eff.map((e) => r2(e[1].ms)) }],
    chartOpts({ x: MX + 6.2, y: 1.55, w: 5.9, h: 3.6, barDir: "bar", chartColors: [HEX.orange, HEX.grey, HEX.violet], barGapWidthPct: 45,
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: '0.00" ms"', valAxisMinVal: 0, valAxisMaxVal: 6, valAxisMajorUnit: 1,
      valAxisTitle: "CPU time to classify one image (milliseconds)", catAxisTitle: "Model", showLegend: false }));
  [[I.compress, "60× smaller", "305k weights vs 18.1M stored SVM numbers"], [I.bolt, "8× faster", "0.68 ms vs 5.36 ms per image on one CPU core"], [I.target, "+1.79 pts", "more accurate at the same time"]].forEach(([ic, n, l], i) => {
    const x = MX + i * 4.1;
    iconDot(s, ic, x, 5.3, 0.6);
    txt(s, n, { x: x + 0.78, y: 5.25, w: 3.2, h: 0.4, fontSize: 18, bold: true, color: HEX.violet });
    txt(s, l, { x: x + 0.78, y: 5.65, w: 3.2, h: 0.5, fontSize: 13, color: HEX.muted });
  });
  takeaway(s, "Logistic regression is still fastest (0.001 ms) — the CNN is the best accuracy-for-cost trade-off, not the fastest model.");
  s.addNotes("Left: how many numbers each model must store - an SVM keeps thousands of support vectors. Right: time for one image on a single CPU thread.");
  }

  // ============================================================ 15. GRAD-CAM
  s = content("Explainability", "Grad-CAM: where does the CNN look?", "Cost & explainability");
  s.addImage({ path: path.join(ROOT, "report/figures/cnn_02_gradcam_gallery.png"), x: MX, y: 1.5, w: 5.35, h: 5.35, altText: "Grad-CAM heat-maps for T-shirt, Pullover, Coat and Shirt test images" });
  txt(s, "Red = image regions that pushed the network toward its answer.", { x: 6.4, y: 1.6, w: 6.3, h: 0.4, fontSize: 14, color: HEX.muted, italic: true });
  const gc = [[I.check, "Ignores the background", "Heat sits on the garment, never on the black border.", HEX.violet],
    [I.eye, "Pullover: cuffs & waistband", "Ribbed edges are a learned cue HOG could not isolate.", HEX.violet],
    [I.warn, "Some maps are diffuse", "Checked Shirt and printed T-shirt light up everywhere — consistent with Shirt still being hard.", HEX.orange],
    [I.search, "Needs measurement", "8 hand-picked images are evidence, not proof — listed as a limitation.", HEX.orange]];
  gc.forEach(([ic, h, b, col], i) => {
    const y = 2.15 + i * 1.08;
    iconDot(s, ic, 6.4, y, 0.55, col);
    txt(s, h, { x: 7.15, y: y - 0.04, w: 5.6, h: 0.38, fontSize: 16, bold: true });
    txt(s, b, { x: 7.15, y: y + 0.33, w: 5.6, h: 0.65, fontSize: 13, color: HEX.muted });
  });
  s.addNotes("Be honest here: the earlier draft claimed activations sit squarely on collars. The maps show something more mixed. That honesty makes the rest of the results more credible.");

  // ============================================================ 16. FROM GOOD TO RELIABLE
  pres.addSection({ title: "Final model" });
  s = content("Final model", "From a good model to a reliable one", "Final model");
  const fin = [
    [I.redo, "Longer + repeated", "40-epoch budget and 3 random seeds per model", "results stable across seeds"],
    [I.layers, "Residual network", "ResNetSmall with Random Erasing and label smoothing", "deeper without vanishing gradients"],
    [I.cubes, "Ensemble + TTA", "Average several nets and a mirrored copy of each image", "fewer random mistakes"],
    [I.shield, "Trustworthy output", "Temperature-scaled confidence, abstain option, Shirt-aware rule", "know when to ask a human"],
  ];
  fin.forEach(([ic, h, b, k], i) => {
    const x = MX + i * 3.08;
    card(s, x, 1.7, 2.82, 3.9, HEX.white);
    iconDot(s, ic, x + 0.28, 1.98, 0.7, i === 3 ? HEX.aqua : HEX.violet);
    txt(s, h, { x: x + 0.28, y: 2.9, w: 2.35, h: 0.45, fontSize: 19, bold: true });
    txt(s, b, { x: x + 0.28, y: 3.42, w: 2.35, h: 1.2, fontSize: 15, color: HEX.muted });
    txt(s, k, { x: x + 0.28, y: 4.75, w: 2.35, h: 0.6, fontSize: 13, bold: true, color: i === 3 ? HEX.aqua : HEX.violet });
  });
  takeaway(s, "Every choice (which ensemble, temperature, Shirt rule, abstain threshold) was made on validation data only.");
  s.addNotes("These four upgrades address the weaknesses found in the first Phase 2 runs: under-training, a single seed, the Shirt problem and uncalibrated confidence.");

  // ============================================================ 17. SEEDS
  s = content("Final model · robustness", "Results hold across three random seeds", "Final model");
  const fams = [["CNN3Block_GAP_E40", "CNN3Block-GAP (40 epochs)", HEX.aqua], ["ResNetSmall", "ResNetSmall", HEX.violet]].filter(([f]) => PER_RUN.some((r) => r.family === f));
  const seeds = [...new Set(PER_RUN.map((r) => r.seed))].sort();
  s.addChart(pres.charts.BAR, fams.map(([f, nm]) => ({ name: nm, labels: seeds.map((x) => "Seed " + x),
    values: seeds.map((sd) => { const r = PER_RUN.find((q) => q.family === f && q.seed === sd); return r ? r2(r.test_acc * 100) : 0; }) })),
    chartOpts({ x: MX, y: 1.55, w: 7.6, h: 4.7, barDir: "col", chartColors: fams.map((f) => f[2]), barGapWidthPct: 60,
      valAxisMinVal: 91, valAxisMaxVal: 95, valAxisMajorUnit: 1, valAxisLabelFormatCode: '0"%"', showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: '0.00"%"',
      valAxisTitle: "Test accuracy on 10,000 images (%)", catAxisTitle: "Training run (same data, different random start)", showLegend: true, legendPos: "t" }));
  fams.forEach(([f, nm, col], i) => {
    const rs = PER_RUN.filter((r) => r.family === f).map((r) => r.test_acc * 100);
    const m = rs.reduce((a, b) => a + b, 0) / rs.length, sd = Math.sqrt(rs.reduce((a, b) => a + (b - m) ** 2, 0) / Math.max(rs.length - 1, 1));
    const y = 1.6 + i * 1.55;
    card(s, 8.55, y, 4.18, 1.38, HEX.lav);
    txt(s, nm, { x: 8.78, y: y + 0.13, w: 3.8, h: 0.35, fontSize: 14, bold: true });
    txt(s, `${m.toFixed(2)}% ± ${sd.toFixed(2)}`, { x: 8.78, y: y + 0.5, w: 3.8, h: 0.5, fontSize: 24, bold: true, color: col });
    txt(s, "mean ± standard deviation over seeds", { x: 8.78, y: y + 1.0, w: 3.8, h: 0.3, fontSize: 12, color: HEX.muted });
  });
  card(s, 8.55, 4.7, 4.18, 1.45, HEX.white);
  txt(s, "Every run beats the Phase 1 SVM (91.30%) and the spread between seeds is small — the improvement is not luck.", { x: 8.78, y: 4.85, w: 3.8, h: 1.2, fontSize: 14, color: HEX.ink });
  takeaway(s, "With 40 epochs the GAP network itself improves over the first 20-epoch run — it really was under-trained.");
  s.addNotes("Each bar is one complete training run. Different seeds change the random initialisation and the order of training images.");

  // ============================================================ 18. LADDER
  s = content("Final model · step by step", "How each improvement adds accuracy", "Final model");
  const bs = PER_RUN.slice().sort((a, b) => b.val_acc - a.val_acc)[0];
  const lad = [["SVM (Ph. 1)", 91.30, HEX.orange], ["First CNN", 93.09, HEX.grey],
    ["Best single", bs.test_acc * 100, HEX.aqua], ["+ TTA", bs.test_acc_tta * 100, HEX.aqua],
    [`Ensemble ×${FS.members.length}`, FS.test_acc_before_bias * 100, HEX.violet], ["FINAL", F.accN, HEX.violetDark]];
  s.addChart(pres.charts.BAR, [{ name: "Test accuracy", labels: lad.map((l) => l[0]), values: lad.map((l) => r2(l[1])) }],
    chartOpts({ x: MX, y: 1.55, w: 8.6, h: 4.7, barDir: "col", chartColors: lad.map((l) => l[2]), barGapWidthPct: 45,
      valAxisMinVal: 90, valAxisMaxVal: Math.ceil(Math.max(...lad.map((l) => l[1])) + 0.6), valAxisMajorUnit: 1, valAxisLabelFormatCode: '0"%"',
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: '0.00"%"', catAxisLabelFontSize: 11,
      valAxisTitle: "Test accuracy on 10,000 images (%)", catAxisTitle: "Each step keeps the previous ones and adds one improvement (TTA = also classify the mirrored image)", showLegend: false }));
  const lc = [[F.acc, "final test accuracy", HEX.violet], [`${F.err}`, `errors of 10,000 (SVM: 870)`, HEX.violet], [pc(FS.ci[0]) + "–" + pc(FS.ci[1]), "95% confidence interval", HEX.muted]];
  lc.forEach(([n, l, c], i) => {
    const y = 1.6 + i * 1.55;
    card(s, 9.5, y, 3.23, 1.38, HEX.lav);
    txt(s, n, { x: 9.7, y: y + 0.15, w: 2.9, h: 0.55, fontSize: i === 2 ? 19 : 26, bold: true, color: c });
    txt(s, l, { x: 9.7, y: y + 0.78, w: 2.9, h: 0.5, fontSize: 13, color: HEX.muted });
  });
  takeaway(s, `Final model vs Phase 1 SVM: McNemar p = ${FS.mcnemar_vs_svm.p.toExponential(1)}; vs first CNN: p = ${FS.mcnemar_vs_gap.p.toExponential(1)}.`);
  s.addNotes("Read left to right: each bar keeps all previous improvements and adds one more. TTA = test-time augmentation: the model also looks at a mirrored copy of the image and averages both answers.");

  // ============================================================ 19. CALIBRATION
  s = content("Final model · calibration", `Calibrated confidence: error ${pc(FS.ece_before, 1)} → ${pc(FS.ece_after, 1)}`, "Final model");
  const cb = FS.calib_bins_after.filter((b) => b.n >= 30);
  s.addChart(pres.charts.BAR, [
    { name: "Confidence the model claimed", labels: cb.map((b) => `${b.bin} (${b.n.toLocaleString("en")})`), values: cb.map((b) => r1(b.conf * 100)) },
    { name: "How often it was actually right", labels: cb.map((b) => `${b.bin} (${b.n.toLocaleString("en")})`), values: cb.map((b) => r1(b.acc * 100)) },
  ], chartOpts({ x: MX, y: 1.55, w: 8.2, h: 4.7, barDir: "col", chartColors: [HEX.grey, HEX.violet], barGapWidthPct: 50,
    valAxisMinVal: 30, valAxisMaxVal: 100, valAxisMajorUnit: 10, valAxisLabelFormatCode: '0"%"', showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: '0"%"', dataLabelFontSize: 10,
    valAxisTitle: "Percentage (%)", catAxisTitle: "Confidence group after calibration (number of test images)", catAxisLabelFontSize: 10, showLegend: true, legendPos: "t" }));
  [[F.eceB, "calibration error before (ECE)", HEX.orange], [F.eceA, "calibration error after temperature scaling", HEX.violet], [`T = ${FS.temperature.toFixed(2)}`, "temperature fitted on validation data", HEX.muted]].forEach(([n, l, c], i) => {
    const y = 1.6 + i * 1.55;
    card(s, 9.1, y, 3.63, 1.38, HEX.lav);
    txt(s, n, { x: 9.32, y: y + 0.15, w: 3.3, h: 0.55, fontSize: 26, bold: true, color: c });
    txt(s, l, { x: 9.32, y: y + 0.78, w: 3.3, h: 0.5, fontSize: 13, color: HEX.muted });
  });
  takeaway(s, `${pc(FS.calib_bins_after[6].n / 10000, 0)} of images sit in the top group, where claimed and actual agree; the small middle groups stay a bit over-confident.`);
  s.addNotes("ECE, expected calibration error, is the average gap between the confidence the model claims and how often it is actually right. Temperature scaling divides the scores by one number, chosen on validation data; it never changes which class is predicted.");

  // ============================================================ 20. ABSTAIN
  s = content("Final model · knowing when to ask", "Let the model say “not sure” on the hardest images", "Final model");
  const cc = FS.coverage_curve, step = Math.max(1, Math.floor(cc.coverage.length / 60));
  const cx = cc.coverage.filter((_, i) => i % step === 0 && cc.coverage[i] >= 0.4), cy = cc.accuracy.filter((_, i) => i % step === 0 && cc.coverage[i] >= 0.4);
  s.addChart(pres.charts.SCATTER, [{ name: "Coverage", values: cx.map((v) => r1(v * 100)) }, { name: "Accuracy on answered images", values: cy.map((v) => r2(v * 100)) }],
    chartOpts({ x: MX, y: 1.55, w: 8.2, h: 4.7, chartColors: [HEX.violet], lineSize: 3, lineDataSymbol: "none",
      valAxisMinVal: 92, valAxisMaxVal: 100, valAxisMajorUnit: 1, valAxisLabelFormatCode: '0"%"', catAxisLabelFormatCode: '0"%"',
      catAxisMinVal: 40, catAxisMaxVal: 100, catAxisMajorUnit: 10,
      valAxisTitle: "Accuracy on the images it answers (%)", catAxisTitle: "Coverage: % of test images answered, most confident first", showLegend: false }));
  [[F.cov, "of images answered automatically", HEX.violet], [F.kept, "accuracy on those images", HEX.violet],
   [pc(FS.abstain.shirt_share_of_abstained, 0), "of the images sent to a human are Shirts", HEX.orange]].forEach(([n, l, c], i) => {
    const y = 1.6 + i * 1.55;
    card(s, 9.1, y, 3.63, 1.38, HEX.lav);
    txt(s, n, { x: 9.32, y: y + 0.15, w: 3.3, h: 0.55, fontSize: 26, bold: true, color: c });
    txt(s, l, { x: 9.32, y: y + 0.78, w: 3.3, h: 0.5, fontSize: 13, color: HEX.muted });
  });
  takeaway(s, `Threshold chosen on validation for 99% accuracy: below ${(FS.abstain.threshold * 100).toFixed(0)}% confidence, a human checks the image.`);
  s.addNotes("Moving left on the curve, the model answers fewer images but is right more often. In a catalogue-tagging system the low-confidence images would go to a person.");

  // ============================================================ 21. SHIRT RULE (optional detail slide, off in the short deck)
  if (FULL) {
  s = content("Final model · the Shirt problem", "A Shirt-aware decision rule, tuned on validation", "Final model");
  const bl = BIAS.map((b) => b.bias.toFixed(2));
  s.addChart(pres.charts.LINE, [
    { name: "Shirt recall (real shirts caught)", labels: bl, values: BIAS.map((b) => r1(b.test_shirt_recall * 100)) },
    { name: "Shirt precision (“Shirt” answers that are right)", labels: bl, values: BIAS.map((b) => r1(b.test_shirt_precision * 100)) },
    { name: "Overall accuracy", labels: bl, values: BIAS.map((b) => r2(b.test_acc * 100)) },
  ], chartOpts({ x: MX, y: 1.55, w: 8.2, h: 4.7, chartColors: [HEX.violet, HEX.orange, HEX.grey], lineSize: 2.5, lineDataSymbol: "none",
    valAxisMinVal: 65, valAxisMaxVal: 100, valAxisMajorUnit: 5, valAxisLabelFormatCode: '0"%"', catAxisLabelFrequency: 5,
    valAxisTitle: "Percentage (%)", catAxisTitle: "Extra score given to “Shirt” before choosing the answer", showLegend: true, legendPos: "t" }));
  [[`+${FS.shirt_bias.toFixed(2)}`, "bias chosen (best validation macro-F1)", HEX.aqua], [F.shirt, "final Shirt recall (Phase 1: 75.0%)", HEX.violet],
   [pc(FS.per_class_recall["T-shirt/top"], 1), "T-shirt recall, the class it trades against", HEX.muted]].forEach(([n, l, c], i) => {
    const y = 1.6 + i * 1.55;
    card(s, 9.1, y, 3.63, 1.38, HEX.lav);
    txt(s, n, { x: 9.32, y: y + 0.15, w: 3.3, h: 0.55, fontSize: 26, bold: true, color: c });
    txt(s, l, { x: 9.32, y: y + 0.78, w: 3.3, h: 0.5, fontSize: 13, color: HEX.muted });
  });
  takeaway(s, "Moving right catches more real shirts but raises false alarms — the rule picks the balance point on validation data.");
  s.addNotes("This is a decision rule, not a new model: we add a small constant to the Shirt score. Its size was chosen on the validation set.");
  }

  // ============================================================ 22. SCORECARD
  s = content("Final model · scorecard", "Phase 1 vs first CNN vs final Phase 2 model", "Final model");
  const firstGap = D.leaderboard.find((r) => r.label === "CNN3Block GAP");
  const hd = (t) => ({ text: t, options: { bold: true, color: HEX.white, fill: { color: HEX.ink }, fontSize: 15, align: "center" } });
  const cell = (t, o = {}) => ({ text: t, options: Object.assign({ fontSize: 15, color: HEX.ink, align: "center", fill: { color: HEX.white } }, o) });
  const hi = (t) => cell(t, { bold: true, color: HEX.violet, fill: { color: HEX.lav } });
  const rowsT = [
    [hd("Measure"), hd("SVM-RBF (HOG) · Phase 1"), hd("CNN3Block-GAP · first"), hd("Final Phase 2 model")],
    [cell("Test accuracy", { align: "left", bold: true }), cell("91.30%"), cell("93.09%"), hi(F.acc)],
    [cell("Errors (of 10,000)", { align: "left", bold: true }), cell("870"), cell("691"), hi(`${F.err}`)],
    [cell("Shirt recall", { align: "left", bold: true }), cell("75.0%"), cell("76.3%"), hi(F.shirt)],
    [cell("Calibration error (ECE)", { align: "left", bold: true }), cell("n/a"), cell("not measured"), hi(F.eceA)],
    [cell("Can abstain when unsure", { align: "left", bold: true }), cell("no"), cell("no"), hi(`yes · ${F.kept} on ${F.cov}`)],
    [cell("Stored numbers", { align: "left", bold: true }), cell("18.1M"), cell("0.30M"), cell((F.params / 1e6).toFixed(2) + "M")],
    [cell("CPU time per image", { align: "left", bold: true }), cell("5.36 ms"), cell(firstGap.ms.toFixed(2) + " ms"), cell(`${F.ms} ms`)],
  ];
  s.addTable(rowsT, { x: MX, y: 1.6, w: W - 2 * MX, colW: [3.2, 2.97, 2.97, 2.99], rowH: 0.52, border: { type: "solid", pt: 1, color: HEX.line }, fontFace: "Calibri", valign: "middle" });
  takeaway(s, `Most accurate and the only model with calibrated confidence — at ${F.ms} ms per image, still real-time.`);
  s.addNotes("Be upfront about the cost: an ensemble with test-time augmentation runs several networks twice. It is slower than a single CNN but still in the millisecond range.");

  // ============================================================ 23. LIMITATIONS
  s = content("Honest limits", "What this project does not show", "Final model");
  const lim = [[I.shirt, "Shirt is still the hardest class", "Even the final model misses some shirts; the abstain option routes most of them to a human."],
    [I.eye, "Grad-CAM is qualitative", "Heat-maps on hand-picked images are evidence, not a measured explanation."],
    [I.db, "Clean, centred thumbnails", "Real shop photos (backgrounds, angles, colour) were not tested; the demo's upload is only indicative."],
    [I.cogs, "Small hyper-parameter search", "Learning rate, width and augmentation strength were not tuned exhaustively."]];
  lim.forEach(([ic, h, b], i) => {
    const x = MX + (i % 2) * 6.17, y = 1.75 + Math.floor(i / 2) * 2.15;
    card(s, x, y, 5.95, 1.9, HEX.lav);
    iconDot(s, ic, x + 0.28, y + 0.3, 0.6, i === 0 ? HEX.orange : HEX.violet);
    txt(s, h, { x: x + 1.1, y: y + 0.28, w: 4.6, h: 0.45, fontSize: 17, bold: true });
    txt(s, b, { x: x + 1.1, y: y + 0.78, w: 4.6, h: 0.95, fontSize: 14, color: HEX.muted });
  });
  takeaway(s, "Naming the limits is part of the result: every claim in this deck is backed by a measured number.");
  s.addNotes("Close the technical part by stating clearly what the results do and do not cover.");

  // ============================================================ 24. CLOSE
  s = pres.addSlide({ masterName: "DARK", sectionTitle: "Final model" });
  s.addShape(pres.shapes.OVAL, { x: 10.4, y: 4.7, w: 4.2, h: 4.2, fill: { color: HEX.violet, transparency: 20 }, line: { color: HEX.violet, transparency: 100 } });
  s.addShape(pres.shapes.OVAL, { x: 11.0, y: -1.2, w: 3.4, h: 3.4, fill: { color: HEX.orange, transparency: 15 }, line: { color: HEX.orange, transparency: 100 } });
  txt(s, "THREE THINGS TO REMEMBER", { x: MX + 0.1, y: 0.8, w: 9, h: 0.35, fontSize: 13, bold: true, color: "B9B0F5", charSpacing: 2 });
  const tk = [["1", "Learned features beat handcrafted ones", `${F.acc} vs 91.30% for the best Phase 1 model — ${870 - F.err} fewer errors.`],
    ["2", "We know why it works", "Depth + GAP only pay off with augmentation; results hold over 3 seeds."],
    ["3", "The final model knows when it is unsure", `Calibrated (ECE ${F.eceA}) and ${F.kept} accurate on the ${F.cov} of images it answers.`]];
  tk.forEach(([n, h, b], i) => {
    const y = 1.45 + i * 1.3;
    s.addShape(pres.shapes.OVAL, { x: MX + 0.1, y, w: 0.75, h: 0.75, fill: { color: i === 2 ? HEX.aqua : HEX.violet }, line: { color: HEX.ink } });
    txt(s, n, { x: MX + 0.1, y, w: 0.75, h: 0.75, fontSize: 24, bold: true, color: HEX.white, align: "center", valign: "middle" });
    txt(s, h, { x: MX + 1.15, y: y - 0.05, w: 9.5, h: 0.45, fontSize: 22, bold: true, color: HEX.white });
    txt(s, b, { x: MX + 1.15, y: y + 0.42, w: 9.5, h: 0.45, fontSize: 16, color: "C9C6E0" });
  });
  txt(s, "Thank you — questions?", { x: MX + 0.1, y: 5.45, w: 9, h: 0.8, fontSize: 36, bold: true, color: HEX.white, fontFace: "Arial" });
  txt(s, [{ text: "Live demo:  ", options: { bold: true, color: "FF9A6B" } }, { text: "streamlit run app/streamlit_app.py", options: { fontFace: "Courier New", color: HEX.white } }], { x: MX + 0.1, y: 6.35, w: 10, h: 0.4, fontSize: 15 });
  s.addNotes("Close with the three takeaways, then switch to the Streamlit demo: page 2 lets you try any test image or upload a photo.");

  await pres.writeFile({ fileName: OUT });
  if (APPLY_THEME) { const { applyTheme } = require(APPLY_THEME); await applyTheme(OUT, THEME); }
  console.log("wrote", OUT);
})();
