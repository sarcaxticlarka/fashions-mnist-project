# Presentation script: English + Hinglish (19 slides)

Deck: `docs/Fashion-MNIST_Phase1_Presentation.pptx`. Each slide has **EN** (say in English), **HI** (Hinglish version) and **If they ask** (answers for follow-up questions).

**Suggested speaker split (5 members, ~2 min each):**
Member 1: slides 1–4 · Member 2: slides 5–7 · Member 3: slides 8–10 · Member 4: slides 11–15 · Member 5: slides 16–19.

**Numbers cheat-sheet (learn these):**
70,000 images (60k train + 10k test) · 28×28 · 10 classes · validation 12,000 (20% of train) · HOG = 864 numbers (6 bins, 4×4 cells, 2×2 blocks) · Best: SVM + HOG = **91.3%**, macro-F1 0.913 · Lowest: LogReg + pixels = 84.6% · HOG gain: LogReg +4.0, kNN +2.1, SVM +1.4, RF +0.2 (not significant, p=0.49) · ~60% of errors inside T-shirt/Pullover/Coat/Shirt · Shirt recall 0.75 · SVM is ~4,000× slower than LogReg per image.

---

## Slide 1: Title (Clothing Category Recognition)
**EN:** Good morning. We are Team T002. Our project is automatic clothing category recognition. A small shop has thousands of product photos and needs a tag on each one, like shirt, trouser or bag. We study one question: if we give a classical machine-learning model a smart image feature called HOG, does it do better than if we give it raw pixels? We use the Fashion-MNIST dataset, 4 models and 3 ways of representing the image. This is Phase 1, the machine-learning part. Phase 2 will use a CNN.

**HI:** Good morning sir/ma'am. Hum Team T002 hain. Hamara project hai automatic clothing category recognition. Maan lijiye ek chhoti shop ke paas hazaaron product photos hain aur har photo pe tag lagana hai, jaise shirt, trouser ya bag. Hum ek sawaal study kar rahe hain: agar hum classical machine-learning model ko raw pixels ki jagah ek smart feature, HOG, dein to kya result behtar hota hai? Dataset Fashion-MNIST hai, 4 models aur image ko dikhane ke 3 tarike. Ye Phase 1 hai, yaani machine learning. Phase 2 mein hum CNN use karenge.

**If they ask:**
- *What is HOG?* Histogram of Oriented Gradients. It describes the direction of edges in small parts of the image. / HOG image ke chhote hisson mein edges ki direction ko number mein badalta hai.
- *Why Phase 1 and 2?* The course splits the project: Phase 1 is classical ML, Phase 2 is deep learning on the same data.

---

## Slide 2: The problem we solve
**EN:** There are three parts. The need: a shop must tag every photo, and doing it by hand is slow and inconsistent. The gap: if we flatten an image into a long row of numbers, the model no longer knows which pixels are neighbours. HOG keeps that local structure, and it is fixed before the model sees any category. The goal: compare 4 classic models on 3 representations and report not only accuracy but also macro-F1, per-class recall, model size and speed. Our research question is on the right: do fixed descriptors beat raw pixels, and for which models?

**HI:** Teen cheezein hain. Need: shop ko har photo tag karni hai aur haath se karna slow aur inconsistent hai. Gap: jab hum image ko flatten karke lambi row bana dete hain to model ko pata nahi chalta ki kaun se pixels ek doosre ke paas hain. HOG ye local structure bacha leta hai, aur ye model ke category dekhne se pehle hi fix hota hai. Goal: 4 classic models ko 3 representations pe compare karna, sirf accuracy nahi, macro-F1, per-class recall, model size aur speed bhi. Hamara research question: kya fixed descriptors raw pixels ko beat karte hain, aur kin models ke liye?

**If they ask:**
- *What is flattening?* A 28×28 image becomes one row of 784 numbers. / 28×28 image ek row of 784 numbers ban jaati hai.
- *What is macro-F1?* The average F1 score over the 10 classes, so every class counts equally.

---

## Slide 3: The data (Fashion-MNIST)
**EN:** Fashion-MNIST is from Zalando. It has 70,000 grayscale images of 28 by 28 pixels in 10 classes: T-shirt, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag and ankle boot. Each class has the same number of images, 6,000 in training and 1,000 in test. We keep the official train and test split. The 12,000 validation images are taken only from the training set, so the test set stays untouched until the very end.

**HI:** Fashion-MNIST Zalando ka dataset hai. Ismein 70,000 grayscale images hain, har ek 28×28 pixel ki, aur 10 classes: T-shirt, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag aur ankle boot. Har class mein barabar images hain, training mein 6,000 aur test mein 1,000. Official train aur test split hum badalte nahi. Validation ke 12,000 images sirf training set se liye hain, isliye test set aakhir tak bilkul untouched rehta hai.

**If they ask:**
- *Why validation set?* To choose hyper-parameters without looking at test. / Hyper-parameters chunne ke liye, test dekhe bina.
- *Is it balanced?* Yes, exactly 10% per class, so accuracy is a fair metric.

---

## Slide 4: Tools we used
**EN:** Everything is in Python. NumPy reads the raw files and scales pixels to 0–1. scikit-image gives us the HOG function. scikit-learn gives the 4 models, PCA, t-SNE, the train-validation split and all metrics. SciPy gives the Sobel edge filter and the statistical tests. Matplotlib, Seaborn and Plotly make the figures. Streamlit is our interactive demo app. The code is modular: one file each for EDA, features, models and analysis, so anyone can re-run it.

**HI:** Sab kuch Python mein hai. NumPy raw files padhta hai aur pixels ko 0 se 1 mein scale karta hai. scikit-image HOG function deta hai. scikit-learn se 4 models, PCA, t-SNE, split aur saare metrics aate hain. SciPy se Sobel edge filter aur statistical tests. Matplotlib, Seaborn aur Plotly se graphs. Streamlit hamara interactive demo app hai. Code modular hai: EDA, features, models aur analysis ki alag-alag files, taaki koi bhi dobara run kar sake.

**If they ask:**
- *Why Streamlit, not a web framework?* Same Python code and saved models can be reused directly, no extra backend.
- *Where is the code?* github.com/sarcaxticlarka/fashions-mnist-project

---

## Slide 5: How we did the EDA
**EN:** Our EDA has four steps. One, audit: we hash every image to find duplicates, and check for missing values, blank images and train-test overlap. We found zero problems. Two, distributions: class counts, pixel-intensity histogram, brightness and bounding box per class, and a KS test comparing train and test. Classes are balanced, train and test are alike, and half of all pixels are exactly zero. Three, structure: how correlated neighbouring pixels are, which pixels separate classes, and how far apart class-average images are. Four, embeddings: PCA and t-SNE to see how classes overlap. 187 PCA components keep 95% of the variance.

**HI:** EDA ke chaar steps hain. Ek, audit: har image ka hash nikaal ke duplicates dhoondhe, aur missing values, blank images aur train-test overlap check kiya. Kuch problem nahi mili. Do, distributions: class counts, pixel intensity histogram, har class ki brightness aur bounding box, aur KS test jo train aur test ko compare karta hai. Classes balanced hain, train aur test similar hain, aur aadhe pixels exactly zero hain. Teen, structure: neighbouring pixels kitne related hain, kaun se pixels classes ko alag karte hain, aur class ke average images kitne door hain. Chaar, embeddings: PCA aur t-SNE se dekha ki classes kahan overlap karti hain. 187 PCA components 95% variance rakhte hain.

**If they ask:**
- *What is a KS test?* It checks whether two samples come from the same distribution. Our p = 0.87, so train and test are similar. / Ye check karta hai ki do samples same distribution se aaye hain ya nahi.
- *What is PCA?* It finds a few new directions that keep most of the variation. 784 numbers can be reduced to about 187 with 95% information.
- *What is t-SNE?* A method to draw high-dimensional data in 2D so that similar items stay close. We used it only for visualisation.
- *Why hash for duplicates?* Exact same image would give exact same hash; it is fast.

---

## Slide 6: Neighbouring pixels are related
**EN:** This graph is our main EDA insight. On the x-axis is the distance between two pixels; on the y-axis, how correlated they are. Green is the real image: adjacent pixels have a correlation of 0.75, and it falls as they get further apart. White is the same pixels with positions shuffled: about 0.16 everywhere. So the gap is real spatial structure. A flattened vector ignores this layout. HOG uses it, because it looks at local patches. That is why we expected HOG to help.

**HI:** Ye graph hamara main EDA insight hai. X-axis par do pixels ke beech ki doori hai aur y-axis par unki correlation. Green line real image hai: paas-paas ke pixels ki correlation 0.75 hai aur doori badhne par girti hai. White line wahi pixels hain lekin positions shuffle karke: har jagah lagbhag 0.16. Matlab beech ka gap real spatial structure hai. Flattened vector is layout ko ignore karta hai. HOG ise use karta hai kyunki wo local patches dekhta hai. Isi wajah se humne expect kiya ki HOG help karega.

**If they ask:**
- *What is correlation?* How much two values move together, from -1 to +1.
- *Why shuffle?* It is a control. It shows what correlation we would get by chance if position did not matter.

---

## Slide 7: EDA predicts the hard classes
**EN:** Here we averaged all images of each class and measured the distance between those average images. A small number means the classes look alike. The closest pairs are pullover and coat at 2.48, pullover and shirt at 2.68, coat and shirt at 3.02, and T-shirt and shirt at 3.72. All four are upper-body clothes. So before training anything, EDA predicted that these are the classes models will confuse.

**HI:** Yahan humne har class ki saari images ka average liya aur un average images ke beech ki doori naapi. Chhota number matlab classes ek jaisi dikhti hain. Sabse paas wale pairs: pullover aur coat 2.48, pullover aur shirt 2.68, coat aur shirt 3.02, aur T-shirt aur shirt 3.72. Chaaron upper-body kapde hain. Yaani model train karne se pehle hi EDA ne bata diya ki inhi classes mein confusion hoga.

**If they ask:**
- *Did the prediction come true?* Yes, see slide 16: about 60% of errors are inside these four classes.
- *Which distance?* Euclidean distance between mean images.

---

## Slide 8: Feature engineering: 3 representations
**EN:** We compare three ways to describe the same image. Raw pixels: 784 numbers, the baseline. HOG: 864 numbers that describe edge directions in small cells. And HOG plus 16 shape statistics, 880 numbers, such as width, height, symmetry and edge density. Labels, split and scaling are identical. Only the representation changes, so any difference comes from the features.

**HI:** Hum ek hi image ko teen tarike se describe karke compare karte hain. Raw pixels: 784 numbers, ye baseline hai. HOG: 864 numbers jo chhote cells mein edges ki direction batate hain. Aur HOG plus 16 shape statistics, 880 numbers, jaise width, height, symmetry aur edge density. Labels, split aur scaling bilkul same hain. Sirf representation badalti hai, isliye jo bhi difference aaye wo features ki wajah se hai.

**If they ask:**
- *What are shape statistics?* Simple facts: bounding-box width and height, aspect ratio, centre of mass, left-right and top-bottom symmetry, edge density, etc.
- *Why check orientation?* HOG depends on edge direction, so images must be upright. EDA showed shoes are heavier at the bottom and tops at the top, so they are upright.

---

## Slide 9: How HOG works
**EN:** Left to right: the 28 by 28 input, its gradients, and the HOG map. Four steps. One, compute the gradient, how fast brightness changes in x and y; magnitude is the square root of gx squared plus gy squared. Two, split the image into 4 by 4 pixel cells, giving a 7 by 7 grid, and in each cell build a 6-bin histogram of edge directions, weighted by magnitude. Three, group 2 by 2 cells into blocks and normalise with L2-Hys so lighting changes matter less. Four, join everything: 36 blocks times 4 cells times 6 bins equals 864 numbers. This is the exact function call in the box.

**HI:** Left se right: 28×28 input, uske gradients, aur HOG map. Chaar steps. Ek, gradient nikalna, yaani brightness x aur y direction mein kitni tezi se badal rahi hai; magnitude gx square plus gy square ka square root hai. Do, image ko 4×4 pixel ke cells mein todna, jisse 7×7 grid banta hai, aur har cell mein 6-bin histogram banana edge directions ka, magnitude se weight karke. Teen, 2×2 cells ke blocks banana aur L2-Hys se normalise karna taaki lighting ka effect kam ho. Chaar, sab jodna: 36 blocks × 4 cells × 6 bins = 864 numbers. Box mein exact function call dikhaya hai.

**If they ask:**
- *What is a gradient?* The change in brightness between neighbouring pixels. Big change = an edge.
- *Why normalise?* So a darker or brighter photo gives a similar descriptor.
- *Why 36 blocks?* With 7×7 cells and 2×2 blocks sliding by one cell, we get 6×6 = 36 blocks.
- *Is HOG learned?* No. It is fixed by a formula, not trained. That is the key difference from a CNN.

---

## Slide 10: Choosing the HOG settings
**EN:** We did not guess the settings. We tried 36 combinations of orientation bins, cell size and block size, and scored them on the validation set with a fast logistic-regression probe. Smaller 2 by 2 cells scored slightly higher, 87.0%, but gave 8,112 numbers per image, which makes kNN and SVM very slow. So we set a budget of 2,000 numbers and chose the best inside it: 6 bins, 4 by 4 cells, 2 by 2 blocks, 864 numbers, 85.6% on the probe. The white card is the second feature idea: 16 shape statistics.

**HI:** Humne settings guess nahi ki. Orientation bins, cell size aur block size ke 36 combinations try kiye aur validation set par ek fast logistic-regression probe se score kiya. Chhote 2×2 cells ne thoda zyada score kiya, 87.0%, lekin 8,112 numbers per image bane, jisse kNN aur SVM bahut slow ho jaate hain. Isliye humne 2,000 numbers ka budget rakha aur uske andar best chuna: 6 bins, 4×4 cells, 2×2 blocks, 864 numbers, probe par 85.6%. White card doosra feature idea hai: 16 shape statistics.

**If they ask:**
- *Why not use the best 8,112-d?* Cost grows with dimensions; SVM and kNN would be several times slower.
- *Limitation?* The settings were picked with one simple probe model, not separately for each classifier. We say this in the report.

---

## Slide 11: A fair experiment
**EN:** Four steps. Keep the official split: 60k train and 10k test. Take 20% of the training data, stratified, as validation. Tune hyper-parameters on validation. Then refit on train plus validation and test once. Three rules: no leakage, because test never touches tuning and scalers are fitted on training data only; equal effort, same seed 42, scaling and grid budget for every feature set; and full reporting of accuracy, macro-F1, per-class recall, parameters and inference time.

**HI:** Chaar steps. Official split rakho: 60k train aur 10k test. Training data ka 20%, stratified, validation banao. Hyper-parameters validation par tune karo. Phir train plus validation par refit karo aur test sirf ek baar karo. Teen rules: koi leakage nahi, kyunki test tuning ko kabhi touch nahi karta aur scalers sirf training data par fit hote hain; equal effort, har feature set ke liye same seed 42, same scaling aur same grid budget; aur poori reporting: accuracy, macro-F1, per-class recall, parameters aur inference time.

**If they ask:**
- *What is stratified?* Every class keeps its same share in the split.
- *Why refit on train+val?* More data for the final model, with already-chosen settings; no test peeking.
- *What is data leakage?* Information from test or validation sneaking into training, which makes results look better than they are.

---

## Slide 12: Four classic models
**EN:** Logistic Regression draws a straight-line boundary, very small and fast; we tuned its regularisation C. kNN stores all training images and votes among the k closest; we tried k of 3, 5 and 9. Random Forest is 300 decision trees that vote; we tried depth and minimum leaf size. SVM with an RBF kernel draws a curved boundary; we tried C of 1, 5 and 20. Because SVM is slow, we tuned it on 15,000 images and then refit on all 60,000.

**HI:** Logistic Regression seedhi line ka boundary banata hai, bahut chhota aur fast; humne uska regularisation C tune kiya. kNN saari training images store karta hai aur sabse paas ki k images se vote karta hai; humne k = 3, 5 aur 9 try kiye. Random Forest 300 decision trees hain jo vote karte hain; humne depth aur minimum leaf size try kiya. RBF kernel wala SVM curved boundary banata hai; humne C = 1, 5, 20 try kiye. SVM slow hai isliye humne use 15,000 images par tune kiya aur phir saari 60,000 par refit kiya.

**If they ask:**
- *What is C?* Controls regularisation: smaller C means a simpler model; in SVM it trades margin width against training errors.
- *What is an RBF kernel?* It measures similarity by distance, so the boundary can curve.
- *Why these four?* They are four different families: linear, instance-based, tree ensemble and kernel.

---

## Slide 13: Results: test accuracy
**EN:** This is the main result, all 12 experiments on the 10,000 test images. Grey is pixels, white is HOG, green is HOG plus stats. The best is SVM with HOG at 91.3%, and 91.4% with HOG plus stats; macro-F1 is 0.913. The lowest is Logistic Regression on raw pixels at 84.6%. The 95% bootstrap interval for the best model is about 90.8 to 91.9, clearly above the best pixel model.

**HI:** Ye main result hai, 10,000 test images par saare 12 experiments. Grey pixels hai, white HOG hai, green HOG plus stats hai. Sabse best SVM with HOG hai, 91.3%, aur HOG plus stats ke saath 91.4%; macro-F1 0.913. Sabse kam Logistic Regression raw pixels par hai, 84.6%. Best model ka 95% bootstrap interval lagbhag 90.8 se 91.9 hai, jo best pixel model se saaf upar hai.

**If they ask:**
- *91.3 or 91.4?* HOG: 91.30, HOG+stats: 91.37. The difference is inside noise, so we call HOG alone the default.
- *What is bootstrap?* Re-sample the test set many times to see how much accuracy would vary.
- *What is macro-F1?* Average of per-class F1, so each class counts equally.

---

## Slide 14: Does HOG help?
**EN:** Gain from HOG over pixels, in accuracy points: Logistic Regression +4.0, kNN +2.1, SVM +1.4, all significant with McNemar's test at p below 0.001. Random Forest gains only +0.2 with p of 0.49, so it is not a real difference. Why? A linear model cannot build edge features by itself, so HOG helps it most. A Random Forest already finds useful pixel thresholds. The more flexible the model, the less it needs HOG.

**HI:** Pixels ke mukable HOG ka gain, accuracy points mein: Logistic Regression +4.0, kNN +2.1, SVM +1.4, teeno McNemar test se significant hain, p 0.001 se kam. Random Forest ko sirf +0.2 mila aur p 0.49 hai, yaani ye asli difference nahi hai. Kyun? Linear model khud edge features nahi bana sakta, isliye HOG usse sabse zyada help karta hai. Random Forest pehle se hi useful pixel thresholds dhoondh leta hai. Model jitna flexible, HOG ki utni kam zaroorat.

**If they ask:**
- *What is McNemar's test?* A paired test on the same test images: it counts images where only one model is right. If the counts differ a lot, the models are truly different.
- *What does p < 0.001 mean?* The chance that the gain is just luck is less than 0.1%.
- *Why not a t-test?* Both models predict the same images, so the data is paired and the cases are yes/no, which McNemar's test is made for.

---

## Slide 15: Recall per class (SVM)
**EN:** Recall means: out of all real shirts, how many did we find. Grey is pixels, green is HOG. Shirt is the hardest, 0.705 with pixels and 0.75 with HOG. Pullover, coat and T-shirt are next, between 0.83 and 0.88. Trouser, bag, sandal, sneaker and ankle boot are all above 0.96. HOG helps the weak upper-body classes the most, but it does not fix them.

**HI:** Recall ka matlab: jitni asli shirts thi, unmein se humne kitni dhoondhi. Grey pixels hai, green HOG hai. Shirt sabse mushkil hai, pixels se 0.705 aur HOG se 0.75. Phir pullover, coat aur T-shirt, 0.83 se 0.88 ke beech. Trouser, bag, sandal, sneaker aur ankle boot sab 0.96 se upar hain. HOG kamzor upper-body classes ko sabse zyada help karta hai, lekin unhe poori tarah theek nahi karta.

**If they ask:**
- *Recall vs precision?* Recall = found / all real ones. Precision = correct / all predicted as that class.
- *Why look per class?* Overall accuracy can hide a class that is doing badly.

---

## Slide 16: Where it still fails
**EN:** This is the confusion matrix of the best model, in percent per row. The diagonal is correct predictions; the off-diagonal cells are errors. The shirt row is the clearest: only 75% are correct, and 10% are called T-shirt. About 60% of all errors stay inside T-shirt, pullover, coat and shirt, as EDA predicted. The reason: at 28 by 28 pixels, collars and buttons are only one or two pixels wide, so there is little detail left to tell them apart.

**HI:** Ye best model ka confusion matrix hai, har row mein percent. Diagonal sahi predictions hain, baaki cells errors hain. Shirt wali row sabse clear hai: sirf 75% sahi hain aur 10% ko T-shirt bol diya. Lagbhag 60% errors T-shirt, pullover, coat aur shirt ke andar hi rehte hain, jaisa EDA ne predict kiya tha. Wajah: 28×28 pixels mein collar aur buttons sirf ek ya do pixel chaude hote hain, isliye alag batane ke liye detail bachti hi nahi.

**If they ask:**
- *What is a confusion matrix?* Rows are true classes, columns are predicted classes.
- *Can it be improved?* A CNN can learn its own filters (Phase 2). Higher resolution would also help.
- *Is it a labelling problem?* Some images are ambiguous even for humans, which also limits any model.

---

## Slide 17: Accuracy has a price
**EN:** Accuracy is not the only thing. Logistic Regression with HOG is 88.5% and takes 0.001 ms per image with about 8,700 stored numbers. The SVM is 91.3% but needs 5.4 ms and 18 million numbers. So SVM is 2.8 points better, but roughly 4,000 times slower and 2,000 times larger. kNN stores 52 million numbers because it keeps the whole training set. HOG extraction itself adds only 0.14 ms per image. The right model depends on the deployment: tiny and fast, or most accurate.

**HI:** Sirf accuracy sab kuch nahi hai. HOG ke saath Logistic Regression 88.5% hai aur ek image par 0.001 ms leta hai, lagbhag 8,700 numbers store karta hai. SVM 91.3% hai lekin 5.4 ms aur 18 million numbers chahiye. Yaani SVM 2.8 points behtar hai lekin lagbhag 4,000 guna slow aur 2,000 guna bada. kNN 52 million numbers store karta hai kyunki wo poora training set rakhta hai. HOG nikalne mein sirf 0.14 ms lagta hai. Sahi model deployment par depend karta hai: chhota aur fast, ya sabse accurate.

**If they ask:**
- *How did you count parameters?* LogReg: weights+biases; SVM: support vectors + coefficients; RF: 2 numbers per tree split; kNN: the stored training matrix.
- *How was speed measured?* Single-threaded predict time on the 10,000 test images divided by 10,000.

---

## Slide 18: Conclusions and next
**EN:** Three conclusions. HOG helps three of the four models, significantly. The best classic model is SVM with HOG at 91.3% accuracy and 0.913 macro-F1. The hard group of shirt, T-shirt, pullover and coat causes about 60% of errors. In Phase 2 we will train a small CNN with 2 to 3 convolution blocks, pooling, dropout and early stopping on the same splits and metrics, and test whether it separates the shirt group better than any fixed descriptor.

**HI:** Teen conclusions. HOG chaar mein se teen models ko significantly help karta hai. Best classic model SVM with HOG hai, 91.3% accuracy aur 0.913 macro-F1. Shirt, T-shirt, pullover aur coat ka mushkil group lagbhag 60% errors karta hai. Phase 2 mein hum chhota CNN train karenge, 2 se 3 convolution blocks, pooling, dropout aur early stopping ke saath, same splits aur same metrics par, aur dekhenge ki kya wo shirt group ko kisi bhi fixed descriptor se behtar alag karta hai.

**If they ask:**
- *Limitations?* HOG settings tuned with one probe model; one validation split; SVM tuned on a 15k subset; 28×28 grayscale loses texture.
- *Why would a CNN be better?* It learns its own features from data instead of using a fixed formula.

---

## Slide 19: Thank you
**EN:** Thank you. We are Team T002, Section C. The members are on the slide, and our full code, notebooks, report and the interactive app are in the GitHub repository. We are happy to take questions.

**HI:** Thank you. Hum Team T002, Section C hain. Team members slide par hain, aur hamara poora code, notebooks, report aur interactive app GitHub repository mein hai. Ab hum aapke questions ke liye ready hain.

---

# General questions they may ask (quick answers)

| Question | Short answer (EN / Hinglish) |
|---|---|
| Why Fashion-MNIST, not MNIST? | Same format but harder; digits are nearly solved. / Format wahi, lekin zyada mushkil. |
| Why not just use a CNN? | Project asks to compare handcrafted vs CNN. Phase 1 is the handcrafted baseline. |
| What is overfitting? | Model memorises training data and fails on new data. We check with validation and test. |
| How did you avoid overfitting? | Tuning on validation, regularisation (C, depth), refit and test once. |
| Why is accuracy fair here? | Classes are balanced, and we also report macro-F1 and per-class recall. |
| What is hyper-parameter tuning? | Choosing settings like k, C or tree depth by trying options on validation. |
| Is the result reproducible? | Yes: seed 42, scripts/run_all.sh regenerates all figures and tables. |
| What did you do beyond the brief? | PCA/t-SNE, HOG ablation (36 settings), shape stats, McNemar test, bootstrap CI, latency and size, Streamlit demo, IEEE report. |
| Can it classify real photos? | Not reliably. It is trained on 28×28 Fashion-MNIST; real photos need a clean background and resizing. / Real photos par reliably nahi, kyunki training Fashion-MNIST par hui hai. |
| What is stratified split? | Each class keeps the same percentage in every split. |
| What does 0.913 macro-F1 mean? | The average balance of precision and recall across the 10 classes is 91.3%. |
| Who did what? | (Say your own split: e.g. EDA, features, models, evaluation, app/report.) |
