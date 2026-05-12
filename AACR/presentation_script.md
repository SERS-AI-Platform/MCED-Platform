# AACR 2026 Poster Presentation Script

## Poster Title
**Innovative label-free and non-invasive urinary metabolite analysis integrating AI and SERS technology for cancer detection: A retrospective clinical study involving five cancer types**

---

## Version 1: Elevator Pitch (1–3 min)

> Hi, thanks for stopping by.
>
> So what we're doing here is — we're trying to detect multiple cancers from a simple urine sample. No blood draw, no biopsy, just urine.
>
> The way it works is pretty straightforward. We take urine, put it on a gold nanoparticle chip, and shoot a laser at it — that's SERS, Surface-Enhanced Raman Scattering. You get a molecular fingerprint in about two minutes.
>
> Then our AI model — we call it **uSERS-Net** — does two things:
> - First, it asks: *"Is there cancer here or not?"*
> - If yes, it then figures out *which* cancer — among prostate, ovarian, lung, pancreatic, and colorectal.
>
> We tested this on **1,240 patients** from multiple hospitals in Korea. And importantly, our controls aren't just healthy people — we included diabetic and hypertensive patients too, because that's what you'd actually see in a real screening population.
>
> What gets us really excited is the potential for **point-of-care screening** — especially for something like pancreatic cancer, where by the time you catch it, it's usually too late.
>
> Happy to walk you through the details if you're interested.

---

## Version 2: Standard Presentation (5 min)

> Hi, thanks for your interest. Let me walk you through what we did.
>
> ### Background & Motivation (30 sec)
>
> So you know there's been a lot of buzz around multi-cancer early detection — ctDNA panels, Galleri, that kind of thing. They're really exciting, but the reality is they're still **expensive** and need a full lab setup. So we thought — what if we could get similar information from **urine**, using something much simpler?
>
> That's where SERS comes in. It gives you a molecular fingerprint of whatever's in the urine — metabolites, proteins, nucleic acids — all in one shot, no labeling needed, and you can do it with a **785-nm Raman spectrometer**.
>
> ### Study Design (45 sec)
>
> *[Point to Figure 1 — Pipeline]*
>
> We collected urine from **1,240 patients** at several hospitals across South Korea. Five cancer types — **prostate (100), ovarian (70), lung (300), pancreatic (70), colorectal (300)** — plus 400 non-cancer controls. And those controls include people with **diabetes, hypertension, and both combined** — not just healthy volunteers. That matters because in real life, the people getting screened have comorbidities.
>
> Each sample goes on a gold nanoparticle substrate, gets measured with a 785-nm Raman spectrometer, **five times per patient** — so 6,200 spectra total.
>
> ### Model Architecture (45 sec)
>
> Our model, **uSERS-Net**, is basically an ensemble — it combines a **logistic regression** (which takes in the SERS spectrum plus age, sex, and BMI) with a **ResNet18** deep learning model. It works in two stages:
>
> - **Stage 1**: Cancer or not? Binary question.
> - **Stage 2**: If cancer, which type? Five-way classification.
>
> We validated everything with **5-fold stratified cross-validation** and compared against five baselines — LR, Random Forest, XGBoost, CNN1D, and ResNet18 alone.
>
> ### Key Results (1 min 30 sec)
>
> *[Point to ROC curves]*
>
> So for the cancer-versus-not question, uSERS-Net hit an **AUC of 0.987** — best in the benchmark. Interestingly, plain logistic regression was at 0.981, which is pretty close. CNN1D was actually the worst at 0.890. So deep learning alone doesn't win here — it's the **combination** that works.
>
> *[Point to Confusion Matrices]*
>
> In terms of real numbers: **93.8% sensitivity, 94.7% specificity**. For cancer type identification — the five-class problem — we got a **Macro F1 of 0.883**. Prostate was correctly identified 95.9% of the time, colorectal 93.1%, lung 90.2%.
>
> *[Point to Subgroup Benchmark Heatmap]*
>
> And what's nice is that uSERS-Net doesn't just do well on average — it's **consistent across all five cancer types**. You can see in the radar chart, it covers the entire baseline range. Stage 2 F1 goes from 0.82 for pancreatic to 0.94 for lung.
>
> ### Spectral Biomarkers (30 sec)
>
> *[Point to SHAP feature importance]*
>
> And this isn't a black box. Each cell in the heatmap shows how much that wavenumber contributes to classifying that cancer type — it combines how different the spectrum is at that position with how much the model weighs it. Darker means more important. So you can see that PRC and OVC share many of the same important peaks, while LC has a very different pattern — the model is using different molecular information to identify each cancer. We're also building a **metabolite reference library** to confirm what those peaks actually represent — that work is ongoing, but preliminary results are promising.
>
> ### Conclusion (30 sec)
>
> So bottom line — a simple urine test, SERS, and AI can **screen for cancer and tell you which type it is**, with pretty high accuracy. We're now running a **prospective validation at Boramae Hospital** to make sure this holds up in a new cohort.
>
> Thanks — happy to take questions.

---

## Version 3: Extended Presentation (10 min)

> Thanks for visiting our poster. Let me take you through the whole story.
>
> ### Clinical Motivation (1 min)
>
> So everyone's talking about multi-cancer early detection, right? The ctDNA-based tests, the Galleri test — there's real momentum there. But here's the thing — those tests need a blood draw, a sophisticated sequencing lab, and they cost a lot. That limits who can actually get screened.
>
> We came at it from a completely different angle. What if you could get this kind of information from **urine**? No needles. The patient can do it at home. And instead of sequencing, you use **SERS** — Surface-Enhanced Raman Scattering — which gives you a full molecular fingerprint in under two minutes.
>
> So the question was: *Is the metabolic information in urine, captured by SERS, rich enough to not just detect cancer, but also tell you which type?*
>
> ### Cohort & Study Design (1 min 30 sec)
>
> *[Point to Table 1 — Demographics]*
>
> We put together a cohort of **1,240 patients** — five cancer types: **prostate** (n=100, median age 72), **ovarian** (n=70), **lung** (n=300), **pancreatic** (n=70), and **colorectal** (n=300). These came from multiple hospitals — Chungbuk National University Hospital, Seoul National University Hospital, Seoul St. Mary's, and others.
>
> Now here's something we were careful about. Our control group of **400 subjects** isn't just healthy people. We included four groups of 100 each — **normal, diabetes, hypertension, and diabetes-plus-hypertension**. Why? Because if you're building a screening test, the people you're screening in the real world have these conditions. If your model can't handle that, it's not useful.
>
> *[Point to Figure 1 — Pipeline]*
>
> The measurement itself is simple. Urine goes on a **gold nanoparticle substrate**, you measure it with a 785-nm Raman spectrometer, and you get a spectrum. We do **five replicates per patient**, which gives us 6,200 spectra total. The whole measurement takes a couple of minutes.
>
> ### Two-Stage AI Architecture (1 min 30 sec)
>
> So our model, **uSERS-Net**, has two stages.
>
> **Stage 1** is the screening step — cancer or not cancer. It takes the SERS spectrum and we also feed in clinical features — age, sex, BMI — because those do carry some signal.
>
> **Stage 2** kicks in for cancer-positive samples and classifies them into one of five types.
>
> Under the hood, it's an **ensemble** of two things: a **Fusion Logistic Regression** — which is basically LR on the spectrum plus clinical variables — and a **ResNet18** deep learning model. We blend them with alpha = 0.8. The idea is that LR captures the linear signal really well — and for SERS data, there's a lot of that — while ResNet picks up non-linear patterns that LR misses. Together they're better than either one alone.
>
> We tested against **five baselines** — LR, Random Forest, XGBoost, CNN1D, ResNet18 — all under the same **5-fold stratified CV** setup.
>
> ### Stage 1 Results — Cancer Screening (1 min 30 sec)
>
> *[Point to ROC curves]*
>
> For Stage 1, uSERS-Net got the top AUC at **0.987**. LR was close at 0.981, XGBoost at 0.969, ResNet at 0.961. Random Forest dropped to 0.931, and CNN1D was the weakest at 0.890.
>
> What's interesting here is that **plain LR almost matches the ensemble**. That tells you the spectral features are very linearly separable. But the deep learning component still adds a bit — that 0.006 AUC gap is real when you're talking about screening.
>
> *[Point to Confusion Matrix A]*
>
> At the operating point: **93.8% sensitivity, 94.7% specificity**. So for every 100 cancer patients, we catch about 94. And for every 100 non-cancer patients, we correctly rule out about 95. Those are solid numbers for a urine-based screening test.
>
> *[Point to Subgroup Heatmap — Panel A]*
>
> If you break it down by cancer type, detection sensitivity ranges from **0.80 for prostate to 0.98 for pancreatic**. Prostate being lower makes sense — most of our prostate patients have localized disease, and localized prostate cancer doesn't alter urinary metabolites as dramatically. Pancreatic and colorectal, on the other hand, cause much bigger metabolic disruptions — and the model picks that up.
>
> ### Stage 2 Results — Cancer Type Identification (1 min 30 sec)
>
> *[Point to Confusion Matrix B]*
>
> OK so this is the part I find most exciting. We're not just detecting cancer — we're telling you **which cancer**. The five-class confusion matrix shows a **Macro F1 of 0.883**. Breaking that down: **prostate 95.9%, colorectal 93.1%, lung 90.2%, ovarian 84.6%, pancreatic 83.5%**.
>
> The main mix-up is between pancreatic and colorectal — about 14.5% of pancreatic cases get called colorectal. But honestly, that makes biological sense — they're both GI cancers, they share metabolic pathways.
>
> *[Point to Radar Chart & Subgroup Heatmap]*
>
> And if you look at the radar chart and the benchmark heatmap, the key takeaway is that uSERS-Net **consistently outperforms all baselines across every cancer type**. The biggest gain is on pancreatic — where CNN1D only managed F1 of 0.28, uSERS-Net gets 0.82.
>
> ### Spectral Biomarkers & Interpretability (1 min)
>
> *[Point to SHAP analysis figures]*
>
> One thing I want to emphasize — this is not a black box. Each cell in the heatmap shows how much that wavenumber contributes to classifying that cancer type — it combines how different the spectrum is at that position with how much the model weighs it. Darker means more important. So you can see that PRC and OVC share many of the same important peaks, while LC has a very different pattern — the model is using different molecular information to identify each cancer.
>
> And these are discriminative peaks from our own data — not borrowed from literature. Each cancer type lights up at different wavenumbers: prostate around 2100, lung around 990, pancreatic around 725, colorectal around 1350.
>
> Now, what those peaks actually *mean* biologically — that's something we're actively working on. We're building a **SERS metabolite reference library**, measuring individual metabolites on the same substrate to directly match them to these peaks. It's still ongoing and I can't share specifics yet, but our preliminary results suggest these peaks do correspond to meaningful metabolic signatures. So we're moving from "the model found this peak useful" to "here's the specific metabolite behind it."
>
> ### Conclusions & Future Directions (30 sec)
>
> So to wrap up — we've shown that a **urine sample, SERS, and an AI model** can:
> 1. Screen for cancer — AUC 0.987
> 2. Identify the cancer type — Macro F1 0.883
>
> The next step is a **prospective validation at Boramae Hospital** — 300 cancer patients, 150 controls — completely independent from the training data. We're expecting results later this year.
>
> Thanks a lot. I'd love to hear your questions or feedback.

---

## Supplementary: Tablet에서 보여줄 때 (포스터에 없는 내용)

> *[Show TOO summary table on tablet]*
>
> This isn't on the poster, but the question everyone asks is: **does this work for early-stage cancers?** Not just detecting them, but correctly identifying the cancer type — the tumor of origin.
>
> And the answer is yes. At the patient level, early-stage TOO accuracy is:
>
> - **Colorectal Stage I–II**: 98.0% (248/253)
> - **Lung Stage I–II**: 97.8% (88/90)
> - **Prostate T1–T2**: 96.6% (28/29)
> - **Pancreatic Stage I–II**: 87.2% (34/39)
>
> *[Show spectra comparison]*
>
> And if you look at the actual spectra — gray is Non-cancer, blue is Early stage, red dashed is Late — you can see that even at early stage, the spectral signature is already **clearly different from normal**. The yellow bands highlight where the biggest differences are. Late stage makes those differences even bigger, but the signal is already there early.

**파일:**
- `suppl_too_summary.png` — Demographics + Stage 분포 + TOO accuracy 테이블
- `suppl_spectra_normal_early_late.png` — Non-cancer → Early → Late 스펙트럼

---

## Q&A Preparation — Anticipated Questions

### Q1: "How do you handle the class imbalance?"
> Yeah so we have 840 cancer versus 400 non-cancer. We used stratified cross-validation, so each fold keeps the same ratio. And importantly, all five replicates from the same patient stay in the same fold — we're really careful about data leakage. We also report AUC, which is threshold-independent, alongside sensitivity and specificity at the Youden optimal point.

### Q2: "Why not just use deep learning end-to-end?"
> That's a great question, and it's actually one of the surprising findings. Standalone CNN1D only got an AUC of 0.890 — way below logistic regression at 0.981. SERS spectra turn out to be very linearly separable, so LR captures most of the signal. But the deep learning component does add something — the ensemble gets 0.987. So it's better to combine them than to bet on either alone.

### Q3: "Is this really detecting cancer biology, or batch/site effects?"
> That's the right question to ask. A few things give us confidence. First, the cancer type identification — Macro F1 of 0.883 distinguishing *between* five cancer types — that's really hard to explain with batch effects alone. If it were just site artifacts, you'd maybe separate cancer from control, but you wouldn't differentiate five different cancers that well. Second, the SHAP peaks map to known biochemistry. And third, we're running a prospective validation right now at a completely separate hospital to confirm generalizability.

### Q4: "What about the cost and throughput?"
> The substrate is about one to two dollars per test. The Raman spectrometer is a one-time purchase. And the measurement takes under five minutes. Compare that to ctDNA panels that run hundreds or thousands of dollars per test with weeks of turnaround — it's a completely different cost structure. We think this could work at the population screening level.

### Q5: "Why urine instead of blood?"
> Practically, urine is about as non-invasive as it gets. No needles, no trained personnel, patients can even collect at home. You get unlimited volume. And from a biology standpoint, urine captures a lot of the metabolic changes that happen systemically with cancer. The SERS fingerprint picks those up. We think this makes it uniquely suited for **decentralized screening** — places where you can't set up a sequencing lab.

### Q6: "What's the clinical pathway you envision?"
> We see this as a **first-line screen** — in primary care, community health, maybe even at-home. If the test flags something, Stage 2 tells you the likely cancer type, and then the clinician orders the appropriate follow-up. Pancreatic flag? CT plus CA 19-9. Prostate flag? PSA and MRI. That way you avoid shotgun diagnostic workups and go straight to targeted confirmation.

### Q7: "Your controls have diabetes and hypertension — those are metabolic diseases too. How do you know you're not just picking up metabolic noise?"
> Right, and that's exactly why we designed it that way. We're measuring metabolites, so diabetes, hypertension — these change the urinary metabolome significantly. If we'd only compared cancer patients against healthy volunteers, you could absolutely argue we're just detecting metabolic disruption in general, not cancer specifically. So we deliberately put those metabolically altered patients in the control group. And the fact that our model still separates cancer from DM, HTN, and DM-plus-HTN controls — with AUC of 0.987 — that tells you the cancer-specific metabolic signal is strong enough to survive those confounders. It's not just "sick versus healthy" — it's "cancer versus other metabolic conditions."

### Q8: "How did you select the ensemble weight (alpha = 0.8)?"
> We actually did a grid search over alpha from 0 to 1 within the cross-validation. Alpha = 0.8 — meaning 80% LR, 20% ResNet — consistently gave the best combined performance. It makes sense: the LR is doing most of the heavy lifting, and the ResNet is just adding a correction for non-linear patterns that LR misses.
