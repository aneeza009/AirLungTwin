# AirLung Twin — 30% Implementation: Plan, Notes & Supervisor-Feedback Response

Prepared after the supervisor meeting (23 Nov 2026). Deadline: ~1 week.
Goal of 30%: show the ONE thing only this project does — **how air quality affects a
specific person's breathing** (the gap we solved) — not a generic air dashboard.

---

## A. Honest read of the supervisor's feedback (what to do vs what to ignore)

**MUST do (he's right):**
1. Stop showing raw values; convert them to **interpreted, plain-language info**.
2. Make the interface **interactive and informative** (good UI, trends, comparisons).
3. Show **our contribution clearly**: air quality → THIS patient's respiration. The
   Punjab EPA site already does generic city air quality; our novelty is the *person link*.
4. Be **live-ready**: structure it so the moment live data flows in, it works.
5. Point to where the **prediction model** (Khazeena's thesis direction) plugs in.

**Be careful / don't over-expand (stay true to our proposal):**
- He drifted into *all pollution types* (waste, water, soil, noise) and full govt-style
  dashboards. Our scope is **respiratory health via IoMT** — air ↔ breathing. Other
  pollution types belong in the **report as research factors**, NOT in the 30% build.
- **Full ML prediction is Jan–Feb** per our Gantt. For 30%, a rule-based status + a
  simple statistical relation (correlation/trend) is correct. Don't fake a prediction.
- Live **patient vitals** can't be real yet (no sensors/patient). Keep the single
  reference recording for vitals; make the **air** side live. That's honest and matches
  the single-patient design.

---

## B. Direct answers to the open questions

- **Live vs stale — what's used now:** Air quality currently runs on a **STATIC saved
  dataset** (PLOS Lahore 2024 hourly). The **OpenAQ live pull works** but was toggled
  off (that station returned stale/no-PM10 data). Vitals (SpO₂/HR/resp) come from one
  **BIDMC recording** (static). Spirometry FEV1/FVC from **NHANES** (static).
  → For 30%, wire **one real live air source** (Punjab EPA AQI or OpenAQ) and show a
  clear LIVE/STATIC badge. The new app.py already shows that badge (`DATA_MODE`).
- **Where FEV1/FVC fits:** it's the **deterministic baseline/anchor**. We don't relate
  FEV1/FVC *directly* to live air; we use it as the fixed reference the patient's live
  readings (SpO₂, resp) are judged against. "Deviation from baseline under exposure" =
  the risk signal. (Spirometry is periodic, not streamed.)
- **The relation (our focus):** as PM2.5 rises, SpO₂ falls / resp rises. Shown three ways
  in the app: fusion time chart, scatter + correlation r, and plain-language insights.
- **Our contribution ("fanqari"):** fusing home air quality + this patient's vitals +
  their lung baseline into ONE interpreted, person-specific view with a status — which
  no generic AQI dashboard does. That's the gap.

---

## C. Step-by-step for the week (priority order)

1. **[DONE] Upgrade the dashboard to patient-centric** — new `app.py`:
   patient header + status badge, interpretation layer, fusion + scatter(r),
   FEV1/FVC anchor, DATA MODE badge, interactive signal explorer, marked MODEL SLOT.
   Run it on the real data via `data_loader.build_dataset()`.
2. **Wire ONE live air source** (makes "live-ready" real):
   - Easiest real live for Lahore: **Punjab EPA AQI** (aqi.punjab.gov.pk) or **OpenAQ v3**.
   - In `data_loader.py`, add `fetch_live_air()` returning `timestamp, pm25, pm10`;
     set `DATA_MODE="LIVE"` in app.py to use it. Keep STATIC as fallback.
3. **Interpretation polish** — make sure every number has a "what it means" line
   (the `build_insights()` function). Add a short "what should the patient do" line
   (generic health advice by AQI band) — mirrors the govt site but **personalised**.
4. **Model slot (lightweight, honest)** — replace the rule in `patient_status()` with a
   simple fitted relation (e.g. linear/logistic on PM2.5→SpO₂ drop) and label it clearly
   as the first model; note full prediction model = next phase. This shows "model lagaya hua".
5. **A comparison element** — small panel: "Generic city AQI vs this patient's response"
   so the novelty is visible side by side (what he asked: your trends vs the website's).
6. **Screenshot + short script** for the demo; confirm it runs clean end-to-end.
7. **(Parallel, lighter)** start the FYP-1 report draft: functional/non-functional
   requirements + UML/DB diagrams. Format comes later — don't block implementation on it.

**Minimum to pass the 30% bar:** steps 1–4 (patient-centric interpreted dashboard,
one live air source, the air↔respiration relation visible, a labelled model slot).

---

## D. Data shapes (for data_loader.py — unchanged target)
`df`: `timestamp | pm25 | pm10 | spo2 | heart_rate | resp`  +  scalars
`fev1_fvc_baseline (%)`, `fvc_ml`, `fev1_ml`, `obstruction_flag (bool)`.
BIDMC = one `bidmc_01_Numerics.csv` recording (the single patient).
