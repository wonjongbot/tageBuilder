# Notes and roadmap

## Note
Track challenges n add notes for comparisons for optimizations

adding individual address tracker -> +10 seconds ish (30% increase!!)
    -> need to optimize (use np vectorization? then I need to change how I store reports)

memory map heatmap (bar with heatmap -> hotter == more mispredictions among instructions in this area)

every machine is different
    -> progress queue acting weird if put in global on mac (ok ubuntu)

- [x] change directory strings automatically
- [x] track per address stats individually and store as csv
  - [x] do top n calcualtion after df conversion (easier to do)
- [ ] mpki update per branch address? (track lost sim BW)
  - [ ] what would be a better way to calculate each branches' contribution to mpki (accuracy is bad since it doesn't account for frequency)
  - [x] use pandas df? (do it and capture perf gain or loss) 

try except catch to delete incomplete tests?

---
Todos

- [x] df test plot
- [x] change per address scoreboard into df bulk update using vectorization
- [x] integrate df test plot into code
- [x] add more class base stats (mean median plot, class frequency pie chart, total mispredictions)
- [ ] logic for storing transitions

- any way to tell if its weak in local / global history ??
---
Some perf improvements:

- 88.15% increase in simulation throughput (after also vectorizing global stats)
  - after adding per predictor stats sim throughput decreased 31% (nooo)
    - after fully vectorizing per predictor stats, sim throughput decreased only 11% (nice)
- vectorized access doesn't account for duplicated index values. so simple increment doesn't work (need to use np.unique instead)
- BATCH size matters! 10,000 -> 1_000_000 :: 19% increase in bandwidth
- after removing unecessary numpy datatype casting, there seem to be a very small but noticeable(ish) amount of BW increase (.4% lol)
---

## Recommended Next Steps (Short-Term Milestones)

Below are prioritized tasks from your to-do list, with suggestions on how to group them into “phases” so you can post incremental results online. Each phase has a natural stopping point that could be a separate blog post, GitHub release, or progress report.

### Phase 1: Enhance Reporting & Visualization

1. Storage Visualization
   - [x] Create a stacked bar chart or pie chart showing storage budget by component (global history, per-table entries, tags, etc.).
    > This is both an excellent “visual wow” feature and relatively quick to implement with matplotlib or seaborn.
2. Expanded Statistics (Incorrect Predictions per Branch Address)
   - [x] Track mispredictions per PC (or branch site) and store them in CSV/JSON.
   - [x] Visualize “top-K worst offenders” as a bar chart or a cumulative fraction plot (e.g., top 10 PCs cause X% of mispredictions).
     - [x] have a graph for accuracy per address (implementation done, integration needed)
     - [x] have another graph for combining top-n offenders
   - Suggested Additional Stats:
     - [x] Per-branch dynamic frequency (how often a branch is executed). (probably class is more useful)
     - [ ] Transitional behavior (the ratio of taken-to-not-taken transitions).
     - [ ] Correlation metrics (e.g., how often a certain set of bits in global history correlates with a branch outcome).
     - [ ] Confidence histogram for incorrect predictions
       - [ ] needs to track confidence (hystersis) per prediction
     - [x] track class per PC ([Type]+[Mode]+[Cond])
       - [x] MPKI/Accuracy per class
       - [ ] Confidence distribution per class
       - [ ] Transition rate vs accuracy heatmaps
       - [x] workload analysis
         - [x] pie --radar-- charts for instruction type

> Why Phase 1 first?
>
> You get immediate, tangible additions without drastically changing the codebase. These improvements boost the “eye test” value of your project and demonstrate deeper analysis, which is compelling for anyone reading your blog or GitHub.
>
> (Natural Stopping Point:)
>
> Release a blog post or GitHub README update showcasing the new charts and stats. This is a nice place to pause because you’ll have a polished visualization suite for the TAGE predictor.

## Phase 2: Energy/Power Insights

1. Energy/Power Estimation
   - Implement a simplistic bit-flip counter for key operations (accessing tables, updating counters, updating tags).
   - Estimate energy from bit-flips + toggling activity (some known constants or rough estimates from the literature).
   - Output these power/energy stats in your JSON/CSV reports and create a simple bar or line chart.

>Why Phase 2 next?
>
>Energy/power data is a big draw for hardware enthusiasts and researchers alike—branch predictors significantly affect power in modern pipelines. It also demonstrates that your simulator can capture more than just correctness metrics.
> 
>(Natural Stopping Point:)
>
>Another blog post or release, highlighting energy/power overhead results, with sample analysis on how “bit-flip heavy” certain workloads are.

### Phase 3: Advanced UI & Reporting

1. HTML Reports
   - Convert your JSON output into a more user-friendly, clickable HTML page with tables/plots embedded.
   - Possibly integrate a small Python server or use something like Jupyter Notebooks for easy demonstration.
2. Basic Web Interface
   - Offer a minimal Flask or Streamlit web front-end to browse different simulation results, queue new runs (upload spec or trace), and view visualizations interactively.
   - This step can be as large or small as you want—just serving static HTML + your CSV/JSON can be enough for a “live” feel.

>(Natural Stopping Point:)
>
>A big project milestone where you show an interactive interface or at least an HTML-based summary page. This is a substantial chunk of work, so it’s ideal to group HTML reporting and a basic web view together.

Phase 4 (Optional): Additional Predictors & More Complexity
	1.	Add Another Advanced Module
	•	For example, a perceptron-based predictor or a statistical correlator.
	•	Evaluate how it differs from TAGE in terms of storage, accuracy, power, etc.
	•	Caution: This can quickly lead to “feature creep,” so only do it if you want to highlight a truly novel design or compare TAGE vs. perceptron side by side.
	2.	Machine Learning Angle
	•	If you have a strong ML background, exploring “online training” of a small NN-based predictor or an LSTM-based approach is advanced.
	•	Likely more of a master’s or even a small PhD-level mini-project, depending on how thorough you are.

(Natural Stopping Point:)
After adding a second predictor type, you can publish a major comparison study. This would cap your project at a truly advanced level suitable for a graduate-level portfolio or a strong blog series.

1. Overall Assessment of Project Level
	•	Undergraduate / Senior Project: If you stop around Phase 2 (improved visualization + basic power estimation), you’ve already built something quite impressive, beyond typical class assignments.
	•	Master’s Project / Thesis: Including Phase 3 (interactive web UI, sophisticated reporting) plus some novel analysis or a second advanced predictor design would place the project squarely in master’s territory.
	•	PhD-level: Typically requires a novel research angle (e.g., new predictor algorithm or formal proof of predictor behavior). If you use this platform to experiment with brand-new predictive methods and publish academically, that can push it into the PhD realm—though it depends on novelty and depth, not just the code.

2. Tips to Avoid Feature Creep
	1.	Define Clear Scope for Each Phase: Have a short, well-defined goal (e.g., “Show misprediction hot spots per PC and produce stacked bar chart for storage usage”). Finish that before moving on.
	2.	Keep the Code Modular: So new features (power estimation, web interface, or new predictor modules) can be added without rewriting everything.
	3.	Document & Release Iteratively: Each phase can become a “release” or a “blog post.” If you keep your audience updated in these small increments, you’ll have multiple portfolio pieces and maintain momentum.
	4.	Weigh ROI: Some tasks require major design changes (e.g., adding a new predictor type) but may yield less immediate “wow” than, say, interactive visualizations. Always consider cost vs. payoff.

3. Concluding Remarks

You’ve already laid a strong foundation with your TAGE-based simulator, parallel execution, and multi-format reporting. By prioritizing:
	1.	Enhanced Visualization and Stats
	2.	Energy/Power Estimation
	3.	HTML / Web-based Reporting
	4.	(Optionally) Another Advanced Predictor or ML-based approach

you’ll have a compelling, high-level project that stands out. Each phase naturally yields something to show off—perfect for a blog series, GitHub release cycle, or a section in your resume/portfolio.

Staying mindful of scope ensures you maintain the “wow” factor without drowning in complexity. Good luck, and have fun shaping this into a flagship project!