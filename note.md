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
  * have pd dataframe to have prev_taken T/F
  * xor it with current taken T/F
  * if 0, no transition, else transition
  * update transition count
  * how can I do Transition rate vs accuracy heatmaps? 

- any way to tell if its weak in local / global history ??
---
Some perf improvements:

- 88.15% increase in simulation throughput (after also vectorizing global stats)
  - after adding per predictor stats sim throughput decreased 31% (nooo)
    - after fully vectorizing per predictor stats, sim throughput decreased only 11% (nice)
- vectorized access doesn't account for duplicated index values. so simple increment doesn't work (need to use np.unique instead)
- BATCH size matters! 10,000 -> 1_000_000 :: 19% increase in bandwidth
- after removing unecessary numpy datatype casting, there seem to be a very small but noticeable(ish) amount of BW increase (.4% lol)
- 3% decrease in throughput after adding transition data per address
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