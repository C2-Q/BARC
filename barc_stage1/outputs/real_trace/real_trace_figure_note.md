# Real-Trace Figure Note

- Plotting was refactored into shared publication-style utilities for trace characterization, capacity sensitivity, buffer sensitivity, and cross-workload summary views.
- A burst is defined deterministically as a timestep whose demand is at least `ceil(mean(trace) + std(trace))`, clipped to be at least 1.
- Burst length is the number of consecutive timesteps that stay above that threshold.
- Critical capacity is the smallest scanned capacity `C` at the representative buffer `B*` where normalized execution time is at most 1.0.
- Critical buffer is the smallest scanned buffer `B` at the representative capacity `C*` where normalized execution time is at most 1.0.
- Stall saturation buffer is the smallest scanned `B` at `C*` where stall cycles reach their minimum finite value over the scanned range.
- `B*` and `C*` are chosen deterministically as the scan slices with the largest policy separation, prioritizing feasibility differences.
- A value of `-1` in a threshold summary means the threshold was not reached within the frozen scan range.
