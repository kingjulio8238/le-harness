# Network Volume Layout

Everything is on the network volume (`/workspace/data/`). Here's what persists across pod sessions:

```
/workspace/data/                              # Network volume (persistent)
├── pusht_expert_train.h5                     # 44GB dataset
├── pusht/
│   ├── lejepa_object.ckpt                   # 69MB checkpoint
│   ├── lejepa_weights.ckpt                  # 69MB checkpoint
│   ├── pusht_results.txt                    # LeWM eval metrics
│   └── rollout_0..49.mp4                    # 50 LeWM rollout videos
└── results/
    ├── phase0_lejepa.log                    # LeWM eval log
    ├── phase0_lejepa_pusht_results.txt      # LeWM metrics (copy)
    ├── phase0_random.log                    # Random eval log
    ├── phase0_random_pusht_results.txt      # Random metrics
    ├── phase0_timing.log                    # Timing data
    ├── phase1_fidelity.json                 # Fidelity audit results (JSON)
    ├── phase1_fidelity.log                  # Fidelity audit log
    └── random_videos/                       # 50 random rollout videos
```
