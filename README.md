AeroMinds — PM2.5 Forecasting (AISEHack Theme 2)
Team Members
Dhanush T (Dhanushthiyaree06)
Kavin M (kavin182005)
Kabilan K (suryaprakash01)
Dharun V (dharunvelmurugan06)
Samariya Rayar S (samariyarayar04)

Problem Statement
Country-level PM2.5 concentration forecasting over India (140×124 grid).

Input: 10h PM2.5 history + 10h meteorological features (Phase 2)
Output: 16-hour PM2.5 forecast
Metric: Weighted combination of GlobalSMAPE, EpisodeCorr, EpisodeSMAPE

Our Approach — PTM-FNO (Physics-informed Temporal Model with Fourier Neural Operator)
Architecture

FNO Block: SpectralConv2d (modes=7, width=8) for global spatial frequency learning
U-Net: Encoder-decoder (64/128/256 channels) with skip connections for multi-scale spatial features
AR Decoder: Autoregressive delta prediction using last-known meteorology
Direct Head: Non-autoregressive 16-step direct forecast
Output: 0.6 × AR + 0.4 × Direct blend

Key Design Decisions
DecisionReasonSmall model (W=8)4 months training data — bigger models overfitAUX_WIN=10Phase 2: no future met available15 input features8 met vars + 7 chemical precursorsGroupNorm(8)Stable at batch_size=4AMP disabledFNO uses torch.cfloat — incompatible with GradScalerTemporal val splitPrevents leakage from overlapping windows
Phase 2 Loss Function
Loss = SMAPE_WT × EpisodeWeightedSMAPE + PHYSICS_WT × AdvectionLoss

Episode cells upweighted 2× in loss
IGP hotspot (rows 68-94, cols 39-80) upweighted 2×
Physics advection constraint prevents unphysical gradients

Results
PhasePublic LBRankPhase 112.562 (RMSE)10thPhase 20.8745TBD
What Didn't Work

Larger FNO (W=12,24,48) → overfits on 4-month dataset
Emission variables (NOx, SO2, NH3) → year-specific to 2016, hurt 2017 test
Anomaly prediction → test month distribution unknown
SWA → averaged good epochs with overfit epochs
Vertical flip TTA → India's N-S pollution gradient makes vflip invalid
Coordinates + correlation loss → score dropped from 0.8745 → 0.70

Repository Structure
├── notebooks/
│   ├── first.ipynb          # Best Phase 2 notebook (score: 0.8745)
│   ├── first(1).ipynb    # Best Phase 1 notebook (score: 12.562)
├── README.md
└── LICENSE
