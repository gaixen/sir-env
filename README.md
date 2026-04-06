# Pandemic Policy Control (OpenEnv)
An RL environment where an AI agent acts as a government health authority,
making daily NPI (non-pharmaceutical intervention) decisions during a
simulated COVID-19 epidemic. The agent must balance epidemic suppression
and economic stability which is a genuine multi-objective policy problem.
Built on the SIR-RL framework (arXiv:2404.08423v1), calibrated to US
COVID-19 data (May 2020 – Oct 2022).

## Overview

This project frames pandemic policy-making as a sequential decision problem: an automated agent selects real policy levers (school closures, workplace rules, travel limits, mask guidance, targeted support measures, etc.) day-by-day to manage infections while limiting economic harm. In the problem statement, every action maps to a realistic policy instrument, the observations are derived from cleaned, interpolated public datasets (cases, deaths, vaccinations, and GDP), and the simulator couples an SIR-inspired epidemic model with empirically fitted economic effects. Tasks reflect real-world objectives (flattening peaks, balancing health and economy, and managing transitions during vaccination rollouts) and graders measure performance with interpretable health/economic/stability components. The environment therefore supports research and evaluation of policies under realistic constraints, noisy data, and multi-objective trade-offs rather than stylized or trivial scenarios.

## Why This Environment Matters

Real policy teams must make sequential trade-offs under uncertainty:
- Health safety: suppress transmission and avoid hospital-overflow regimes.
- Economic continuity: avoid excessive GDP collapse from prolonged restrictions.
- Transition management: shift from emergency controls to vaccination-era recovery.

This benchmark captures those competing objectives with dense trajectory reward and task-level grading.

## Data Acquisition and Cleaning
We build the dataset from public and curated sources and then clean and align them into a single daily US timeseries. Case and death counts, policy response indicators, and vaccination data are fetched from Our World in Data (via `owid.catalog.fetch`) and converted to pandas; GDP is read from the local `data/gdp/GDP.csv`. Each source is reindexed to a common daily date range and missing values are filled by time-based interpolation (GDP is interpolated using a low-order polynomial where appropriate), with forward/backfill used for any remaining gaps. Unnecessary columns are dropped, the date range is clipped to the overlap across all sources, and the frames are inner-joined on date; rows missing core fields (for example `total_cases`) are removed. The final combined file is written to `data/combined_us_data.csv` and is the canonical input used by the environment.

## Why is this relevant?
Pandemic policy is one of the most consequential real-world decision problems
governments face. Existing RL benchmarks don't capture:

- The temporal tradeoffs between health and economics
- Phase transitions (pre-vaccine - rollout - endemic)
- Multi-lever policy spaces with ordinal, interacting controls
- Dense reward tied to real epidemiological mechanisms (SIR model)

This environment fills that gap, providing a benchmark grounded in real
epidemiological dynamics and validated against actual COVID-19 data.

## OpenEnv API

Endpoints:
- `POST /reset`
- `POST /step`
- `GET /state`
- `POST /grade`

Typed models are implemented in `train/environment/openenv_service.py`:
- `PandemicPolicyAction`
- `PandemicPolicyObservation`
- `PandemicPolicyReward`
- `ResetResponse`, `StepResponse`, `StateResponse`, `GradeResponse`

## Action Space

Discrete policy levers:
- `school_closing` (0-3)
- `workplace_closing` (0-3)
- `cancel_public_events` (0-2)
- `restrictions_on_gatherings` (0-4)
- `close_public_transport` (0-2)
- `stay_at_home` (0-3)
- `internal_movement` (0-2)
- `international_travel` (0-4)
- `facial_coverings` (0-4)
- `vaccination_policy` (0-5)
- `income_support` (0-2)
- `debt_relief` (0-2)

## Observation Space

- `susceptible`, `infected`, `recovered` in `[0,1]`
- `re` (effective reproduction number)
- `new_cases_norm`
- `weekly_growth_rate`
- `gdp_index` in `[0,100]`
- `stringency_index` in `[0,100]`
- `vaccinated_pct`, `fully_vaccinated_pct` in `[0,100]`
- `day`
- `phase` (`pre_vaccine`, `rollout`, `endemic`)

## Tasks and Graders

Three deterministic tasks with score in `[0.0, 1.0]`:
- `flatten_curve` (easy): prioritize low infection and low `Re`.
- `balanced_response` (medium): harmonic health/economy trade-off with stability.
- `optimal_transition` (hard): pre-vaccine suppression, rollout progress, post-rollout recovery.

Each task has a `success_threshold` in `openenv.yaml` and a programmatic grader in `PandemicPolicyOpenEnv.grade()`.

## Reward Design
Dense step reward (clipped to `[0,1]`) combines several interpretable subcomponents to give a rich, shaped learning signal at every timestep rather than only terminal feedback:

- **Health component:** captures epidemiological outcomes such as normalized new cases, the effective reproduction number `Re`, and overflow risk relative to hospital capacity. Policies that reduce infections and keep `Re` < 1 increase this term; large spikes or sustained high incidence produce sharp penalties.

- **Economic component:** measures short-term economic impact using the `gdp_index` (interpolated daily) and its recent trend. Large sudden drops in GDP or prolonged decline reduce this component, while policies that preserve GDP recovery improve it.

- **Stability component:** rewards policy consistency and penalizes excessive oscillation. Frequent large swings in NPIs (for example toggling between open/closed repeatedly) incur negative stability penalties; smooth, gradual adjustments are favored.

- **Support component:** credits targeted social/economic support actions (income support, debt relief) that mitigate negative economic/humanitarian impacts of restrictive NPIs. This term encourages using compensatory measures when restrictive policies are necessary.

- **Penalty component:** applies explicit penalties for catastrophic outcomes (health system overflow), unrealistic or impossible actions, and repeated non-productive behaviors (e.g., locking in a policy that yields no marginal benefit). It also encodes small step costs for high-intensity interventions to reflect real implementation burdens.

Implementation details: each subcomponent is computed from normalized observables (so every sub-score lies in a bounded range), then combined with pre-defined weights chosen to reflect the intended trade-offs for the task. The per-step reward is the weighted sum of subcomponents, clipped to `[0, 1]`. Trajectory-level scoring (returned by the grader) aggregates step rewards and applies task-specific success transforms to produce a final task score in `[0, 1]` (for example, combining cumulative health/economic performance and additional checks such as vaccination progress). This design gives agents both immediate feedback for incremental improvements and an interpretable decomposition to analyze policy behavior.

## Local Setup

Use your existing conda env:

```bash
conda activate myenv
pip install -r requirements.txt
```

Run API server:

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

Quick checks:

```bash
curl -X POST http://127.0.0.1:7860/reset -H "Content-Type: application/json" -d '{"task":"flatten_curve"}'
curl http://127.0.0.1:7860/state
```

## Baseline Inference

The required script is at repository root: `inference.py`.

It:
- Uses the OpenAI client.
- Reads `API_BASE_URL`, `MODEL_NAME`, and API key from `HF_TOKEN` (fallback `OPENAI_API_KEY` / `API_KEY`).
- Runs all 3 tasks.
- Emits strict logs in `[START]`, `[STEP]`, `[END]` format.

Run:

```bash
conda activate myenv
python inference.py
```

## Reproducibility and Runtime

- Deterministic reset seed is used per task in baseline script.
- Reward and grader logic are deterministic.
- Designed to run on 2 vCPU / 8GB memory constraints.

## Docker

Build and run:

```bash
docker build -t pandemic-policy-openenv .
docker run --rm -p 7860:7860 pandemic-policy-openenv
```

## Hugging Face Space

Use this repository as a Docker Space and keep `openenv` tag in metadata (`openenv.yaml`).

## Validation

You can run the provided validator script:

```bash
bash train/environment/validation.sh <your_space_url> .
```
