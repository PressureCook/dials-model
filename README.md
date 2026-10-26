# Dials: a three dial model of psychopathy

My working idea: the traits we group under psychopathy are not a separate category. They are settings on three dials everyone has.

| Dial | Low setting looks like |
|---|---|
| **Reward** | not much drive to chase gains |
| **Fear** | punishment and danger don't land |
| **Guilt** | other people's pain doesn't register |

Psychopathy is the corner where fear and guilt are both low. Temperament sets where your dials start, and upbringing nudges them from there.

This repo is the measurement side: can you actually read someone's dials from data?

## What's in here

| Path | What it does |
|---|---|
| `01_factor_analysis.py` | Factor analysis on real Short Dark Triad (SD3) answers. How many dimensions does the data really support? |
| `02_proto_dials.py` | First pass at turning SD3 items into dial scores, with reliability checks |
| `03_igt_norms.py` | Pools healthy control scores from an Iowa Gambling Task meta analysis into norms |
| `04_calibrate_agent.py`, `dial_agent.py` | A learning agent with dials that plays the Iowa Gambling Task, calibrated to match those norms |
| `05_recovery_igt.py` | Can you get the dials back out of task behaviour? (Parameter recovery, plus the alternative explanations for the same score) |
| `06_mixed_gamble_recovery.py` | A 50/50 gamble task that separates the reward dial from the fear dial |
| `07_guilt_agent.py` | A harm aversion task (after Crockett et al., 2014) for the guilt dial |
| `ai/` | A small PyTorch network that combines questionnaire, gambles and harm task into one dial reading with uncertainty, trained on synthetic people built from the real SD3 measurement model |
| `ai/12_upbringing_simulation.py`, `ai/13_upbringing_predictor.py` | How much of your adult dials can upbringing explain, using effect sizes from the literature |
| `app/dials-lab.html` | The browser version: take the tasks, see your three dials. No server needed |
| `reversal-task/` | A reversal learning task and modelling pipeline, the first prototype |

## Run it

```
pip install numpy pandas scipy scikit-learn factor_analyzer torch openpyxl
python 02_proto_dials.py
python 06_mixed_gamble_recovery.py
```

Open `app/dials-lab.html` in a browser to try the tasks.

## Data

`sd3_numerical.csv` is the public Short Dark Triad dataset from Open Psychometrics. The IGT meta analysis export used by `03_igt_norms.py` is not included; `igt_norms.csv` is the pooled result.

## Honest status

The task to dial links in `ai/synth.py` are assumptions, flagged as such in the code, until there's pilot data from real people. This is a hypothesis with tooling, not a diagnostic.
