# Dials project

A small research toolkit for testing one idea: that personality differences, including the traits grouped under psychopathy, are better described as settings on a few shared dials (reward sensitivity, fear, guilt or empathy) than as separate categories. Upbringing and environment are assumed to shift those settings from a starting point set by temperament.

The toolkit has two parts. Both are prototypes meant to be changed.

## Files

* `reversal_task.html` is a browser task that measures reward and punishment learning. Open it by double clicking. No server and no internet needed.
* `pipeline.py` is the modelling code. It checks how many dimensions a dataset supports and compares a simple model against a complex one on held out data.
* `README.md` is this document.

## Setup

1. Install Python 3.10 or newer.
2. Install the libraries: `pip install numpy pandas scikit-learn`
3. Run the demo: `python pipeline.py synthetic`

For real questionnaire data, download the Big Five dataset from Open Psychometrics (the file is called `data-final.csv`, tab separated) and run: `python pipeline.py ipip data-final.csv`

## The reversal learning task

**What it does.** Two cards are shown on every round. One is the better card and pays out on 80 percent of picks, the other on 20 percent. After eight consecutive picks of the better card, the roles swap silently. Winning gives 10 points and losing takes 10.

**Why this task.** Reversal learning is a standard way to measure reward sensitivity and how hard someone finds it to let go of a strategy that has stopped working. Slow adaptation after a flip (perseveration) and the pattern of staying after wins versus switching after losses are the two measures most often linked to reward and punishment sensitivity.

**Settings** sit at the top of the script: `N` (rounds, default 80), `P_BEST` (0.8), `CRIT` (8 picks before a flip) and `ITI` (pause after feedback in milliseconds).

**Output.** Two downloads at the end of a session.

1. A CSV with one row per trial: participant, trial, choice (0 blue, 1 gold), best_option, chose_best, rewarded, rt_ms, reversal_after_trial, timestamp.
2. A JSON file with the same trials plus the task parameters and a summary: accuracy, win stay rate, lose shift rate, reversals reached, mean wrong picks after a flip, median response time and total points.

**Known limits.** Browser timing is accurate to a few milliseconds, which is fine for this task but not for reaction time work needing more precision. People can work out the rules by thinking, so scores reflect strategy as well as sensitivity. The task covers the reward dial only. Fear and distress response need their own tasks (see roadmap).

## The pipeline

1. **Load data.** Synthetic mode simulates three hidden dials, a small environmental influence on them, noisy questionnaire items and a noisy informant rating. IPIP mode loads real Big Five responses and drops rows with missing answers.
2. **Parallel analysis** compares each eigenvalue of the item correlation matrix against random data. Factors that beat chance are kept. This answers how many dials the data supports.
3. **Factor analysis** with varimax rotation shows which items belong to which factor and produces a score per person.
4. **Model comparison** with five fold cross validation checks three models on held out data: a mean only baseline, a simple ridge regression on a few features and a gradient boosting model on everything. Scores are R squared. Zero means no better than guessing the average.

The synthetic run recovers the three dials correctly (fear, reward and guilt items group into three factors), and the simple model beats the complex one, which is what the junk columns are there to demonstrate. Because the data is simulated, this checks that the code works. It says nothing about people.

## Data format for real studies

One row per participant, joined on a participant code. Suggested columns:

* Self report items, one column per item, plus the questionnaire name.
* Informant ratings on the same items, one row per informant.
* Task summaries from the reversal task, plus the raw trial files kept separately.
* Environment: adversity count, parental warmth, age, gender, country, and anything else you plan to test, defined in advance.

Keep raw trial data. Better methods can be applied to it later.

## Ethics and data protection

1. Participants must give informed consent. The task page contains a short consent line, but a real study needs a full information sheet stating what is collected, why, who sees it and how long it is kept.
2. Under UK GDPR, information about mental health and childhood experiences is special category data. Use participant codes instead of names and store any code to identity key separately from the responses.
3. Participants can withdraw and have their data deleted. Say so and make it easy.
4. Questions about childhood adversity can be upsetting. End the study with support resources.
5. Never tell a participant they are or are not a psychopath. Show dial profiles in plain language and say it is not a diagnosis.
6. If you plan to publish or work with a university, you will need formal ethics approval before collecting data.

## Roadmap

1. Run the pipeline on the IPIP data and read the loadings to get a feel for the method.
2. Pilot the reversal task on yourself and a few friends. Check timing, wording and whether the summary measures look sensible.
3. Add a second task for distress response (emotion recognition with fearful faces) and a third for fear learning.
4. Add short questionnaires (check licences first) and an informant rating link.
5. Write your hypotheses down before collecting data, for example that win stay and lose shift rates should correlate with informant rated impulsivity more strongly than with self report.
6. Recruit 30 to 50 people. Check whether task measures and questionnaire measures separate into different dimensions.
7. Only after that, add the model from environment to dials, always compared against a simple baseline on held out data.

## Reading list

* Gray's reinforcement sensitivity theory (reward and punishment sensitivity).
* Kochanska's work on fearfulness and conscience in toddlers.
* Blair's work on distress response and psychopathy.
* Hare's Psychopathy Checklist and the TriPM for measuring psychopathic traits.
* Gelfand on tight and loose cultures, and the Global Preferences Survey, for the population comparison idea.
