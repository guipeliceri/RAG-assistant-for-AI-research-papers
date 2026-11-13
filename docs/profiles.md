# Profiles

A profile describes what a reader wants to see. The **Profiles** tab lists every new paper that
matches it, not only the top few. Anyone can add one by pull request.

## Add a profile

1. Copy `profiles/frontier-models.md` to `profiles/<your-slug>.md` (lowercase letters, digits and
   dashes).
2. Fill in the front matter and the four sections:

   ```markdown
   ---
   name: Computer vision for medicine
   categories: [cs.CV, eess.IV]
   maintainer: your-github-login
   ---
   ## Description
   One paragraph describing the papers you want. This is what papers are matched against.

   ## Includes
   - one topic per line

   ## Excludes
   - topics that look close but you do not want

   ## Examples
   relevant: 2501.01234, 2502.04321
   not_relevant: 2503.00001
   ```

3. Open the pull request. The **profile check** workflow fails if the file does not parse or if
   the pull request changes anything outside `profiles/`. A profile never runs code; it is only read.

Only categories that are ingested produce results; today that is `cs.CL`. A new profile appears
with the tag "not evaluated" until it has human labels.

## How matching works

1. Papers outside the profile's categories are dropped.
2. Optionally, papers far from the description (embedding similarity below a threshold chosen on
   dev labels so that 98% of relevant dev papers are kept) are dropped without an LLM call.
3. An LLM reads each remaining title and abstract against the description, includes and excludes.
   Borderline papers are included, because a missed paper costs the reader more than an extra one.

## How a profile is evaluated

The goal is an honest number, so the procedure is fixed before anyone looks at results.

- **Sample.** 200 papers from the last 30 days, 50 from each similarity band (top 5%, 5 to 20%,
  20 to 50%, bottom half). Without bands, a random sample would hold too few relevant papers to
  measure recall. Each paper carries a weight (band size over papers drawn from the band), so the
  metrics describe the whole stream, not the sample. The bottom band catches relevant papers that
  are far from the profile's wording.
- **Blind labelling.** The labeller sees title and abstract in random order, with no band, score
  or prediction.
- **Dev and test.** Each labelled paper is dev or test by a hash of its ID. Profile text, includes,
  excludes, the borderline rule and the threshold are only ever adjusted on dev. Example IDs in
  the profile are never evaluated, since the classifier sees them.
- **Lock and ledger.** Test labels are frozen with a fingerprint; a changed label stops the
  evaluation. Every test run is logged, and the app shows test metrics only for the exact profile
  version they were measured on.
- **Baselines.** Every report includes "always yes" (precision equals the share of relevant
  papers) and "similarity only" (no LLM, threshold set on dev for 90% recall).
- **Intervals.** Precision and recall come with 95% intervals from a bootstrap within bands.

Run it from the Actions tab with the **profile labels** workflow: `sample`, then label
`eval/profiles/<slug>/to_label.csv` into `labels.csv`, then `dev`, `tune`, `lock`, `test`.
