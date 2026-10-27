# October, degenerate validation window (archived)

Ingest window **2025-10-03 .. 2025-11-01**, panel 10 Oct – 1 Nov.

Kept as the demonstration that validation-split *composition*, not the
partition or the optimiser, drove the FedAvg instability seen here.

## Why this window is degenerate

Its two validation days are the only ones in any window with no ordinary
weekday:

| window | validation days | atypical |
|---|---|---|
| jan | Fri 24 Jan, Sat 25 Jan | 1 of 2 |
| jul | Thu 17 Jul, Fri 18 Jul | 0 of 2 |
| **oct (this)** | **Sun 26 Oct, Mon 27 Oct — Republic Day observed** | **2 of 2** |

The offset window was chosen to keep the test week holiday-free; it did so
by pushing the observed holiday into a two-day validation split. Model
selection then optimised against a Sunday and a public holiday while the
test week was Tue–Fri.

## What that produced

`BestParameterTracker` selected round 1 or 2 in 5 of 9 route runs with a
stored curve (January and July: 0 of 9, earliest minimum round 16).
For the random partition at seed 1:

    val min 12.61 @ round 1  -> test there 20.12
    test min 10.69 @ round 38
    cost of selecting on validation: +9.43 MAE

FedAvg trained correctly — the round-40 model scores 10.99 on test, better
than seeds 0 and 2 — so the reported 20.25 is entirely selection cost.
SGD+momentum and Adam at lr/3 fail identically, ruling out optimiser and
learning rate.

## Contents

- seeds 0–2, 10 partitions x 3 arms each
- `ablation.csv`, `summary.csv` and `paper_numbers.*` summarise these seeds.

Superseded by the `oct` tag, ingested 2025-09-22 .. 2025-10-21 (panel
29 September - 21 October). Its validation split is Wed 15 Oct + Thu 16 Oct.
Republic Day (25 Oct) and its observed Monday (27 Oct) fall outside the
window entirely, as January's holidays fall in its discarded warm-up.
A second candidate, 6 Oct - 4 Nov, was rejected because 3 November is
absent from the export (`03.11.2025вал..csv` contains 2 November), which
would have produced a 22-day panel.

Under the Check 5 audit (`src/split_audit.py`) this window returns FAIL:
its validation split contains no ordinary weekday while its test split does.
