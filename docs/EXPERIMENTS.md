# Phase 5: Double DQN vs DQN

One change against the Phase 4 raw-reward agent: `--double` makes the online net pick the next placement and the target net value it (`training/dqn.py`, `update`). Everything else is identical: 100k steps, same hyperparameters, raw reward.

Dueling heads were skipped on purpose. The network already outputs one value per afterstate, so there are no per-action advantages to separate from a state value.

Three training seeds per variant (`--seed 0,1,2`, which sets the network init and exploration; the training game seeds are the same for every run). Seed 0 of plain DQN is the `raw` run from Phase 4. Each run is evaluated with its best-validation checkpoint on the baseline seeds 1-100, greedy, 3000-piece cap. The cap is raised from 1000 because at 1000 pieces every agent was already at the cap (`docs/DQN.md`).

```
python train.py --name ddqn_s0 --seed 0 --double
python evaluate.py checkpoints/ddqn_s0_best.pt --max-pieces 3000
go run ./cmd/baseline -games 100 -max-pieces 3000     # heuristic
```

| Agent | Training seed | Avg score | Avg lines | Avg pieces |
|---|---|---|---|---|
| DQN | 0 | 173983.0 | 1173.3 | 2948.3 |
| DQN | 1 | 172449.0 | 1195.8 | 3000.0 |
| DQN | 2 | 171587.0 | 1170.3 | 2938.4 |
| Double DQN | 0 | 173757.0 | 1184.5 | 2972.7 |
| Double DQN | 1 | 172417.0 | 1179.8 | 2962.1 |
| Double DQN | 2 | 173742.0 | 1173.8 | 2948.6 |
| Heuristic | - | 119880.0 | 1098.8 | 2764.2 |

Means over the three seeds: DQN 172673 score, 1179.8 lines, 2962.2 pieces. Double DQN 173305 score, 1179.4 lines, 2961.1 pieces.

## Reading

- Double DQN made no measurable difference: +632 score (0.4%) against a spread of about 2400 between seeds of the same variant, and lines and pieces are equal to within noise. Three seeds cannot detect an effect this small, and I do not claim one.
- Both variants beat the heuristic by about 44% in score at this cap. The heuristic's pieces are lower mainly because 17 of its 100 games ended before the cap; the learned agents rarely did.
- A likely reason for the null result (not tested): the afterstate value is a function of only four features, so there is little room for the overestimation bias that Double DQN corrects.
- The score is still limited by the 3000-piece cap, and the agents' scores are close to each other. Telling better agents apart would need a different measure, such as score per piece or a higher cap.
