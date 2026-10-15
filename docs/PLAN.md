# Tetris ML: Project Plan

**Goal:** Learn reinforcement learning by building a Tetris engine in Go, training an agent with PyTorch, and documenting the whole process in a GitHub repo and a portfolio blog post.

## Architecture

- **Engine (Go):** deterministic Tetris rules, seedable RNG, headless by default
- **Server (Go):** exposes the engine over gRPC or a Unix socket
- **Training (Python):** Gymnasium wrapper around the server, PyTorch agent
- **Viewer (Go, Ebitengine):** separate client for human play, agent replays, and live visualization

Training never depends on rendering.

## Phase 1: Engine

- Board, 7 tetrominoes, 7-bag randomizer, rotation (SRS), gravity, lock, line clear, game over
- Seedable RNG so every game is reproducible
- Unit tests for rotation, line clears, and bag behavior
- Clean API: `Reset(seed)`, `Step(action)`, state, reward, done

**Done when:** the engine passes its tests and plays a full game headless.

## Phase 2: Heuristic baseline

- Placement-based action space: for each piece, enumerate all (rotation, column) placements
- Features: aggregate height, holes, bumpiness, lines cleared
- Hand-weighted evaluation function that picks the best placement
- Record average score, lines, and survival over N seeded games

**Done when:** you have baseline numbers to compare the agent against.

## Phase 3: Bridge to Python

- Server wrapping the engine, one game session per connection
- Python Gymnasium environment calling the server
- Run multiple engine instances in parallel for training throughput
- Test with a random agent

**Done when:** a random agent runs through the Python env and results match the Go-side numbers.

## Phase 4: First RL agent

- DQN on the feature-based state, evaluating the afterstate of each placement
- Replay buffer, target network, epsilon decay
- Log with TensorBoard or Weights & Biases: reward, lines, loss, epsilon
- Run reward experiments one at a time (raw score vs. shaped reward with hole and height penalties)

**Done when:** the agent beats the random agent clearly, ideally approaches or beats the heuristic.

## Phase 5: Iterate and compare

- Try one extension at a time: Double DQN, dueling heads, or PPO
- Optional: CNN on the raw board instead of hand-made features
- Keep failed runs and notes, they go into the blog post
- Compare everything against the baseline on the same seeds

## Phase 6: Viewer and visualization

- Ebitengine client: human play, agent playback from checkpoints, episode replay
- Overlays: value heatmap over all candidate placements, live score and lines
- Learning curve panel and checkpoint selector
- Optional: human vs. agent mode

## Phase 7: Documentation and blog post

- README in English: architecture diagram, setup, how to train, how to run the viewer
- Blog post structure: motivation, engine design, baseline, what failed, what worked, results table, GIFs of agent progression (early, mid, late training)
- Link the repo prominently from the portfolio

## Scope rules

- Do not start the viewer before phase 4 works
- Change one thing per experiment and keep real numbers for each run
- Use the latest releases of all dependencies
- Commit with Conventional Commits
- If a phase runs long, cut phase 5 first, never phase 2 or 7

## Risks

- **Engine rabbit hole:** keep rendering out of phase 1
- **Agent does not learn:** check the engine and features first, using the heuristic as a sanity check
- **Slow training:** switch to placement-based actions and parallel envs before anything else