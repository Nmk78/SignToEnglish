# WORKERS.md — prompts for parallel worker sessions

How to run the plan with several sessions at once. `PLAN.md` stays the source of truth for what each step delivers and how it is tested; this file says which steps can overlap, who owns which files, and gives a prompt to paste into each session.

## What can run in parallel

```
Wave 0 (one session, alone)
  W0  Step 0.1  scaffold, pinned environment, shared plumbing
        │
Wave 1 (up to four sessions at once, after W0 is merged)
  A   Step 0.2 → 0.3 → 1.1 → 1.2 → 2.1      critical path, one step per session
  B   Web app (Step 4.1 build)               web/ only
  C   Recording tool, manifest, splits       the model-free part of Steps 1.2 and 2.2
  D   Metrics and evaluation harness         the model-free part of Step 2.1
        │
        │   meanwhile, you: fill in ASL versions, learn the sentences,
        │   then record with C's tool as soon as it is merged
        │
Wave 2 (after A finishes Step 2.1; up to three at once)
  E   Step 3.1 streaming engine  →  then F and G in parallel:
        F   Step 3.2 sentence boundaries     needs your continuous takes
        G   Step 3.3 live service            uses the fixed-length boundary until F lands
  H   Step 2.3 → 2.4 zero-shot and adaptation   needs your recordings
        │
Wave 3 (one session at a time)
  I   Step 4.1 acceptance against the real service, then Step 4.2
```

Why these splits are safe:

- **B** touches only `web/` and builds against the WebSocket and REST protocol already fixed in `ARCHITECTURE.md` section 6.3.
- **C** needs no model and no pose estimator. Getting it done early matters most, because your recording time is the longest task in the project and it can start as soon as the tool exists.
- **D** is pure text-metric code.
- **E** needs only the model wrapper from Step 2.1, not the adapted model.
- **G** was listed after Step 3.2 in `PLAN.md`; it can start after 3.1 because the boundary detector sits behind an interface.

What must stay serial:

- **W0 before everything.** It creates the repository, the lockfile, and the command registry the others plug into.
- **Track A's five steps**, in order. Step 0.3 is the go/no-go test for the base model on this Mac. If it fails, B and C are still useful; anything model-specific stops.
- **Step 2.3 before 2.4**, and both after recordings exist.
- **Step 4.2 last.**

Limits to keep in mind: this Mac has 16 GB of memory, so run only one session at a time that loads the model or runs pose extraction (A, E, F, G, H compete; B, C, D are light). Each session will stop and ask you at its own stop points.

## Ground rules for parallel work

1. **One git worktree and one branch per worker**, created from `main` after W0 is merged: `git worktree add ../s2e-<track> -b track/<track>`. Never two sessions in one working directory.
2. **File ownership** (a worker edits only its own paths):

   | Worker | Owns |
   | --- | --- |
   | W0 | everything, once |
   | A | `src/signtoenglish/preprocess/`, `models/`, `infer/offline.py`, `infer/loader.py`, `docs/base_model_notes.md`, `docs/feasibility.md`, `docs/DATA_LICENSES.md`, `OPEN_QUESTIONS.md` (approved assets and verification log), their tests |
   | B | `web/` |
   | C | `src/signtoenglish/record/`, `data/manifest.py`, `data/domain.py`, `data/splits.py`, `docs/recording_protocol.md`, their tests |
   | D | `src/signtoenglish/eval/metrics.py`, `eval/run_eval.py`, their tests |
   | E | `src/signtoenglish/infer/streaming.py`, `eval/stream_eval.py`, their tests |
   | F | `src/signtoenglish/models/boundary.py`, its tests |
   | G | `src/signtoenglish/serve/`, `scripts/ws_replay.py`, their tests |
   | H | `src/signtoenglish/train/`, `models/adapters.py`, `data/dataset.py`, `data/collate.py`, `docs/results.md`, their tests |

3. **Nobody but W0 edits `pyproject.toml`, `uv.lock`, `.gitignore`, `CLAUDE.md`, `PLAN.md`, or `ARCHITECTURE.md`.** A worker that needs a new dependency or a format change stops and tells you; you make the change on `main` in one place and the workers rebase.
4. **Commands register themselves.** W0 sets up `cli.py` to discover a `commands.py` in each package, so workers add subcommands without editing a shared file.
5. **Merge order:** merge each track into `main` when its gate passes; other tracks rebase onto `main` before their own merge. A track that depends on an unmerged one waits.
6. **All of `CLAUDE.md` still applies in every session**, including every stop point.

## Prompts

Paste the shared block first, then the block for that worker.

### Shared block (start of every worker prompt)

```
You are one of several worker sessions implementing the SignToEnglish project in parallel.

Before doing anything, read in full: CLAUDE.md, PLAN.md, ARCHITECTURE.md, WORKERS.md, and the parts of OPEN_QUESTIONS.md and REUSE.md that your task touches. They are binding.

Rules for this session:
- Work only in the git worktree and branch you were started in. Edit only the paths WORKERS.md assigns to your track. Do not edit pyproject.toml, uv.lock, .gitignore, CLAUDE.md, PLAN.md, or ARCHITECTURE.md; if you need a dependency or a format change, stop and tell me exactly what and why.
- Obey every stop point in CLAUDE.md. Do not download datasets, weights, or model files, do not install system tools, do not start jobs longer than 10 minutes, and do not write training code, without my approval in this session.
- Do not invent facts about the base model, licenses, or ASL. If something has to be looked up, read the source, record it, and label it verified, inferred, or assumed.
- Tests must exercise the real thing. A mock or stub does not satisfy a gate; if your gate needs something that does not exist yet, say which checks are deferred and to which step.
- Do the one task below, run its acceptance tests, commit on your branch with a clear message, and stop. Do not start another step.
- Finish with a report: what was built, each acceptance test and whether it passed (with the output for any failure), every measured number, anything deferred, and anything the next session or another track needs to know.
```

### W0 — Step 0.1, scaffold (run alone, first)

```
Task: PLAN.md Step 0.1, in the repository root on a new `main` branch. You own the whole repo for this step.

In addition to the Step 0.1 deliverables:
- Ask me before installing uv and ffmpeg.
- Pin, up front, every Python dependency the plan names so later workers do not touch pyproject.toml: PyTorch, transformers, onnxruntime, rtmlib, a video decoding library, numpy, pydantic, PyYAML, FastAPI, uvicorn, python-multipart, websockets support, sacrebleu, rouge-score, an adapter library for low-rank fine-tuning, pytest, ruff, a type checker. Read current versions from the package index; do not pick them from memory. Check that the set resolves together on this Mac, and tell me if rtmlib or the Uni-Sign code is likely to need older versions.
- Implement `common/` now (config loading into frozen pydantic models, canonical-JSON config hashing, device selection, logging) with tests, because every track depends on it.
- Make `cli.py` discover subcommands from an optional `commands.py` module in each package, so tracks add commands without editing a shared file. Test it with a trivial `s2e doctor` command.
- Create the package skeleton and the import-rule test from ARCHITECTURE.md section 2.
- Commit `domain/sentences.yaml`, the five planning files, and WORKERS.md as they are.
```

### A — critical path (one session per step, in order)

```
Track A, critical path. Task for this session: PLAN.md Step <0.2 | 0.3 | 1.1 | 1.2 | 2.1>.

Notes for this track:
- Step 0.2: list every asset by name with its license and wait for my approval of each before any download. Write docs/base_model_notes.md from reading the Uni-Sign code; every "per base model" item in ARCHITECTURE.md must be answered there with file and line references.
- Step 0.3 is a go/no-go gate. If the model cannot run on this Mac in 32-bit floats on MPS or CPU, or misses the speed target badly, stop and report; do not work around it by changing the plan.
- Step 1.2: `data/manifest.py` belongs to Track C. Use it if it is merged; otherwise depend only on the manifest format in ARCHITECTURE.md section 4.2 and note the integration as deferred.
- Step 2.1: `eval/metrics.py` and `eval/run_eval.py` belong to Track D. Use them if merged.
- Only one model-loading session should run on this machine at a time. Tell me before starting a run that loads the model.
```

### B — web app

```
Track B, web app. Task: build the `web/` application described in PLAN.md Step 4.1 and ARCHITECTURE.md section 6.4, against the protocol in ARCHITECTURE.md section 6.3. Follow REUSE.md item 7 for what to carry over from the old Next.js page and what to remove.

Scope:
- Camera view with framing guide, start and stop, committed captions and partial text shown differently, status bar, domain panel listing what the demo covers (read example sentences from a small JSON file you generate from domain/sentences.yaml at build time), and the video-input panel with both options: "translate file" (POST /translate) and "play as live" (play the file in the page and send its frames with media timestamps over the WebSocket exactly as the webcam path does).
- Exact-pinned package.json and committed package-lock.json. Ask me before running `npm install`.
- For development only, you may write a small local fake server under web/dev/ that speaks the documented protocol with canned text. Label it clearly as a development aid. It does not satisfy any gate.

Gate for this session: `npm ci`, lint, type check, and build pass; browser tests against the fake server cover the webcam path with a video file as the camera, both video-input options, refused camera permission, and a dropped connection. Report explicitly that acceptance against the real service is deferred to Wave 3.
```

### C — recording tool, manifest, splits

```
Track C, data collection. Task: the model-free parts of PLAN.md Steps 1.2 and 2.2.

Build:
- `data/manifest.py`: the manifest and continuous-take formats from ARCHITECTURE.md section 4.2, with validation (duplicate ids, empty text, bad durations, missing files, content hash).
- `data/domain.py`: loader and validator for domain/sentences.yaml, including a check that warns for every sentence whose `asl` or `source` field is still empty.
- `data/splits.py`: the split builder and its guarantees from ARCHITECTURE.md section 4.3, as tests: no recording in two splits; no unseen-test sentence text in train or validation; every sign in an unseen-test sentence appears in at least three training sentences.
- `record/recorder.py` and the `s2e record` command from ARCHITECTURE.md section 5: prompt, countdown, webcam capture with real timestamps, accept or redo, balanced choice of the next sentence, session labels, and continuous mode with a key press marking each sentence end.
- docs/recording_protocol.md: framing, lighting, takes per sentence, at least two sessions on different days, one session in show-like conditions, hands lowered to rest between sentences, and the rule to record only once a sentence is signed the same way every time.

Do not write any ASL content. Do not fill in `asl` or `signs` fields; those are mine to supply.
Gate: all tests above pass; I can run `s2e record` on this Mac, record three takes and one continuous take, and the resulting manifests validate. Ask me to do that trial before you finish.
```

### D — metrics and evaluation harness

```
Track D, evaluation. Task: `eval/metrics.py` and `eval/run_eval.py` per ARCHITECTURE.md sections 4.7 and 7.

Build:
- sacreBLEU wrapper returning BLEU-1 to BLEU-4 and the signature; ROUGE-L; exact-match rate after light normalization (case and punctuation), with the normalization defined once and tested.
- `run_eval.py` written against a `translate(batch) -> list[str]` callable so it does not depend on the model: writes hyps, refs, metrics, and config hash to a run directory; reports the seen-sentence and unseen-sentence test sets separately; appends to test_runs.log and refuses a second test-split run for the same checkpoint without --force.

Gate: wrappers reproduce hand-computed values on a three-sentence fixture; the refusal and the log are tested; the harness runs end to end with a trivial callable in a test, and you state that the real-model run is deferred to Step 2.1.
```

### E, F, G, H, I — later waves

```
Task for this session: PLAN.md Step <3.1 | 3.2 | 3.3 | 2.3 | 2.4 | 4.1 acceptance | 4.2>, on track <E | F | G | H | I> as assigned in WORKERS.md.

Notes:
- Step 3.3 may start once Step 3.1 is merged: use the fixed-length boundary detector until Step 3.2 lands, and list the boundary-dependent acceptance checks as deferred.
- Step 3.2 needs my continuous takes; Steps 2.3 and 2.4 need my recordings. If they are not there, stop and tell me what is missing.
- Step 2.4: state the training design and wait for my confirmation before writing training code. Ask before the single test-set scoring.
- Only one model-loading session should run on this machine at a time. Tell me before starting one.
```

## Paste-ready prompts

The blocks above are the full instructions. Because every session reads this file, what you paste can be short.

### 1. Scaffold (one session, run first, in the repo root)

```
Read CLAUDE.md, PLAN.md, ARCHITECTURE.md, and WORKERS.md in full. You are worker W0. Follow the "Shared block" and the "W0" block in WORKERS.md exactly. Start now.
```

### 2a. Wave 1 from a single session (orchestrator)

Paste this into one session in the repo root after W0 is committed on `main`.

```
Read CLAUDE.md, PLAN.md, ARCHITECTURE.md, and WORKERS.md in full. You are the orchestrator for Wave 1. Do not implement anything yourself.

1. Confirm W0 is committed on main and `make check` passes. If not, stop and tell me.
2. Create one git worktree and branch per track, as WORKERS.md describes: track/a, track/b, track/c, track/d.
3. Launch four worker agents in parallel, one per worktree. Give each one, verbatim, the "Shared block" from WORKERS.md followed by its own block, plus the absolute path of its worktree as its working directory:
   - A: Step 0.2 only.
   - B: the web app block.
   - C: the recording tool, manifest, and splits block.
   - D: the metrics and evaluation harness block.
4. Workers cannot ask me directly. When a worker reaches a stop point or needs a decision (installing a tool, running `npm install`, any download, a new dependency, a format change, the recording trial), it must stop and report it to you. Relay the question to me word for word, wait for my answer, and send my answer back to that worker. Never approve anything on my behalf, and never treat a worker's report as my approval.
5. When a worker finishes, show me its report unedited in substance: acceptance tests passed and failed, measured numbers, and deferred checks. Then run the full test suite on its branch yourself and tell me the result.
6. Ask me before merging any branch into main. After a merge, have the remaining tracks rebase.
7. Track A continues one step at a time (0.3, 1.1, 1.2, 2.1): start the next step only when I say so. Only one model-loading task may run at a time.
8. Keep a short status table (track, step, state, waiting on) and show it whenever something changes.
```

### 2b. Wave 1 as four separate sessions

Create the four worktrees yourself, open one session in each, and paste one line per session:

```
Read CLAUDE.md, PLAN.md, ARCHITECTURE.md, and WORKERS.md in full. You are worker A. Follow the "Shared block" and the "A" block in WORKERS.md exactly. Your task this session is Step 0.2. Start now.
```
```
Read CLAUDE.md, PLAN.md, ARCHITECTURE.md, and WORKERS.md in full. You are worker B. Follow the "Shared block" and the "B" block in WORKERS.md exactly. Start now.
```
```
Read CLAUDE.md, PLAN.md, ARCHITECTURE.md, and WORKERS.md in full. You are worker C. Follow the "Shared block" and the "C" block in WORKERS.md exactly. Start now.
```
```
Read CLAUDE.md, PLAN.md, ARCHITECTURE.md, and WORKERS.md in full. You are worker D. Follow the "Shared block" and the "D" block in WORKERS.md exactly. Start now.
```

For later steps, use the same line with the worker letter and the step number from the wave diagram.
