# Recording protocol

How to record the domain sentences with `s2e record` so that the takes are usable for
adapting the model and for testing it honestly. Written 2026-11-16.

Claim labels as in `PLAN.md`: **[verified]** read from a primary source or measured here,
**[inferred]** reasoned from verified facts, **[assumed]** not checked. Most numbers below
are planning defaults and are marked [assumed]; none of them has been measured against
model quality yet.

This document says nothing about how any sentence is signed. That comes from your ASL
source and goes into `domain/sentences.yaml` (`asl` and `source`), not here.

## 1. Before the first take

1. **Every sentence has an ASL version from a reliable source**, written in `asl:` with
   where it came from in `source:` (`OPEN_QUESTIONS.md`, Q10). Check what is still missing:

   ```sh
   uv run --frozen s2e domain check
   ```

   It prints one warning per sentence whose `asl` is empty and one per sentence whose
   `source` is empty, and fails on real errors (duplicate ids or text, a sign missing from
   the vocabulary, an unseen-test sign that appears in fewer than three training sentences).
   If your ASL version of a sentence uses different signs than its `signs:` list, correct
   the list and run the check again.
2. **Record a sentence only once you sign it the same way every time.** Practise first.
   If you are still changing how you sign a sentence, it is not ready: early, inconsistent
   takes hurt more than they help [assumed, `OPEN_QUESTIONS.md` Q10]. In the tool, press
   `N` to skip a sentence that is not ready; it will not be offered again in that session.
3. If a fluent signer can review your signing, do that before recording, not after.
4. If you change how you sign a sentence after recording it, the old takes of that
   sentence no longer match. Tell the next session so they can be removed from the manifest;
   do not mix the two versions.

## 2. Setup

### Framing

- One person in the picture. Camera at about chest-to-eye height, straight on, not tilted.
- Upper body in frame: from a little above the head to the hips or lap, with **the face and
  both hands visible for the whole sentence**, including the widest and highest movement
  you make. Leave a margin on both sides; a hand that leaves the frame is lost
  (`ARCHITECTURE.md` section 6.2).
- Sit or stand the way you will at the show, at about the same distance. Check the distance
  with the preview before each session: stretch both arms out and up, and make sure the
  hands stay in the picture.
- The preview is mirrored so it behaves like a mirror. The saved video is not mirrored and
  has no text on it [verified by test].
- Landscape orientation. The tool asks the camera for 1280x720 and accepts what it gets,
  with a minimum of 640x480. It prints the size and the measured frame rate at start; if
  the rate is below 20 frames per second, add light or close other apps.

### Lighting

- Light the front of your body and both hands evenly. A lamp or window in front of you is
  good; a window or bright lamp **behind** you is not.
- Avoid hard shadows across the hands and face, and avoid flicker.
- More light also means a faster shutter and less motion blur on moving hands
  [inferred from how cameras work; not measured here].

### Background and clothing

- A plain, still background if possible. Nobody walking behind you.
- Sleeves and top in a plain colour that contrasts with your skin, so the hands stand out.
- Do not wear the same clothes in every session (see section 3).
- Remove anything that covers the hands or face.

### Machine

- Mains power, other heavy apps closed.
- Click the camera window once so it receives key presses.

## 3. Sessions

A session is one sitting under one set of conditions. Its id is the date plus the label you
give, for example `2026-11-20_desk-evening`.

- **At least two sessions on different days** [assumed, `PLAN.md` Step 2.2]. Different days
  bring different light, clothing, and small differences in how you sit and sign, and the
  model has to cope with all of that at the show.
- **Different clothing between sessions**, and where practical a different background or
  seat position.
- **At least one session in show-like conditions**: the lighting, background, distance, and
  standing or sitting position you expect at the show, ideally in the same kind of room.
  Mark it with `--show-conditions`, which tags every take `show_conditions`.
- Keep a session to a length you can sign consistently in. Stop when you are tired; quit
  with `Q` and run the same command again later the same day to continue the same session.

Commands (replace `me` with a short id for yourself):

```sh
uv run --frozen s2e record --signer me --session desk-evening
uv run --frozen s2e record --signer me --session hall --show-conditions
```

## 4. Takes per sentence

- **Target: at least 6 accepted takes of every training sentence, over at least 2 sessions**
  [assumed, `PLAN.md` Step 2.2 metric target]. With the default `--takes 3`, two full
  sessions reach it.
- **Record the 20 unseen-test sentences the same way.** They are never trained on, but they
  need takes to be validated and tested on. The same 3 takes per session is a reasonable
  default [assumed].
- The tool keeps the counts balanced: it always offers the sentence with the fewest takes
  in this session, then the fewest overall, in an order that changes from session to
  session. You do not choose the next sentence.
- More takes of fewer sentences is not better than balanced takes of all of them: the
  unseen-sentence test only works if every sign in those sentences was seen in several
  different training sentences.

## 5. One take

1. The window shows the sentence (and your `asl` line, if you have filled it in).
   **Start with your hands lowered at rest**: at your sides, or in your lap if seated, out
   of the signing space.
2. Press `SPACE`. A 3-second countdown runs. Nothing is recorded during it.
3. When recording starts (red REC), sign the sentence at your normal pace.
4. **Lower your hands back to rest and hold still for about one second**, then press
   `SPACE` to stop. A take stops by itself after 20 seconds.
5. Review: `A` accept, `R` redo, `P` replay, `N` drop it and skip this sentence, `Q` quit
   (drops the take on screen).

Other keys: during recording `R` starts the same sentence over and `Q` quits; during the
countdown `SPACE` or `ESC` cancels.

Why rest at both ends: live, the system decides that a sentence has started when the hands
leave the rest position and that it has ended when they return (`ARCHITECTURE.md` section
6.1). Takes that begin and end at rest look like what the model will get live [inferred].

About the key press: reaching for the keyboard is itself a hand movement and ends up at the
tail of the take. Hold the rest position first, so the sentence and the reach are clearly
separated. Better, have a second person press the keys, or use a presenter remote or foot
switch that sends the space key, so your hands stay at rest [assumed; not yet measured
whether the reach affects the model].

### Accept or redo

Accept a take only if all of these hold. When in doubt, redo; a redo costs a few seconds.

- You signed the sentence the way you practised it, start to finish, without restarting,
  correcting yourself, or hesitating in the middle.
- Both hands and your face stayed in the picture.
- You started and ended at rest.
- Nothing else happened: nobody walked through, the light did not change, you did not look
  at the keyboard or talk while signing.

Use `P` to replay when you are not sure. A redone take leaves no file and no manifest row.

## 6. Continuous takes

Continuous takes are several sentences signed one after another in a single recording.
They are used to tune and score the sentence-end detector (`PLAN.md` Step 3.2), not to
adapt the translator.

```sh
uv run --frozen s2e record --signer me --session desk-evening --continuous
```

1. The window lists the sentences (4 by default). Start with hands at rest; press `SPACE`
   for the countdown.
2. Sign the first sentence. **Lower your hands to rest between sentences and hold for about
   one second.** This is the signal the system uses live to end a sentence; without it,
   sentences run together. The detector's default threshold is 0.6 seconds at rest
   (`ARCHITECTURE.md` section 6.1), so a full second leaves a margin [inferred].
3. **Press `SPACE` once each time a sentence ends**, when the hands have reached rest. The
   window then shows the next sentence. A second press within 0.7 seconds is ignored.
4. After the last sentence's key press, keep your hands down; recording stops by itself a
   second later. Then `A` accept, `R` redo, `P` replay.

The key press marks the sentence end. It is scored with a tolerance of one second
(`ARCHITECTURE.md` section 7), so press promptly and at the same point every time: just as
the hands arrive at rest. A helper or a remote for the key press matters more here than for
single takes, because your hands should be resting when it happens.

How many: aim for at least 10 continuous takes over at least two sessions, including the
show-like session, so there are enough for both validation and test [assumed]. If you get a
sentence wrong inside a continuous take, redo the whole take.

## 7. After each session

```sh
uv run --frozen s2e manifest validate
```

This checks every recorded row: ids, text against the sentence list, files present, content
hashes, readable video with increasing timestamps, and marked sentence ends inside the
video. It must end with `VALID`.

- Recordings are written under `data/domain/<session>/` and rows are appended to
  `data/manifests/domain_takes.jsonl` and `data/manifests/domain_continuous_takes.jsonl`.
  All of `data/` is git-ignored. Nothing is uploaded anywhere.
- **Back up `data/`** to another disk after each session. It is not in git and cannot be
  regenerated.
- The recordings belong to the person in them. If someone other than you signs, get their
  consent for this use and for being shown at the demo (`CLAUDE.md`, license rules), and
  give them their own `--signer` id.
- Do not rename, trim, or re-encode the files: each row stores the file's content hash and
  validation fails if the file changes.

## 8. When all recording is done

Build the splits once, after the last session:

```sh
uv run --frozen s2e splits build --dry-run
uv run --frozen s2e splits build
```

It assigns each take to `train`, `val`, `test_seen`, or `test_unseen`, and refuses to write
anything unless: no recording is in two splits; no unseen-test sentence appears in train or
val; and every sign in an unseen-test sentence appears in at least three training sentences
that really have a training take. Rebuilding after adding more takes can move takes between
train, val, and test_seen, so build once and keep the result.

Use `--holdout-session <session id>` to keep one whole session out of training and use it
for the seen-sentence test.

## 9. What the tool stores

- Video: H.264 in MP4. Each frame is stored at its real arrival time, so the file has a
  variable frame rate exactly as the camera delivered it; no frame rate is assumed
  [verified by test: container timestamps equal the source's capture times to the
  millisecond].
- The timestamp is taken on this machine's monotonic clock when the frame is handed over by
  the camera driver. It is not the sensor's exposure time; the difference is the camera's
  internal delay [inferred; the delay itself is not measured].
- On this Mac, encoding one 1280x720 frame takes about 3 ms and the whole per-frame loop
  (encode, draw, show, read keys) about 20 ms, which fits inside the 33 ms between frames
  at 30 frames per second [verified, measured 2026-11-16 on generated frames, not with the
  camera]. Each saved take prints its own measured frame rate; if it is well below what the
  camera reported at start, tell the next session.
