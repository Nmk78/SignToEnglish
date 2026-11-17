"""CLI commands that connect the model to the evaluation harness (added in Step 2.1).

- `s2e eval`: score a manifest split with `run_eval`, translating each row's clip through
  the offline pipeline.
- `s2e reproduce`: the reproduction check of the port against published keypoints and
  labels in the Uni-Sign authors' format (PLAN.md, Step 2.1). It reports the harness's
  sacreBLEU and, beside it, BLEU computed the way the Uni-Sign code computes it, so the
  number can be set against the paper's.

The harness itself (`run_eval.py`, `metrics.py`) is unchanged. Heavy imports stay inside
the handlers.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

#: A dataset name for reproduction items; they are not manifest rows of this project.
REPRODUCTION_SPLIT = "test"


def _load(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    from signtoenglish.common.config import find_repo_root, load_paths
    from signtoenglish.infer.loader import load_keypoint_config, load_model_config, load_translator
    from signtoenglish.infer.offline import OfflineTranslator

    paths = load_paths(find_repo_root())
    loaded = load_translator(
        paths,
        load_model_config(args.model_config),
        load_keypoint_config(args.keypoint_config),
        base_checkpoint=args.checkpoint,
        fine_tuned=getattr(args, "fine_tuned", None),
        device=args.device,
    )
    return paths, loaded, OfflineTranslator(loaded, paths)


def reference_style_bleu(hypotheses: list[str], references: list[str]) -> dict[str, Any]:
    """BLEU as the Uni-Sign code computes it (`SLRT_metrics.py:26-43`: `raw_corpus_bleu` of
    its vendored sacreBLEU 1.4.2, that is no tokenization, floor smoothing with value 0,
    effective order), computed here with the pinned sacreBLEU. For comparison with the
    paper's numbers only; the project's own metric is the harness's."""
    import sacrebleu

    out: dict[str, Any] = {
        "definition": "tokenize=none, smooth=floor(0), effective_order, case-sensitive"
    }
    for order in (1, 2, 3, 4):
        metric = sacrebleu.BLEU(
            tokenize="none", smooth_method="floor", smooth_value=0.0, force=True,
            effective_order=True, max_ngram_order=order,
        )  # fmt: skip
        out[f"bleu{order}"] = round(metric.corpus_score(hypotheses, [references]).score, 4)
    return out


def _eval(args: argparse.Namespace) -> int:
    from signtoenglish.eval.run_eval import EvalError, format_report, load_eval_items, run_eval
    from signtoenglish.infer.loader import CheckpointRefusedError
    from signtoenglish.infer.offline import OfflineError
    from signtoenglish.models.translator import TranslatorError

    try:
        items = load_eval_items(args.manifest, split=args.split)
        if args.limit:
            items = items[: args.limit]
        paths, loaded, offline = _load(args)
    except (
        EvalError,
        CheckpointRefusedError,
        TranslatorError,
        FileNotFoundError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    def translate(batch: Any) -> list[str]:
        texts = []
        for item in batch:
            row = item.row
            result = offline.translate_file(
                paths.data_root / row["source_path"],
                start_ms=row.get("start_ms"),
                end_ms=row.get("end_ms"),
                expected_sha256=row.get("source_sha256"),
                clip_id=item.id,
            )
            texts.append(result.text)
        return texts

    try:
        result = run_eval(
            translate,
            items,
            split=args.split,
            run_dir=args.run_dir,
            checkpoint_id=loaded.checkpoint_id,
            config=loaded.model_cfg,
            batch_size=1,
            force=args.force,
            device=loaded.translator.device,
            extra_hashes={"preprocessing_hash": loaded.preprocessing_hash},
        )
    except (EvalError, OfflineError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(format_report({args.split: result}))
    return 0


def _reproduce(args: argparse.Namespace) -> int:
    from signtoenglish.eval.run_eval import EvalError, EvalItem, format_report, run_eval
    from signtoenglish.infer.loader import CheckpointRefusedError
    from signtoenglish.infer.offline import (
        OfflineError,
        load_published_labels,
        published_pose_path,
    )
    from signtoenglish.models.translator import TranslatorError

    try:
        labels = load_published_labels(args.labels)
        total = len(labels)
        if args.subset and args.subset < total:
            # A fixed random subset: the same seed always picks the same sentences.
            chosen = sorted(random.Random(args.seed).sample(range(total), args.subset))
            labels = [labels[i] for i in chosen]
        missing = [
            lab.name for lab in labels if not published_pose_path(args.pose_dir, lab).is_file()
        ]
        if missing:
            raise OfflineError(
                f"{len(missing)} of {len(labels)} pose files are missing under {args.pose_dir} "
                f"(first: {missing[0]})"
            )
        _, loaded, offline = _load(args)
    except (OfflineError, CheckpointRefusedError, TranslatorError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    by_id = {lab.name: lab for lab in labels}
    if len(by_id) != len(labels):
        print("error: duplicate sample names in the label file", file=sys.stderr)
        return 2
    items = [
        EvalItem(
            id=lab.name,
            split=REPRODUCTION_SPLIT,
            text=lab.text,
            dataset=args.dataset,
            sentence_id=None,
            row={"video_path": lab.video_path},
        )
        for lab in labels
    ]
    done = 0
    started = time.perf_counter()

    def translate(batch: Any) -> list[str]:
        nonlocal done
        texts = []
        for item in batch:
            pose = published_pose_path(args.pose_dir, by_id[item.id])
            texts.append(offline.translate_published(pose, clip_id=item.id).text)
            done += 1
            if done % 25 == 0 or done == len(items):
                rate = (time.perf_counter() - started) / done
                print(
                    f"  {done}/{len(items)} sentences, {rate:.2f} s each",
                    file=sys.stderr,
                    flush=True,
                )
        return texts

    try:
        result = run_eval(
            translate,
            items,
            split=REPRODUCTION_SPLIT,
            run_dir=args.run_dir,
            checkpoint_id=loaded.checkpoint_id,
            config=loaded.model_cfg,
            batch_size=1,
            force=args.force,
            device=loaded.translator.device,
            extra_hashes={
                "keypoints": "published by the Uni-Sign authors; not this project's extraction"
            },
        )
    except (EvalError, OfflineError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    hyps = (result.eval_dir / "hyps.txt").read_text(encoding="utf-8").splitlines()
    refs = (result.eval_dir / "refs.txt").read_text(encoding="utf-8").splitlines()
    extra = {
        "dataset": args.dataset,
        "sentences_scored": len(items),
        "sentences_in_label_file": total,
        "is_subset": len(items) < total,
        "subset_seed": args.seed if len(items) < total else None,
        "checkpoint": loaded.translator.report.checkpoint_name,
        "device": loaded.translator.device,
        "seconds": round(time.perf_counter() - started, 1),
        "reference_style_bleu": reference_style_bleu(hyps, refs),
        "note": "Inputs over the model's maximum are thinned to evenly spaced frames; the "
        "reference samples them at random.",
    }
    (result.eval_dir / "reproduction.json").write_text(
        json.dumps(extra, indent=2), encoding="utf-8"
    )
    print(format_report({REPRODUCTION_SPLIT: result}))
    print(json.dumps(extra))
    return 0


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="where eval/<split>/ and test_runs.log are written",
    )
    p.add_argument("--checkpoint", help="base checkpoint name from the model config")
    p.add_argument("--device", choices=["auto", "mps", "cpu"], help="override the config's device")
    p.add_argument(
        "--force",
        action="store_true",
        help="score a test split again for the same checkpoint (logged)",
    )
    p.add_argument("--model-config", help="model config YAML")
    p.add_argument("--keypoint-config", help="keypoint config YAML")


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    e = subparsers.add_parser("eval", help="translate and score one manifest split")
    e.add_argument("--manifest", type=Path, required=True)
    e.add_argument("--split", required=True, help="train, val, test_seen, test_unseen, or test")
    e.add_argument("--fine-tuned", type=Path, help="a checkpoint written by this project")
    e.add_argument(
        "--limit",
        type=int,
        default=0,
        help="score only the first N rows (not for reported results)",
    )
    _common(e)
    e.set_defaults(func=_eval)

    r = subparsers.add_parser(
        "reproduce",
        help="score the pretrained model on published keypoints and labels (reproduction check)",
    )
    r.add_argument(
        "--pose-dir", type=Path, required=True, help="folder of .pkl pose files for the split"
    )
    r.add_argument(
        "--labels", type=Path, required=True, help="Uni-Sign label file for the split (gzip pickle)"
    )
    r.add_argument("--dataset", default="how2sign", help="name recorded in the results")
    r.add_argument(
        "--subset", type=int, default=0, help="score a fixed random subset of this many sentences"
    )
    r.add_argument("--seed", type=int, default=0, help="seed of the subset")
    _common(r)
    r.set_defaults(func=_reproduce)
