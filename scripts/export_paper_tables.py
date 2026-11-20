"""Export the five manuscript tables from a completed reproduction run.

This is a formatting/export step, not a new evaluation or calibration.
CSV floats preserve the available precision; the manuscript rounds Tables 2/4
to four decimals and Table 3 to six decimals. Only the standard library is used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path


TABLE_FILES = {
    1: "table1_dataset_counts.csv",
    2: "table2_retrieval.csv",
    3: "table3_primary_effects.csv",
    4: "table4_verifier_statistics.csv",
    5: "table5_clean_case_ranks.csv",
}
LABELS = {
    "audio_only": "Audio only",
    "caption_only": "Caption only",
    "ungated_fusion": "Ungated fusion",
    "gate_first": "First-excerpt gate",
    "gate_mean12": "Mean-of-12 gate",
    "random_gate_matched_first": "Random gate, first-matched count",
    "random_gate_matched_mean12": "Random gate, mean-matched count",
    "tfidf": "TF-IDF",
}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf8"))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def make_tables(data: Path, result: Path):
    inputs = {}

    def load(base, name):
        path = base / name
        inputs[("data/" if base == data else "result/") + name] = sha(path)
        return read(path)

    # A table export must not turn an incomplete or failed run into evidence.
    for name in ("development_match.json", "development_verification.json",
                 "reference_comparison.json", "reproduction_summary.json",
                 "evaluation/independent_verification.json"):
        require(load(result, name)["status"] == "passed", f"Required check not passed: {name}")
    planned = load(data, "data/experiment_manifest.json")
    active = load(data, "cache/active_manifest.json")
    excluded = load(data, "cache/exclusions.json")
    results = load(result, "evaluation/results.json")
    cases = load(result, "evaluation/cases.json")
    run_spec = load(result, "evaluation/run_spec.json")
    require(results["status"] == "completed", "Evaluation is incomplete")
    require(results["evaluation_settings_retuned"] is False, "Evaluation settings were retuned")
    for name, checksum in results["artifact_sha256"].items():
        require(Path(name).name == name, "Unexpected artifact path")
        require(sha(result / "evaluation" / name) == checksum, f"Artifact hash mismatch: {name}")

    planned_ids = {row["track_id"] for row in planned}
    active_ids = {row["track_id"] for row in active}
    excluded_ids = {row["track_id"] for row in excluded}
    require(len(planned_ids) == len(planned) and len(active_ids) == len(active)
            and len(excluded_ids) == len(excluded), "Duplicate track IDs in metadata")
    require(not (active_ids & excluded_ids) and active_ids | excluded_ids == planned_ids,
            "Retained and excluded tracks do not partition the plan")
    require(all(row["reason"] == "fewer_than_twelve_complete_windows" for row in excluded),
            "Table 1 length-exclusion label does not describe all exclusions")
    counts = {}
    for split in ("development", "evaluation"):
        planned_split = [row for row in planned if row["split"] == split]
        active_split = [row for row in active if row["split"] == split]
        excluded_split = [row for row in excluded if row["split"] == split]
        queries = [row for row in active_split if len(row["captions"]) > 1]
        counts[split] = [len(planned_split), len(excluded_split), len(active_split),
                         len({row["artist_id"] for row in active_split}), len(queries),
                         len({row["artist_id"] for row in queries})]
        if split == "evaluation":
            require({row["track_id"] for row in active_split} == set(run_spec["catalog_ids"]),
                    "Evaluation catalog differs from the data manifest")
            require({row["track_id"] for row in queries} == set(run_spec["query_source_ids"]),
                    "Evaluation queries differ from the data manifest")
    require(counts["evaluation"][2:] == [results["catalog_tracks"], results["catalog_artists"],
            results["query_tracks"], results["query_artists"]], "Evaluation population counts differ")
    table1 = [(label, counts["development"][i], counts["evaluation"][i])
              for i, label in enumerate(("Planned tracks", "Length exclusions", "Retained candidates",
                                          "Candidate artists", "Queries", "Query artists"))]

    retrieval = {(row["rate"], row["condition"]): row["metrics"] for row in results["retrieval"]}
    require(set(results["rates"]) == {0, 0.25}, "Manuscript requires both reassignment rates")
    require(set(results["conditions"]) == set(LABELS), "Manuscript condition set changed")
    require(len(retrieval) == len(results["retrieval"]) == 2 * len(LABELS), "Missing or duplicate retrieval rows")
    table2 = [(LABELS[condition], retrieval[0, condition]["MRR"]["estimate"],
               retrieval[0, condition]["R1"]["estimate"],
               retrieval[0.25, condition]["MRR"]["estimate"],
               retrieval[0.25, condition]["R1"]["estimate"])
              for condition in LABELS]
    table3 = []
    for hypothesis in ("H1", "H2"):
        effect = results["primary_effects"][hypothesis]
        finding = ("Supports predicted direction" if effect["lower"] > 0 else
                   "Opposite to prediction" if effect["upper"] < 0 else
                   "Improvement not established")
        table3.append((hypothesis, effect["definition"], effect["estimate"], effect["lower"],
                       effect["upper"], effect["level"], finding))
    verifiers = {row["verifier"]: row for row in results["verifiers"]}
    table4 = []
    for key, label in (("first_window", "First"), ("normalized_12_window_mean", "Mean12")):
        verifier = verifiers[key]
        table4.append((label, verifier["clean_rejection_rate"]["estimate"],
                       verifier["swapped_caption_acceptance_mean"],
                       verifier["paired_track_source_auroc_mean"]))
    table5 = []
    for case in cases:
        require(case["track_id"] in run_spec["query_source_ids"], "Case is not an evaluation query")
        row = [case["track_id"]]
        for condition in ("audio_only", "ungated_fusion", "gate_first", "gate_mean12"):
            ranks = case["rank_by_rate_condition_seed"]["0"][condition]
            require(len(ranks) == len(results["seeds"]) and len(set(ranks)) == 1,
                    "Clean deterministic case ranks differ across seeds")
            row.append(ranks[0])
        table5.append(row)
    require(len({row[0] for row in table5}) == len(table5), "Duplicate case IDs")
    return {
        1: (("Measure", "Development", "Evaluation"), table1),
        2: (("Condition", "MRR_0pct", "R1_0pct", "MRR_25pct", "R1_25pct"), table2),
        3: (("Effect", "Definition", "Estimate", "Lower", "Upper", "CI_level", "Directional_finding"), table3),
        4: (("Verifier", "Source_reject", "Swap_accept", "AUROC"), table4),
        5: (("Track_ID", "Audio", "Ungated", "First", "Mean12"), table5),
    }, inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True,
                        help="Completed reproduction root containing evaluation/ and pass reports")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--table", type=int, choices=range(1, 6), action="append",
                        help="Export a selected table; repeat for multiple tables; default: all five")
    args = parser.parse_args()
    data, result, output = (path.resolve() for path in (args.data_dir, args.result_dir, args.output_dir))
    require(output != data and data not in output.parents, "Do not export into immutable input data")
    require(output != result and output != result / "evaluation", "Use a distinct table-output directory")
    tables, inputs = make_tables(data, result)
    output.mkdir(parents=True, exist_ok=True)
    outputs = {}
    selected = sorted(set(args.table or TABLE_FILES))
    for number in selected:
        header, rows = tables[number]
        stream = io.StringIO(newline="")
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
        path = output / TABLE_FILES[number]
        path.write_text(stream.getvalue(), encoding="utf8", newline="")
        outputs[path.name] = {"table": number, "rows": len(rows), "sha256": sha(path)}
    report = {"status": "passed", "scope": "table export from completed results; no new scoring or tuning",
              "tables": selected, "inputs_sha256": inputs, "outputs": outputs,
              "float_policy": "CSV preserves precision; manuscript uses 4 decimals in Tables 2/4 and 6 in Table 3"}
    report_name = "table_export_manifest_" + "_".join(map(str, selected)) + ".json"
    (output / report_name).write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"status": "passed", "tables": selected, "files": list(outputs)}, indent=2))


if __name__ == "__main__":
    main()
