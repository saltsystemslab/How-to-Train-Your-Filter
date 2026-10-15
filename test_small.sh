# Script to check that all experiments run, using a small version of them: run_queries.sh on url only, at two sizes
# (r = 5 and 6) and with 1 million queries, then the plots. It checks that every experiment finished with the
# expected number of results, that nothing failed, and that every filter and trial saw the same queries and churns in
# the dynamic workloads. Its results, logs and figures are moved to test_runs/<date and time>/, so results/ is left as
# it was.
# Takes about an hour.

#!/bin/bash
set -e # exit immediately when a command fails

if ls results/*.csv > /dev/null 2>&1; then
    echo "results/ already has results files; move them elsewhere first so the test doesn't mix with them"
    exit 1
fi
out="test_runs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$out/logs" logs
marker="$out/started"
touch "$marker"

echo "running the experiments (log in $out/run_queries.txt)"
status=0
DATASETS="url" R_VALUES="5 6" NUM_QUERIES=1000000 bash run_queries.sh > "$out/run_queries.txt" 2>&1 || status=$?

echo "plotting"
(cd learned && python3 plot_generic_results.py --results ../results --out "../$out/figures" --datasets url) > "$out/plots.txt" 2>&1 || status=$?

# move this test's results and logs out of results/ and logs/
mv results/*.csv "$out/" 2> /dev/null || true
find logs -newer "$marker" -type f -exec mv {} "$out/logs/" \;

echo "checking the results"
python3 - "$out" "$status" << 'EOF'
import glob, os, re, sys
import pandas as pd

out, status = sys.argv[1], int(sys.argv[2])
ok = status == 0
def check(condition, message):
    global ok
    print(("ok      " if condition else "PROBLEM ") + message)
    ok = ok and condition

check(status == 0, f"run_queries.sh and the plots finished (exit status {status})")
failed = f"{out}/logs/failed_jobs.txt"
check(not os.path.exists(failed), "no experiment failed" + (f": see {failed}" if os.path.exists(failed) else ""))
crashes = [log for log in glob.glob(f"{out}/logs/*.txt")
           if re.search(r"Segmentation fault|Aborted|core dumped|couldn't|PROBLEM", open(log, errors="replace").read())]
check(not crashes, "no log reports a crash or a missing file" + (f": {crashes}" if crashes else ""))

filters = ["aqf", "stacked", "bloom", "plbf", "clbf"]
# rows per filter: 2 sizes x 3 trials for each static workload, 5 adversarial frequencies x 3 trials, and 3 trials x
# 90 measurements (9 recorded windows x 10) for each dynamic workload
expected = {("distribution_results.csv", "unif"): 6, ("distribution_results.csv", "zipf"): 6,
            ("distribution_results.csv", "onepass"): 6, ("advers_results.csv", None): 15,
            ("dynamic_results.csv", None): 270, ("shifting_results.csv", None): 270}
for (name, dist), count in expected.items():
    path = f"{out}/{name}"
    if not os.path.exists(path):
        check(False, f"{name} was written")
        continue
    df = pd.read_csv(path)
    if dist:
        df = df[df["query_dist"] == dist]
    counts = df.groupby("filter").size().reindex(filters, fill_value=0)
    label = name + (f" ({dist})" if dist else "")
    check((counts == count).all(), f"{label}: {count} rows for each filter (got {counts.to_dict()})")
    if "size" in df and "params" in df:
        # learned filters report their budget; the others must not be larger than the AQF they're matched to
        over = df[df["params"].astype(str).str.contains("budget=")].apply(
            lambda row: row["size"] > int(re.search(r"budget=(\d+)", row["params"]).group(1)), axis=1)
        check(not over.any(), f"{label}: no learned filter is over its budget")

# the timed uniform rows have times; the untimed workloads record where the learned filters' false positives came from
dist = pd.read_csv(f"{out}/distribution_results.csv")
unif = dist[dist["query_dist"] == "unif"]
# rows built to the r = 5 size (69160 bytes for url; learned filters report it as their budget, and the bloom filter
# rounds down to whole 64-bit words)
budget = unif.apply(lambda row: int(m.group(1)) if (m := re.search(r"budget=(\d+)", str(row["params"]))) else row["size"], axis=1)
timed = (69160 - budget).between(0, 7)
check(unif[timed]["build_time_ns"].notna().all() and unif[~timed]["build_time_ns"].isna().all(),
      "the uniform experiments are timed at r = 5 (the size times are reported for) and untimed at r = 6")
untimed = dist[(dist["query_dist"] != "unif") & dist["filter"].isin(["plbf", "clbf"])]
check(untimed["fp_from_model"].notna().all(), "the untimed experiments record the learned filters' false positive sources")

# every filter must see the same queries and churns in each dynamic workload
for workload in ["swapping", "shifting"]:
    sums = {}
    for f in filters:
        logs = glob.glob(f"{out}/logs/{workload}_{f}_url_*.txt")
        sums[f] = "".join(line for log in logs for line in open(log, errors="replace") if "checksum" in line)
    same = len(set(sums.values())) == 1 and all(sums.values())
    check(same, f"{workload}: every filter saw the same queries and churns ({sum(len(s.splitlines()) for s in sums.values())} windows logged)")
    # and every trial saw the same ones
    trials = {}
    for line in sums["aqf"].splitlines():
        match = re.match(r"trial (\d+) (window .*)", line)
        if match:
            trials.setdefault(match.group(1), []).append(match.group(2))
    check(len(trials) == 3 and len({tuple(windows) for windows in trials.values()}) == 1,
          f"{workload}: every trial saw the same queries and churns")

check(len(glob.glob(f"{out}/figures/*.png")) >= 7, "the figures were written")
print(f"\n{'everything ran' if ok else 'problems found'}; results, logs and figures are in {out}")
sys.exit(0 if ok else 1)
EOF
