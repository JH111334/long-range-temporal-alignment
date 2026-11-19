import json
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
results_dir = repo_root / "Documents" / "Documenting" / "Results" / "STAGE03_3B_V2_20261026"

paths = {
    "Arm 1: Ticktack Base": results_dir / "ticktack_3b_base_canonical_results.json",
    "Arm 2: Targeted Base": results_dir / "stage03_3b_base_targeted_results.json",
    "Arm 3: MaT-LoRA FoNE": results_dir / "stage03_3b_matlora_fone_results.json",
    "Arm 4: Static MLP LoRA": results_dir / "stage03_3b_static_mlp_lora_results.json",
}

for name, p in paths.items():
    data = json.load(open(p, encoding="utf-8"))
    if "Ticktack_3B_Base" in data:
        data = data["Ticktack_3B_Base"]
    cases = data.get("case_results", {})

    # Task mapping into 2 research branches:
    # 1. Nhận diện thời gian theo năm (Point Fact): task DB (In year Y, index was ...)
    # 2. Suy luận thời gian theo năm (Temporal Reasoning):
    #    - So sánh thứ tự tương đối: task Rel (In year Y1 vs Y2, was it higher? Yes/No)
    #    - Biến động khoảng năm: tasks MM, RB (Min/Max, Mean in range Y1-Y2)
    #    - Quỹ đạo thời gian: tasks WB, TB (Trend across years, trajectory)
    ident_cases = [c for c in cases.values() if c.get("task_id") == "DB"]
    rel_cases = [c for c in cases.values() if c.get("task_id") == "Rel"]
    interval_cases = [c for c in cases.values() if c.get("task_id") in ["MM", "RB"]]
    traj_cases = [c for c in cases.values() if c.get("task_id") in ["WB", "TB"]]
    reasoning_cases = [c for c in cases.values() if c.get("task_id") != "DB"]

    def calc_accs(clist):
        n = len(clist)
        if n == 0:
            return (0.0, 0.0, 0.0, 0)
        can = sum(1 for c in clist if c.get("is_correct")) / n * 100.0
        deb = sum(c.get("debiased_score", 0.0) for c in clist) / n * 100.0
        st = sum(1 for c in clist if c.get("is_strict")) / n * 100.0
        return round(can, 2), round(deb, 2), round(st, 2), n

    overall = (data.get("canonical_acc"), data.get("debiased_mean_acc"), data.get("strict_acc"))
    id_res = calc_accs(ident_cases)
    rel_res = calc_accs(rel_cases)
    inter_res = calc_accs(interval_cases)
    traj_res = calc_accs(traj_cases)
    re_res = calc_accs(reasoning_cases)

    cf = data.get("cf_54p", {})
    both = cf.get("both_correct", 0)
    total_cf = cf.get("total", 54)
    rate_cf = cf.get("rate", round(both / total_cf * 100, 2) if total_cf else 0)

    print(f"\n==========================================")
    print(f"=== {name} ===")
    print(f"==========================================")
    print(f"Toàn bộ tập kiểm chuẩn (N={data.get('total', len(cases))}):")
    print(f"  - Canonical Accuracy: {overall[0]}%")
    print(f"  - Debiased Mean Accuracy: {overall[1]}%")
    print(f"  - Strict Accuracy (4-fold): {overall[2]}%")
    print(f"\n[NHÁNH 1] Nhận diện Thời gian theo Năm (Point-in-Time Fact Recognition, N={id_res[3]}):")
    print(f"  - Canonical: {id_res[0]}% | Debiased: {id_res[1]}% | Strict: {id_res[2]}%")
    print(f"\n[NHÁNH 2] Suy luận Thời gian theo Năm (Temporal Reasoning by Year, N={re_res[3]}):")
    print(f"  - Tổng hợp Suy luận (Reasoning All): Canonical={re_res[0]}% | Debiased={re_res[1]}% | Strict={re_res[2]}%")
    print(f"  - So sánh thứ tự hai năm (Pairwise Order Rel, N={rel_res[3]}): Canonical={rel_res[0]}% | Debiased={rel_res[1]}% | Strict={rel_res[2]}%")
    print(f"  - Suy luận khoảng biến thiên năm (Interval Dynamics, N={inter_res[3]}): Canonical={inter_res[0]}% | Debiased={inter_res[1]}% | Strict={inter_res[2]}%")
    print(f"  - Suy luận quỹ đạo thời gian đa năm (Multi-Year Trajectory, N={traj_res[3]}): Canonical={traj_res[0]}% | Debiased={traj_res[1]}% | Strict={traj_res[2]}%")
    print(f"  - Suy luận phản thực nghiệm đối ngẫu (Counterfactual 54 Pairs): {both}/{total_cf} ({rate_cf}%)")
