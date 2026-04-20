"""对比 prediction.csv 和 gold.csv 的内容匹配度"""
import csv
import sys
from pathlib import Path

RUNS_DIR = Path(r"C:\Users\win\Desktop\kdd\kdd\artifacts\runs\20260417T121251Z")
GOLD_DIR = Path(r"C:\Users\win\Desktop\kdd\kdd\data\public\output")


def read_csv_rows(path: Path) -> list[list[str]]:
    """读取 CSV，返回所有行的列表（包含表头）"""
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        return [row for row in reader if any(cell.strip() for cell in row)]


def normalize_rows(rows: list[list[str]]) -> list[tuple[str, ...]]:
    """标准化：小写、去空格、排序，方便比较"""
    normalized = []
    for row in rows:
        norm = tuple(cell.strip().lower() for cell in row)
        normalized.append(norm)
    return sorted(normalized)


def compare_task(task_id: str) -> dict:
    pred_path = RUNS_DIR / task_id / "prediction.csv"
    gold_path = GOLD_DIR / task_id / "gold.csv"

    result = {"task_id": task_id, "has_prediction": pred_path.exists(), "has_gold": gold_path.exists()}

    if not result["has_prediction"]:
        result["status"] = "MISSING_PRED"
        return result

    pred_rows = read_csv_rows(pred_path)
    gold_rows = read_csv_rows(gold_path)

    if not gold_rows:
        result["status"] = "NO_GOLD"
        return result

    # 分离表头和数据行
    pred_header = pred_rows[0] if pred_rows else []
    pred_data = pred_rows[1:] if len(pred_rows) > 1 else []
    gold_header = gold_rows[0] if gold_rows else []
    gold_data = gold_rows[1:] if len(gold_rows) > 1 else []

    result["pred_columns"] = pred_header
    result["gold_columns"] = gold_header
    result["pred_rows"] = len(pred_data)
    result["gold_rows"] = len(gold_data)

    # 标准化后比较数据行（忽略顺序）
    pred_norm = normalize_rows(pred_data)
    gold_norm = normalize_rows(gold_data)

    pred_set = set(pred_norm)
    gold_set = set(gold_norm)

    matched = pred_set & gold_set
    extra = pred_set - gold_set
    missing = gold_set - pred_set

    # Recall（标准答案覆盖度）
    if len(gold_set) > 0:
        recall = len(matched) / len(gold_set)
    else:
        recall = 1.0 if len(pred_set) == 0 else 0.0

    result["matched"] = len(matched)
    result["gold_total"] = len(gold_set)
    result["extra_rows"] = len(extra)
    result["missing_rows"] = len(missing)
    result["recall"] = recall

    # 列名匹配
    pred_cols_lower = {c.strip().lower() for c in pred_header}
    gold_cols_lower = {c.strip().lower() for c in gold_header}
    result["col_match"] = pred_cols_lower == gold_cols_lower

    # 状态判定
    if len(pred_set) == len(gold_set) and len(matched) == len(gold_set):
        result["status"] = "EXACT_MATCH"
    elif recall == 1.0 and len(extra) > 0:
        result["status"] = "EXTRA_ROWS"
    elif recall >= 1.0:
        result["status"] = "FULL_RECALL"
    elif recall >= 0.5:
        result["status"] = "PARTIAL_MATCH"
    else:
        result["status"] = "POOR_MATCH"

    return result


def main():
    # 收集所有有 gold 的任务
    gold_tasks = sorted([d.name for d in GOLD_DIR.iterdir() if d.is_dir()])
    # 收集有 prediction 的任务
    pred_tasks = sorted([d.name for d in RUNS_DIR.iterdir() if d.is_dir()])

    all_tasks = sorted(set(gold_tasks) | set(pred_tasks))

    results = []
    for task_id in all_tasks:
        r = compare_task(task_id)
        results.append(r)

    # 打印统计
    statuses = {}
    for r in results:
        s = r["status"]
        statuses[s] = statuses.get(s, 0) + 1

    print("=" * 80)
    print(f"{'KDD Cup Benchmark Compare Report':^60}")
    print("=" * 80)
    print(f"\n总任务数: {len(gold_tasks)}")
    print(f"已跑任务: {len(pred_tasks)}")
    print(f"有 prediction: {sum(1 for r in results if r.get('has_prediction'))}")
    print(f"无 prediction: {sum(1 for r in results if not r.get('has_prediction'))}")

    print(f"\n--- 状态统计 ---")
    print(f"  [OK]  EXACT_MATCH (完全匹配):      {statuses.get('EXACT_MATCH', 0)}")
    print(f"  [!!]  EXTRA_ROWS (多了一些行):    {statuses.get('EXTRA_ROWS', 0)}")
    print(f"  [OK]  FULL_RECALL (全部覆盖):       {statuses.get('FULL_RECALL', 0)}")
    print(f"  [~~]  PARTIAL_MATCH (部分匹配):    {statuses.get('PARTIAL_MATCH', 0)}")
    print(f"  [XX]  POOR_MATCH (匹配差):         {statuses.get('POOR_MATCH', 0)}")
    print(f"  [--]  MISSING_PRED (无输出):       {statuses.get('MISSING_PRED', 0)}")

    # 平均 recall
    recalls = [r["recall"] for r in results if "recall" in r]
    if recalls:
        avg_recall = sum(recalls) / len(recalls)
        print(f"\n  📊 平均 Recall (含无输出=0): {avg_recall:.2%} ({len(recalls)} 个有效)")
        matched_count = sum(1 for r in recalls if r >= 1.0)
        print(f"  📊 完全覆盖的任务: {matched_count}/{len(gold_tasks)}")

    # 详细表格
    print(f"\n--- 详细对比 ---")
    print(f"{'Task':<12} {'状态':<16} {'Recall':<8} {'匹配':<8} {'标准':<8} {'多余':<6} {'缺失':<6}")
    print("-" * 80)

    for r in results:
        if "recall" in r:
            recall_str = f"{r['recall']:.0%}"
            matched_str = str(r['matched'])
            gold_str = str(r['gold_total'])
            extra_str = str(r['extra_rows'])
            missing_str = str(r['missing_rows'])
        else:
            recall_str = "-"
            matched_str = "-"
            gold_str = "-"
            extra_str = "-"
            missing_str = "-"

        status = r['status']
        # 状态图标
        icon = {
            "EXACT_MATCH": "[OK]",
            "EXTRA_ROWS": "[!!]",
            "FULL_RECALL": "[OK]",
            "PARTIAL_MATCH": "[~~]",
            "POOR_MATCH": "[XX]",
            "MISSING_PRED": "[--]",
            "NO_GOLD": "[  ]",
        }.get(status, "[??]")

        print(f"{r['task_id']:<12} {icon} {status:<13} {recall_str:<8} {matched_str:<8} {gold_str:<8} {extra_str:<6} {missing_str:<6}")

    # 打印列名不匹配的任务
    col_mismatch = [r for r in results if r.get("col_match") == False and "pred_columns" in r]
    if col_mismatch:
        print(f"\n--- 列名不匹配详情 ---")
        for r in col_mismatch:
            print(f"  {r['task_id']}:")
            print(f"    预测列名: {r['pred_columns']}")
            print(f"    标准列名: {r['gold_columns']}")


if __name__ == "__main__":
    main()
