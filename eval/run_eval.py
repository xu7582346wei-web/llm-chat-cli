"""Agent 评估：批量跑测试用例，统计成功率、工具准确率"""
import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# 把项目根目录加入 path
sys.path.insert(0, str(Path(__file__).parent.parent))

import agent as agent_module
from agent import Agent, VectorStore, load_documents
from config import KNOWLEDGE_DIR

CASES_FILE = Path(__file__).parent / "test_cases.json"
REPORT_FILE = Path(__file__).parent / "report.json"


def run_one(agent, case):
    """执行单个测试用例，返回结果。"""
    result = {
        "id": case["id"],
        "input": case["input"],
        "tools_called": [],
        "answer": "",
        "elapsed": 0,
        "passed": False,
        "reason": "",
    }

    # 用事件回调捕获工具调用
    tools_called = []

    def on_event(kind, data):
        if kind == "tool_call":
            tools_called.append({
                "name": data["name"],
                "args": data["args"],
            })

    start = time.time()
    try:
        answer = agent.chat(case["input"], verbose=False, on_event=on_event)
        result["answer"] = answer
        result["tools_called"] = tools_called
    except Exception as e:
        result["answer"] = f"ERROR: {e}"
        result["reason"] = f"异常：{e}"
    result["elapsed"] = round(time.time() - start, 2)

    # ---------- 校验 ----------
    expected_tool = case.get("expected_tool", "SKIP")

    # 1. 检查工具调用
    if expected_tool is not None and expected_tool != "SKIP":
        called_names = [t["name"] for t in tools_called]
        if expected_tool not in called_names:
            result["reason"] = f"未调用期望工具 {expected_tool}，实际：{called_names}"
            return result

        # 2. 检查参数（如有）
        if "expected_args" in case:
            for t in tools_called:
                if t["name"] == expected_tool:
                    for k, v in case["expected_args"].items():
                        if str(t["args"].get(k, "")) != str(v):
                            result["reason"] = f"参数不匹配：期望 {k}={v}，实际 {t['args'].get(k)}"
                            return result
                    break

    # 3. 检查回答内容
    if "expected_answer_contains" in case:
        if case["expected_answer_contains"] not in result["answer"]:
            result["reason"] = f"回答中未包含 '{case['expected_answer_contains']}'"
            return result

    result["passed"] = True
    return result


def main():
    # 初始化知识库（如果 Agent 需要）
    store = VectorStore()
    chunks = load_documents(KNOWLEDGE_DIR)
    if chunks:
        store.add(chunks)
    agent_module.STORE = store

    # 加载测试集
    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    print(f"共 {len(cases)} 个测试用例\n")

    all_results = []
    passed = 0

    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case['id']}: {case['input'][:40]}...")

        # 每个用例用全新的 Agent（清空历史）
        agent = Agent()
        result = run_one(agent, case)
        all_results.append(result)

        status = "✓" if result["passed"] else "✗"
        print(f"    {status} ({result['elapsed']}s)")
        if not result["passed"]:
            print(f"    原因：{result['reason']}")
            print(f"    回答：{result['answer'][:100]}")
        if result["passed"]:
            passed += 1

    # ---------- 统计 ----------
    total = len(all_results)
    success_rate = passed / total if total else 0
    avg_time = sum(r["elapsed"] for r in all_results) / total if total else 0

    # 分类统计
    by_category = {}
    for case, result in zip(cases, all_results):
        cat = case.get("category", "other")
        by_category.setdefault(cat, {"total": 0, "passed": 0})
        by_category[cat]["total"] += 1
        if result["passed"]:
            by_category[cat]["passed"] += 1

    report = {
        "timestamp": datetime.now().isoformat(),
        "total": total,
        "passed": passed,
        "success_rate": round(success_rate, 3),
        "avg_elapsed": round(avg_time, 2),
        "by_category": by_category,
        "results": all_results,
    }

    REPORT_FILE.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ---------- 打印 ----------
    print("\n" + "=" * 50)
    print(f"总计：{passed}/{total} 通过（{success_rate * 100:.1f}%）")
    print(f"平均耗时：{avg_time:.2f}s")
    print("\n分类统计：")
    for cat, stats in by_category.items():
        rate = stats["passed"] / stats["total"] * 100
        print(f"  {cat}: {stats['passed']}/{stats['total']} ({rate:.0f}%)")
    print(f"\n详细报告：{REPORT_FILE}")


if __name__ == "__main__":
    main()
