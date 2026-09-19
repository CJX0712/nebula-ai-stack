"""生成困难评测集：在黄金语料里加入「关键词枢纽文档」作为干扰项。

背景：原始 12 篇 × 24 条查询是**可饱和**的 —— 消融实验里六种排序策略全部 24/24，
说明该集合对排序策略没有区分度，不能用来验证重排设计（这本身是个诚实结论：
评测集必须能区分被评对象，否则它只是仪式）。

困难集的做法：加入 4 篇「枢纽文档」，它们刻意复用查询里的关键词
（线程数、分块、重排权重、端口、评测指标……）但不提供任何结论。
这类文档对 BM25 极其友好，会大量抢占 top-1，正是检验
「重排能否接管排序」的真实压力场景。

    uv run python scripts/make_hard_set.py     # 生成 data/eval/set_hard.json
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

HUB_DOCS = [
    {
        "source": "docs/hub-params.md",
        "text": (
            "参数总览。本仓库涉及大量可调参数：线程数、num_ctx、分块大小、分块重叠、"
            "重排权重、RRF 常数 k、top_k、端口、超时时间、输出预算、嵌入维度、向量距离度量。"
            "每一项参数的取值都会影响最终效果，建议结合实际情况综合评估后再决定，"
            "没有放之四海而皆准的固定值。调参是一个持续迭代的过程。"
        ),
    },
    {
        "source": "docs/hub-faq.md",
        "text": (
            "常见问题。问：量化模型体积多大？答：视参数规模而定。"
            "问：GGUF 是什么格式？答：是一种模型文件格式。"
            "问：为什么十六线程比四线程慢？答：与硬件特性有关。"
            "问：Q4_K_M 与 Q5_K_M 哪个好？答：各有取舍。"
            "问：显存占用如何估算？答：与量化位宽和参数量相关。以上问题都需要结合具体环境判断。"
        ),
    },
    {
        "source": "docs/hub-eval.md",
        "text": (
            "评测方法总述。检索评测常见的指标包括 top-1 命中率、top-3 命中率、"
            "MRR 平均倒数排名、召回率、精确率，以及端到端延迟。"
            "黄金集的质量决定了评测结论的可信度，需要注意集合的区分度与覆盖度。"
            "排序策略、融合权重、分块参数都会影响这些指标，因此评测应当反复进行。"
        ),
    },
    {
        "source": "docs/hub-ops.md",
        "text": (
            "运维与部署注意事项。涉及 Windows 平台的端口绑定、WinError 报错处理、"
            "保留端口段、服务默认端口选择、版本锁定与锁文件、依赖导出、"
            "干净环境复现、一键引导脚本、模型下载源、路径越界防护、"
            "工具调用安全边界等诸多方面，具体配置请按目标环境调整。"
        ),
    },
]


def main() -> int:
    src = ROOT / "data" / "eval" / "set.json"
    data = json.loads(src.read_text(encoding="utf-8"))
    hard = {
        "name": data["name"] + "-hard",
        "description": (
            data["description"] + " ｜困难集：加入 4 篇关键词枢纽干扰文档，"
            "用于检验排序策略（尤其重排是否接管排序）"
        ),
        "corpus": data["corpus"] + HUB_DOCS,
        "queries": data["queries"],
    }
    dst = ROOT / "data" / "eval" / "set_hard.json"
    dst.write_text(json.dumps(hard, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已生成：{dst}")
    print(f"  语料 {len(hard['corpus'])} 篇（原 {len(data['corpus'])} + 干扰 {len(HUB_DOCS)}）"
          f"，查询 {len(hard['queries'])} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
