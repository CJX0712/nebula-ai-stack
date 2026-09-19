# ADR-002: 移除 jieba，中文改用「单字 + 双字」切分

## Status
Accepted (2026-09-20)

## Background
`jieba` 只发布 sdist（无 wheel）。本机 `uv sync` 构建时失败：

```
Failed to build `jieba==0.42.1`
Call to `setuptools.build_meta:__legacy__.build_wheel` failed (exit code: 1)
```

项目第一原则是"干净环境一键复现"，不允许依赖本机编译链（本机无 MSVC、无 cmake）。

## Decision
自实现中文切分：CJK 连续串取**单字 + 相邻双字**，latin/数字按词切并小写归一。
约 15 行，零依赖，全平台确定性。

## Consequences
- 正面：移除一个编译依赖；分词结果完全可复现，测试更稳
- 负面：中文 BM25 精度略低于词典分词；对未登录词的处理不如 jieba
- 缓解：BM25 只是混合检索的一路，稠密检索（bge-m3）承担语义召回；
  若日后需词典分词，可在 `sparse.tokenize` 内替换而**不影响任何调用方**

## Related ADRs
ADR-001（同属"零编译依赖"约束）
