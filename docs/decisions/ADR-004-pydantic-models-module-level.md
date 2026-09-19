# ADR-004: FastAPI 请求模型必须定义在模块级

## Status
Accepted (2026-09-20, 由一次真实故障倒逼)

## Background
初版把 Pydantic 请求模型定义在 `create_app()` 函数内部。
`test_gateway.py` 中 8 个用例全部返回 **422**，而 `assert` 显示 body 正常传入：

```
assert 422 == 200
```

根因：文件顶部有 `from __future__ import annotations`，
所有注解变成字符串。FastAPI 通过 `get_type_hints` 解析这些字符串时，
用的是**模块全局命名空间**，找不到函数内部定义的类 → 参数校验全部失败 →
静默 422（不是 500，所以很容易误判为"客户端传错了"）。

## Decision
所有请求/响应模型定义在 **模块级**（`gateway.py` 顶部集中声明）。
在文件头写入注释说明原因，防止后人重构时移回函数内。

## Consequences
- 正面：端点正常；422 语义恢复为"真的没传对参数"
- 负面：模型与路由不再"就近定义"，需要跳到文件顶部查看
- 排错口诀（已写入 DEPLOYMENT.md）：**看到全部端点集体 422，先查模型作用域**

## Related ADRs
无
