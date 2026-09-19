# ADR-008: 自检判据必须来自结构化计数，不得抓取人眼输出

## Status
Accepted (2026-09-20)

## Background
`scripts/verify.py` 是"干净环境是否可复现"的唯一机器判据，它自己必须先可靠。
本 ADR 记录它连续暴露的两次缺陷——**两次都是"报告与事实不一致"，而不是代码 bug**。

### 第一次：措辞错误导致排障方向错误
```python
summary = next((ln for ln in reversed(out.splitlines()) if "passed" in ln or "failed" in ln), "")
record("pytest 全绿", ok, summary or f"rc={rc} 输出为空（疑似进程被终止）")
```
实际输出里明明有 `ERROR: file or directory not found: tests`（不含 `passed`/`failed` 字样），
却被报告成"输出为空（疑似进程被终止）"。**报告的因果解释是错的**，
按它去查内存/进程问题会白费一小时。

### 第二次：偶发形态让"汇总行"判断彻底失效
修好措辞后（改为"无汇总行即重试"），出现新的偶发失败：
```
rc=1  本次运行未产出 pytest 汇总行
stdout 尾部='................................................(60 个点)......... [100%]'
```
含义是：**60 个测试全部跑完，进程在打印汇总行之前就退出了**。
此时无论怎么解析 stdout 都拿不到可靠结论——判据本身不可靠。

### 第三次：真凶是 pytest 收尾的批量删除被环境守卫拦截
改用进程内执行并把**异常类型**打印出来后，真凶暴露：

```
[safe-delete][SAFE_DELETE_BULK_CONFIRM_REQUIRED] {"count":337,"threshold":50,...,
 "targets":["...\\AppData\\Local\\Temp\\pytest-of-Administrator\\garbage-<uuid>"]}
pytest 进程内执行异常：SystemExit: 1
```

pytest 默认把临时目录放在系统临时根 `%TEMP%/pytest-of-<user>`，
并在会话收尾时**回收历史运行目录**：把多个旧目录重命名进一个 `garbage-<uuid>`
再递归删除。一次要删 **337 个文件**，超过守卫的单轮阈值（50）而被拒绝，
于是收尾抛 `SystemExit: 1` —— **60 个测试全部通过，却永远打不出汇总行**。

这也解释了"偶发"：垃圾目录累积到超阈值后才必然触发。

### 第四次：`--basetemp` 只是把问题挪了个位置
第一次修法是指定 `--basetemp` 到仓库内目录。跑通了三次，**但第一次完整自检又红了**：

```
ERROR at setup of test_eval_is_isolated_from_production_index
_safe_shutil_rmtree(path='.../data/.pytest-tmp', ...)
```

环境在 Python 进程内**改写了 `shutil.rmtree`**（`_safe_shutil_rmtree`），
pytest 清理 basetemp 时同样被拦，这次是在测试 **setup 阶段**报 ERROR。
**根因没变：只要 pytest 发起递归删除，就会被拦。**

## Decision
1. **判据必须来自结构化数据，不来自人眼输出。**
   `verify.py` 在**本进程内**调用 `pytest.main(..., plugins=[collector])`，
   插件按 `report.when == "call"` 精确统计 `passed / failed / errors / skipped`，
   以**计数**（而非退出码、更非输出文本）作为唯一判据：
   ```python
   ok = rc == 0 and c.failed == 0 and c.errors == 0 and c.passed > 0
   ```
2. **不让 pytest 使用它的临时目录机制**：`tests/conftest.py` 用同名夹具覆盖内建
   `tmp_path`，改为 `tempfile.mkdtemp()` 且**不做任何清理** ——
   测试进程不再发起递归删除，问题从根上消失。
   （不用 `--basetemp` 绕行，因为那只是换了个被拦的位置。）
3. **报告必须回显原始计数与异常类型**，而不是只给 PASS/FAIL：
   `passed=60 failed=0 errors=0 skipped=0 exit=0` —— 任何异常一眼可见；
   且 `SystemExit` 与 `AssertionError` 必须能被区分（处置方式完全不同）。
4. **严格区分「运行没跑起来」与「测试没通过」**，只对前者重试；
   只因后者重试等于掩盖缺陷。
5. 保留"完整 stdout + 截断尾部"双份：尾部给人看，全文给机器解析
   （JSON 报告远超 1500 字符，只留尾部必然解析失败）。

## Consequences
- 正面：连续三次复跑结果逐位相同（`exitcode=0`、`60 passed`）；判据变成确定性的
- 正面：报告与事实一致，排障不再被误导
- 负面：`tmp_path` 不再自动回收，临时目录会留在系统 temp 下。
  代价可接受（测试产物都很小），换来的是"测试不会因为环境守卫而假失败"
- 负面：pytest 与 verify 同进程，测试若污染全局状态会影响 verify 自身
  （当前 60 个测试均无此问题；若将来出现，再回到子进程 + junitxml 方案）
- **通用教训（三层才查到底）**：
  ① 质量门禁必须先证明自己不说谎；
  ② 报告里的因果解释必须是**实测**得来，不能是把症状套进常见模板；
  ③ 遇到"测试全过但判失败"，先怀疑**收尾阶段**而不是测试本身。

## Related ADRs
ADR-004（同属"报错信息与真实根因不一致"家族）
