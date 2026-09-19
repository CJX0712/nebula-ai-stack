"""测试夹具：接管 `tmp_path`，绕开 pytest 临时目录的批量删除。

背景（本项目实测，三层才查到根因）：
1. pytest 默认在 `%TEMP%/pytest-of-<user>` 下创建 `pytest-N`，并在会话收尾把
   多个历史运行目录重命名进 `garbage-<uuid>` 再**递归删除**（实测 337 个文件）。
   在受限/沙箱环境下，这种批量删除会被删除守卫拒绝，收尾抛 `SystemExit: 1`：
   **60 个测试全部通过，汇总行却永远打不出来**（rc=1、stderr 为空）。
2. 改 `--basetemp` 指向仓库内目录后，问题换了个地方发生：环境在 Python 进程里
   改写了 `shutil.rmtree`（`_safe_shutil_rmtree`），pytest 清理 basetemp 时
   在测试 setup 阶段就报 ERROR。
3. 结论：**不要用 pytest 的临时目录机制**。这里用同名夹具覆盖内建的 `tmp_path`，
   改为 `tempfile.mkdtemp()` 创建一次性目录并**不做任何清理** ——
   目录交给操作系统的临时目录策略处理，测试进程不再发起任何递归删除。

为什么不直接改名（如 `workdir`）：覆盖同名夹具可让既有测试零改动继承新行为，
避免"有人在别处写 `tmp_path` 又把坑引回来"。
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_path() -> Path:
    """一次性目录，不回收。

    返回新建目录的 Path；不注册清理钩子，因此不触发任何 bulk 删除。
    需要清理时请自行只删自己创建的文件（本项目测试创建的都很小）。
    """
    return Path(tempfile.mkdtemp(prefix="nebula-test-"))
