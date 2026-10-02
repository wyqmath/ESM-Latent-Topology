# R5.01 独立审核

审核者：bootstrap_repair（未编写曝光审计脚本或 future guard）。审核时间：2026-10-01T21:51:05+08:00。

## 首轮 FAIL 与整改

1. BLOCKER：未来政策仅从有标注样本建立 current-development 锁定，遗漏重划分后 stage-B 已拟合的背景-only 开发分量。初次核查有 1,941 个现行 development 分量 force_future_development=0。整改将锁定扩展至完整 global splitsets 的 development 分量；与旧开发历史字段分列。
2. BLOCKER：确认侧 FS 六对已读取描述分数/tier，在范围清单登记，但没有进入未来分量闭包。三对 FS 在未来预演仍列 no_use_found。整改单列 P5_descriptive_read 并传播六个分量，未来 FS 96 对均进入 development；不把描述读取混同已观测阴性。
3. Guard 接口加固：候选 native 显式 component_id 原先未核对，现与权威映射对照；全局就绪原先没有校验组件全集覆盖，现要求 labeled+background assignments 完整覆盖 policy IDs。两个回归测试通过。

## 终轮 PASS

独立复算 audit.main，OUT 替换为独立临时目录，读取同一组不可变来源，不修改原划分。10 个产物逐字节相同，包括固定 mtime gzip 背景映射和审计脚本自身输入 SHA。复算记录为 exposure_independent_reproduction.json；脚本 SHA256=b6a8352124e0612421c52fe9fbc9b382da2089aeafd1b8657fd240ab0aa606f7。

核对完整图 68,875 分量、4,858 标注样本、3,538 标注分量；现行 split 没有跨绑定分量；与保存 background group-node 分区一一等价。旧 674 打结输入与有序列的旧 development 精确一致；旧无序 2,278 输入五折均参与有效训练。旧监督使用与旧 L2 operational PU 拟合分列，推理及导入不单独推断为训练。

现有 native 打结评价直接旧开发 107/71，闭包旧开发 131/81，事后剩余 17/27。无序旧开发闭包 294/323，事后剩余 124/171 均为无序类。表中的 fresh_preregistered=0 与 P5 结果已读相符。未来迁移预演的总条目与可评价分母分列，没有将缺序列或冻结域外条目计作可恢复测试数据。

未来政策中全部 current-development 组件锁定，漏项为 0；FS 六个描述读取组件单列锁定。对全部 labeled+background 构建独立只读候选，guard 的 passed=true、ready_for_global_split=true、violations=[]。仅 labeled 验证不得获得全局就绪资格，改ID、漏组件、绑定分量跨集或已用组件迁入评估会拒绝。现行历史划分在新未来政策下被拒绝（行级违例 30,841；包含背景成员，不能解释为独立分量数），符合目的。候选不写入生产数据。

原 knot_resplit_finalize.py 在任何写文件动作前已退休，历史实现有 baseline 存档，避免再次运行遗漏曝光约束的旧分配器。原锁及既有预测保留历史意义；本轮 guard 不宣称恢复原评价独立性。

16 项 unittest 通过（bootstrap 6 + guard 10）。此审核负责曝光/未来政策与接口保护；统计工具实现由 exposure_repair 独立审核，详见 bootstrap_independent_review.json。

文档 exposure_audit.md 已核对，保留 min/max 限制、分量非家族、14 缺结构链、外部 P5.07/P5.08 未伪造映射、已读数据不能重新作为未读验证。新增独立验证仍需外部合格数据及更新完整图和历史使用审计。

相关 guard/退休脚本 SHA256 在 future_guard_independent_review.json。结论适用于这些已审核字节；后续实质修改需追加审核。
