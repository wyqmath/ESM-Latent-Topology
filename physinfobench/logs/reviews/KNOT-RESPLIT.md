# KNOT-RESPLIT 独立审核记录（打结结构绑定重划分，方案 A；覆盖 P3.03 收口）

- 一审时间：2026-09-25 18:2x–18:5x；窄范围复审：18:5x–19:0x Asia/Shanghai（真实时钟）
- 审核人：独立审核（ZCode 代理 agent_af9b6ce9，未参与执行）
- 对象：KNOT-RESPLIT 五步执行（步骤 1 冻结校验 / 2 曝光核查 / 3 预演 / 4 新版划分生成+审计+重跑 / 5 本审核前置材料）+ P3.03 收口材料 + 新版划分 data/splits 三件套 + 探针/L2/基线重跑结果。

## 一审结论：不通过（1 BLOCKER / 2 MAJOR / 7 MINOR）

科学内容**全部独立复算通过**：3,854/2,998/679（301/230/148）口径、mode-0 逐字节复现、68,875 组、
L6=0/2,998、原 679 跨集合对全部同集合、审计 L1–L5 全零+L4=55+L4b=253、S1/S2 35/35 全 dev、
stage-B 宇宙 342,014 零泄漏、基线与 percentile 0.00031、旧版留档与 git 3c5a596 逐字节一致。
不通过原因全在再生成链条与陈述：

- **BLOCKER-1**：knot_resplit_finalize.py 的 bg 写出块损坏（csv.writer 覆盖 f 变量，脚本不可运行）；
  交付 bg.gz 由旧 gzip.open 路径产生（头含 mtime/FNAME，sha 2c8fb8e4 不可由声称的确定性路径再生）；
  step4 报告"三轮 sha 全同"系空转。
- **MAJOR-1**：曝光登记不穷举（fetch preflight/预演构成/finalize 构成 3 条漏登记）。
- **MAJOR-2**：T-DISORDER-RES 0.904±0.049 实际 CV 分母=4 个双类蛋白，以 n=2,279 呈现未披露。
- MINOR-1～7：1,521/1,019 口径混用；exposure 行 4 链数口径；TODO 三处陈旧（KNOT-RESPLIT 进度、
  P3.03 树行、头部时间戳）；P3.03"disorder 0.906"无留档出处；run_baselines ._ 过滤未列名；
  未列名新文件归属；冻结校验脚本仅步骤 1 时点可重跑未注明。

## 整改与复审结论：通过（0 BLOCKER / 0 MAJOR / 4 MINOR，随过审批次强制完成后提交）

- BLOCKER-1 修复独立实证：bg 写出块修正后**连续两次完整运行 checksums 聚合 sha 全同**
  （daf9ec40…）；三文件 manifest 41676fd7…/edges 835bbba5…（与交付一致）、bg **a4b9fea6…**
  （mtime=0，内容与交付逐字节相同）；仓库落地已于 18:56 重生成、FLG=0x0、解压内容逐字节相同；
  step4_determinism_record.md 留档含空转更正。
- MAJOR-1/2 修复独立实证：exposure_log 现 11 数据行（新增 3 行字段齐整 influenced=false）；
  双类蛋白独立复算全库 5/新 dev 4 与披露一致。
- MINOR-5/6/7 落地；MINOR-3（TODO 三处）按用户步骤 5 顺序并入过审批次——复审确认合规，
  条件=该批次必须包含 MINOR-3 三处+残余 MINOR（已执行）。
- 残余 4 MINOR（MINOR-4 disorder 留档值口径、R-1 exposure 行 9 链数口径澄清、R-2 step4 报告
  陈旧时点单元格、R-3 qc 手工追加注明）已随过审批次全部完成。

## 验收对照（用户五步）

1. 冻结校验 ✓（1,006 覆盖+14 未验证不当作通过）；2. 曝光核查 ✓（8 行登记、influenced=No、
  无降级、后续计算只用 ID+结构+已登记约束）；3. 预演 ✓（可行判定+四项论证）；4. 新版生成 ✓
  （留档旧版、L6=0、差异驱动重跑、全部任务受影响均重跑）；5. 本审核 ✓ 过审后批次执行完毕。
