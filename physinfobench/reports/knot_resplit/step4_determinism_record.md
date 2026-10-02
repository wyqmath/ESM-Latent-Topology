# 步骤 4 确定性留档（BLOCKER-1 修复后）

- 2026-09-25 18:4x：修复 finalize bg 写出块（csv.writer 覆盖 f 变量的 botched edit）后，
  连续两次完整运行 checksums.txt 聚合 sha256 全同（daf9ec40514b9495…）：
  split_manifest=41676fd7…、fs_l2_background=**a4b9fea6…**（mtime=0 封装，内容与交付版逐字节相同）、
  knot_structural_edges=835bbba5…。
- 历史说明：交付版 bg.gz（2c8fb8e4…）由旧 gzip.open 路径产生（头含 mtime/FNAME，内容与新版
  逐字节相同但封装不同）；其"三轮 sha 全同"记录为空转（当时脚本已损坏未写文件）——已按
  审核 BLOCKER-1 如实更正。checksums.txt 已更新为 a4b9fea6 版并可确定性再生成。
