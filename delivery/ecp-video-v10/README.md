# ECP 论文视频 v10 制作包

本目录提供 v10 原始制作包。ZIP 内含 4K 和 1080p 成片、Figure 3 的可编辑 Draw.io 与分层 SVG、18 段英文旁白和母带、中英文解说稿、动画和合成脚本、2,902 张仿真帧以及对应实验记录。解压后从包内 `README.md` 开始。

| 文件 | 大小 | SHA-256 |
|---|---:|---|
| [ECP_v10-production-package.zip](ECP_v10-production-package.zip) | 245,338,165 字节 | `013de064336ffcbc1122d165d5ee95a2a6b3c3b946e35174f95d9228e0e30945` |

ZIP 通过 Git LFS 存储。可以在 GitHub 文件页下载，也可以按以下方式只检出这个目录。

```sh
git lfs install
git clone --depth 1 --filter=blob:none --no-checkout --single-branch --branch codex/ecp-video-v10-20261124 https://github.com/Celveen/JEPA-Stall.git
cd JEPA-Stall
git sparse-checkout set delivery/ecp-video-v10
git checkout codex/ecp-video-v10-20261124
git lfs pull --include="delivery/ecp-video-v10/ECP_v10-production-package.zip"
```

包内的 18 段与母带已验证可重组成与 v10 交付文件逐字节一致的 4K 成片。渲染、音轨重建所需的工具版本与运行步骤见包内《工具与复现》。
