# V100 服务器 GGUF 大模型部署指南

## 📑 项目背景与环境说明

本项目旨在将 **Gemma、Qwen3.5** 等小参数大模型部署到 **有限的 V100 服务器环境** 中。由于物理机环境的特殊性，整个实施过程可在严格的离线状态下进行。

**基础环境约束：**

1. **网络状态：** 支持纯离线内网环境
2. **硬件基础：** 纯物理机部署（搭载 V100 计算卡）
3. **初始状态：** 裸机环境，无预装 NVIDIA 显卡驱动

---

## 🎯 为什么选择 GGUF 量化 + llama.cpp？

### 1. GGUF 量化模型的选择原因（V100 最佳适配视角）

| 考量维度 | 说明 |
|----------|------|
| **显存空间有限** | V100 单卡 16GB/32GB 显存，FP16 全量模型（如 27B、35B）无法加载。GGUF Q4_K_M 量化可将显存占用压缩至约 1/4，使大参数量模型在单卡上运行成为可能。 |
| **V100 架构兼容性** | GGUF 基于 llama.cpp 的 CUDA 后端，对 Volta 架构（SM 7.0）有成熟支持，无需依赖 Flash-Attention 等新架构特性。 |
| **精度与速度平衡** | Q4_K_M 在显著缩减显存的同时，保留超过 98% 的原始推理精度，是 V100 场景下性价比最高的量化方案。 |

### 2. 为什么采用 llama.cpp 而非 PyTorch/vLLM？

| 技术限制 | 说明 |
|----------|------|
| **Flash-Attention 不支持** | V100 架构（Compute Capability 7.0）不支持 Flash-Attention 2/3，而 vLLM、Transformers 等主流框架的优化路径严重依赖该特性，在 V100 上无法发挥最佳性能。 |
| **llama.cpp 优势** | llama.cpp 采用纯 CUDA kernel 实现，不依赖 Flash-Attention，对 Volta 架构有完整优化。配合 **连续批处理（Continuous Batching）**，可显著提升并发能力与推理速率。 |
| **部署简洁** | 官方提供 `llama.cpp:server-cuda` Docker 镜像，开箱即用，无需复杂依赖链。 |

---

## 🤗 模型下载（GGUF 量化模型）

本项目使用 **GGUF 量化模型**，模型文件较大（约 15GB–25GB/个），**未纳入 Git 仓库**。请在有网环境从 Hugging Face 或官方渠道下载后放入 `models/` 目录，再拷贝至离线环境。

| 模型 | 用途 | 推荐量化 | 下载方式 |
|------|------|----------|----------|
| [Gemma 3 27B Instruct](https://huggingface.co/google/gemma-3-27b-it) | 通用对话、代码生成 | Q4_K_M | 使用 [llama.cpp 转换](https://github.com/ggml-org/llama.cpp#convert-hugging-face-model-to-gguf) 或从社区获取 `gemma-3-27b-it-Q4_K_M.gguf` |
| [Qwen3.5 35B A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) | 金融语境、公文写作、**图像理解 (VLM)** | Q4_K_M | 同上，获取 `Qwen3.5-35B-A3B-Q4_K_M.gguf` |
| [Qwen3.5 9B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | 高速推理、OCR 校验、**图像理解 (VLM)** | Q4_K_M | 同上，获取 `Qwen3.5-9B-Q4_K_M.gguf` |
| Qwen3.5-35B mmproj（视觉投影器）| Qwen3.5-35B 的多模态视觉编码组件 | FP16 | 与上方 35B 模型配套，获取 `mmproj-F16_qwen35-35.gguf` |
| Qwen3.5-9B mmproj（视觉投影器）| Qwen3.5-9B 的多模态视觉编码组件 | FP16 | 与上方 9B 模型配套，获取 `mmproj-F16_qwen35-9.gguf` |

> **什么是 mmproj？** mmproj（multimodal projector，多模态投影器）是 llama.cpp 实现视觉语言模型（VLM）的关键组件。它扮演"视觉编码器"的角色：将输入图片转化为模型可理解的向量表示，再与文本 token 一起送入语言模型。启动时通过 `--mmproj` 参数加载，与主模型 `.gguf` 文件**分开存储**，便于版本管理与按需加载。

**目录结构要求：**

```
Models/
├── gemma-3-27b-it-Q4_K_M.gguf
├── Qwen3.5-35B-A3B-Q4_K_M.gguf
├── Qwen3.5-9B-Q4_K_M.gguf
├── mmproj-F16_qwen35-35.gguf
└── mmproj-F16_qwen35-9.gguf
```

---

## 🛠️ 第一阶段：系统内核（Kernel）一致性确认

NVIDIA 驱动需要挂载至系统内核，因此确保内核（Kernel）、开发包（Kernel-devel）与头文件（Kernel-headers）的版本 100% 匹配是第一要务。

### 1. 离线资源准备

请提前准备或在介质文件夹 `Base/Kernel_Base` 中确认以下三个核心 RPM 包（针对 `kernel-4.19.90-23.60.v2101.ky10.x86_64`）：

| 组件 | 文件名 | 下载链接 |
|------|--------|----------|
| Kernel 本体 | `kernel-4.19.90-23.60.v2101.ky10.x86_64.rpm` | [麒麟 V10SP1.1 更新源](https://update.cs2c.com.cn/NS/V10/V10SP1.1/os/adv/lic/updates/x86_64/Packages/kernel-4.19.90-23.60.v2101.ky10.x86_64.rpm) |
| Kernel-devel | `kernel-devel-4.19.90-23.60.v2101.ky10.x86_64.rpm` | [麒麟 V10SP1.1 更新源](https://update.cs2c.com.cn/NS/V10/V10SP1.1/os/adv/lic/updates/x86_64/Packages/kernel-devel-4.19.90-23.60.v2101.ky10.x86_64.rpm) |
| Kernel-headers | `kernel-headers-4.19.90-23.60.v2101.ky10.x86_64.rpm` | [麒麟 V10SP1.1 更新源](https://update.cs2c.com.cn/NS/V10/V10SP1.1/os/adv/lic/updates/x86_64/Packages/kernel-headers-4.19.90-23.60.v2101.ky10.x86_64.rpm) |

> 备用：若上述链接不可用，可访问目录 [Packages 索引](https://update.cs2c.com.cn/NS/V10/V10SP1.1/os/adv/lic/updates/x86_64/Packages/) 手动查找对应版本。

### 2. 版本一致性校验

在安装驱动前，必须通过以下命令核对内核版本：

```bash
# 1. 查看当前正在运行的内核版本
uname -r

# 2. 查看系统中实际安装好的开发包版本
rpm -qa | grep kernel-devel
```

### 3. 冲突修复（如版本不一致）

如果上述两条命令输出的版本号不一致，必须进行强制替换：

```bash
# 强制卸载错误的高版本 (请将命令中的输出替换为你查出的实际错误版本号)
sudo rpm -e --nodeps $(rpm -qa | grep kernel-devel)

# 强制安装 100% 匹配的版本（在离线资源目录下执行）
sudo rpm -ivh kernel-devel-*.rpm kernel-headers-*.rpm
```

至此，内核底层依赖准备完毕。

---

## 🚫 第二阶段：彻底禁用 Nouveau 开源驱动

在安装官方 NVIDIA 驱动前，必须彻底封杀系统自带的 Nouveau 驱动。

> **💡 核心原理解析：为什么要「死磕」Nouveau？**
>
> - **一山不容二虎：** Nouveau 是 Linux 默认的开源显卡驱动，开机即接管 GPU。官方闭源驱动（`nvidia.ko`）同样需要底层硬件的独占控制权。若 Nouveau 已在运行，GPU 资源（显存、寄存器等）会被「锁死」，官方驱动直接挂载失败。
> - **深度绑定无法热卸载：** Nouveau 通常被编译进初始内存盘（`initramfs`），在真正的根文件系统挂载前就已驻留内存。因此，简单的 `rmmod` 命令无效，必须通过修改内核引导参数并 **重启物理机** 才能彻底拔除。

### 方案 A：常规封杀法（优先尝试）

```bash
# 1. 屏蔽 nouveau，写入黑名单配置文件
echo -e "blacklist nouveau\noptions nouveau modeset=0" \
    | sudo tee /etc/modprobe.d/blacklist-nouveau.conf

# 2. 更新内核引导镜像
sudo dracut --force

# 3. 重启服务器
sudo reboot
```

重启后检查：执行 `lsmod | grep nouveau`。如果 **没有任何输出**，说明屏蔽成功，可直接跳至 **第三阶段**。如果仍有输出，请执行方案 B。

### 方案 B：GRUB 核心底层封杀法（方案 A 失败时使用）

如果 Nouveau 依然诈尸，我们需要直接修改内核启动参数进行精准打击。

#### 第一步：修改 GRUB 核心引导文件

```bash
sudo vi /etc/default/grub
```

找到以 `GRUB_CMDLINE_LINUX=` 开头的那一行。在行末的双引号 `"` 前加一个空格，并补充以下参数：

```
nouveau.modeset=0 rd.driver.blacklist=nouveau
```

修改后示例：

```
GRUB_CMDLINE_LINUX="...原来的参数... nouveau.modeset=0 rd.driver.blacklist=nouveau"
```

保存并退出（`:wq`）。

#### 第二步：重新编译生成引导菜单（兼容传统 BIOS 与 UEFI）

```bash
# 刷新传统 BIOS 引导配置
sudo grub2-mkconfig -o /boot/grub2/grub.cfg

# 刷新 UEFI 引导配置（若提示找不到路径请忽略）
sudo grub2-mkconfig -o /boot/efi/EFI/kylin/grub.cfg
```

#### 第三步：针对当前内核的精确打包

强制指定当前运行的内核版本重新构建 `initramfs`：

```bash
sudo dracut --force -v /boot/initramfs-$(uname -r).img $(uname -r)
```

#### 第四步：重启验证

```bash
sudo reboot
```

重启后再次运行 `lsmod | grep nouveau`，此时应彻底干净无输出。

---

## 🖥️ 第三阶段：安装 NVIDIA 官方闭源驱动

确认 Nouveau 死亡后，开始安装官方驱动。

**离线资源：** 将驱动文件置于 `Base/` 目录，本项目已包含 `NVIDIA-Linux-x86_64-550.163.01.run`。

| 组件 | 文件名 | 下载链接 |
|------|--------|----------|
| NVIDIA 驱动 (V100 推荐) | `NVIDIA-Linux-x86_64-550.163.01.run` | [NVIDIA 数据中心驱动 550.163.01](https://www.nvidia.com/download/driverResults.aspx/243537/en-us/) · [驱动归档](https://developer.nvidia.com/datacenter-driver-archive) |

```bash
# 请将文件名替换为实际传输到服务器的 .run 文件（本项目为 550.163.01）
sudo ./NVIDIA-Linux-x86_64-550.163.01.run -no-x-check -no-nouveau-check -no-opengl-files
```

### 安装过程中的关键交互选项指南

| 安装程序提示 | 应答选择 | 原因说明 |
|-------------|----------|----------|
| *The distribution-provided pre-install script failed...* | **Continue installation** | 发行版自带脚本失败是正常的，直接跳过。 |
| *Register the kernel module sources with DKMS?* | **No** | 纯内网离线环境不折腾动态内核模块支持。 |
| *Install NVIDIA's 32-bit compatibility libraries?* | **No** | 现代 AI 模型和容器环境无需 32 位支持。 |
| *Run the nvidia-xconfig utility?* | **No** | V100 是纯计算卡，千万不要让它接管图形桌面环境。 |
| *Would you like to rebuild the initramfs?* | **Yes** | **极度重要！** 这会将防冲突配置和官方模块深层注入启动镜像，给驱动加上「开机即生效」的保险，彻底防止 Nouveau 重启诈尸。同时也确保内核在启动早期就能识别硬件 ID。 |

### 安装完成后验证

```bash
nvidia-smi
```

若成功输出显卡信息面板，驱动环节全部结束。

---

## 🐳 第四阶段：容器化环境构建

为确保稳定性并避开复杂的 RPM 依赖，我们采用 Docker 官方静态二进制包进行离线部署，并安装 NVIDIA Container Toolkit 以透传 GPU 给容器。

### 1. 准备离线资源

请在 `Base/Docker_Base` 目录下准备以下文件（或通过给定的外网链接提前下载备用）：

| 组件 | 文件名 | 下载链接 |
|------|--------|----------|
| Docker 静态包 | `docker-24.0.9.tgz` | [Docker 官方静态二进制](https://download.docker.com/linux/static/stable/x86_64/docker-24.0.9.tgz) |
| Docker Compose | `docker-compose-linux-x86_64` | [GitHub Releases](https://github.com/docker/compose/releases)（如 [v2.24.0](https://github.com/docker/compose/releases/download/v2.24.0/docker-compose-linux-x86_64)） |
| NVIDIA Container Toolkit | 见下表 | 见下表 |

**NVIDIA Container Toolkit 依赖包（4 个）：**

| 文件名 | 下载说明 |
|--------|----------|
| `libnvidia-container1-1.14.6-1.x86_64.rpm` | 配置 [NVIDIA 仓库](https://nvidia.github.io/libnvidia-container/) 后执行 `yumdownloader` 离线拉取，或从已有 `Base/Docker_Base` 获取 |
| `libnvidia-container-tools-1.14.6-1.x86_64.rpm` | 同上 |
| `nvidia-container-toolkit-base-1.14.6-1.x86_64.rpm` | 同上 |
| `nvidia-container-toolkit-1.14.6-1.x86_64.rpm` | 同上 |

> **离线获取 NVIDIA Container Toolkit：** 在有网环境配置 `https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo` 后，使用 `yum install --downloadonly` 或 `dnf download` 将 4 个 RPM 下载至 `Base/Docker_Base`。

### 2. 安装 Docker 与注册系统服务

```bash
cd /root/
# 解压二进制包
tar -zxvf docker-24.0.9.tgz

# 迁移可执行文件到系统路径
sudo cp docker/* /usr/bin/

# 注册为 Systemd 系统服务
cat <<EOF | sudo tee /etc/systemd/system/docker.service
[Unit]
Description=Docker Application Container Engine
Documentation=https://docs.docker.com
After=network-online.target firewalld.service
Wants=network-online.target

[Service]
Type=notify
ExecStart=/usr/bin/dockerd
ExecReload=/bin/kill -s HUP \$MAINPID
LimitNOFILE=infinity
LimitNPROC=infinity
TimeoutStartSec=0
Delegate=yes
KillMode=process
Restart=on-failure
StartLimitBurst=3
StartLimitInterval=60s

[Install]
WantedBy=multi-user.target
EOF

# 启动 Docker 并设置开机自启
sudo systemctl daemon-reload
sudo systemctl start docker
sudo systemctl enable docker
```

### 3. 安装 Docker Compose

```bash
# 复制并重命名
sudo cp /root/docker-compose-linux-x86_64 /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 验证安装
docker-compose --version
```

### 4. 安装 NVIDIA Container Toolkit 并配置运行时

此步骤用于打通 Docker 与 V100 物理硬件的壁垒。进入存放 4 个 RPM 包的目录执行：

```bash
# 批量离线安装工具包
sudo rpm -ivh *.rpm

# 配置 Docker 使用 NVIDIA 作为默认运行时
sudo nvidia-ctk runtime configure --runtime=docker

# 重启 Docker 服务使配置生效
sudo systemctl restart docker
```

至此，底层的系统、驱动与容器化环境已全部打通并固化。

---

## 🚀 第五阶段：Docker 镜像下载与推理服务部署

本阶段将部署 **llama.cpp 推理服务** 与 **API 门户网站**，构建多 GPU 大模型推理集群。

### 1. llama.cpp 推理引擎简介

**llama.cpp**（[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)）是专为 GGUF 模型设计的轻量级推理引擎，对 V100 等传统架构有完整 CUDA 支持。

| 特性 | 说明 |
|------|------|
| **无 Flash-Attention 依赖** | 纯 CUDA kernel 实现，完美适配 Volta 架构 |
| **连续批处理 (Continuous Batching)** | 流式请求插入，在不增加首字延迟的情况下显著提升吞吐量 |
| **OpenAI 兼容 API** | 暴露 `/v1/chat/completions` 等标准路径，可直接对接 OpenAI SDK、LangChain 等 |
| **多 GPU 物理隔离** | 通过 Docker 将不同模型映射至独立 GPU，互不干扰 |
| **视觉语言模型 (VLM)** | 通过 `--mmproj` 参数加载视觉投影器，支持图像 + 文本混合输入；**`image_url` 兼容 OpenAI 约定**，可使用 **`data:image/…;base64,…`（内嵌 Base64）** 或 **`http(s)://…` 可访问图片 URL** |

### 2. 镜像下载（有网环境）

在 **Mac** 上拉取时需指定 `--platform linux/amd64`，以生成适用于内网 Linux 服务器的镜像：

```bash
# llama.cpp 推理服务（CUDA 版）
docker pull --platform linux/amd64 ghcr.io/ggml-org/llama.cpp:server-cuda

# Nginx API 门户
docker pull --platform linux/amd64 nginx:alpine
```

### 3. 导出为 .tar 文件（便于 U 盘拷贝）

```bash
# 导出 llama.cpp
docker save -o llama_cpp_server_cuda_amd64.tar ghcr.io/ggml-org/llama.cpp:server-cuda

# 导出 Nginx
docker save -o nginx_alpine_amd64.tar nginx:alpine
```

将生成的 `llama_cpp_server_cuda_amd64.tar`、`nginx_alpine_amd64.tar` 拷贝至 U 盘，导入内网服务器。

### 4. 内网服务器导入镜像

```bash
# 导入 llama.cpp
docker load -i llama_cpp_server_cuda_amd64.tar

# 导入 Nginx
docker load -i nginx_alpine_amd64.tar
```

### 5. 目录结构与启动

确保以下结构就绪：

```
glm-deploy/
├── Models/
│   ├── gemma-3-27b-it-Q4_K_M.gguf
│   ├── Qwen3.5-35B-A3B-Q4_K_M.gguf
│   ├── Qwen3.5-9B-Q4_K_M.gguf
│   ├── mmproj-F16_qwen35-35.gguf      ← Qwen3.5-35B 视觉投影器
│   └── mmproj-F16_qwen35-9.gguf       ← Qwen3.5-9B 视觉投影器
├── www/
│   └── index.html           # API 使用文档页端
├── nginx/
│   └── nginx.conf           # Nginx 静态页配置
├── docker-compose.yml
└── README.md
```

在项目根目录执行：

```bash
docker-compose up -d
```

所有节点显示为绿色（Up）即表示启动成功。服务对外暴露端口 **80**（API 门户）、**8080**（Gemma）、**8081**（Qwen 35B VLM）、**8082**（Qwen 9B VLM）。

### 6. nginx.conf 配置

`nginx/nginx.conf` 用于 API 门户的静态资源与路由配置，示例：

```nginx
server {
    listen 80;
    server_name localhost;
    root /usr/share/nginx/html;
    index index.html;
    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

### 7. 页端访问

启动后，在浏览器访问 **http://\<服务器IP\>:80/** 即可打开 **API 使用文档页**，提供 cURL、Requests、OpenAI SDK、Node.js、LangChain 等多种调用示例，支持一键复制。选择 Qwen（8081/8082）并切换到 **「多模态 (图像+文本)」** 时，示例代码会演示如何通过 **Base64 Data URL** 传图。

### 8. 验证与调用

```bash
# 健康检查（Gemma 3 27B）
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma-3-27b-it-Q4_K_M", "messages": [{"role": "user", "content": "你好"}]}'

# 健康检查（Qwen 3.5 35B VLM）
curl -X POST http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3.5-35B-A3B-Q4_K_M", "messages": [{"role": "user", "content": "你好"}]}'

# 健康检查（Qwen 3.5 9B VLM）
curl -X POST http://localhost:8082/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3.5-9B-Q4_K_M", "messages": [{"role": "user", "content": "你好"}]}'

# 多模态图像理解测试（以 Base64 编码图片为例，适用于 8081/8082）
curl -X POST http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-35B-A3B-Q4_K_M",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<YOUR_BASE64>"}},
        {"type": "text", "text": "请描述图片中的内容。"}
      ]
    }]
  }'
```

#### 多模态传图方式说明（Base64 与 URL）

Qwen VLM 节点（**8081** / **8082**）已按 **OpenAI Chat Completions** 的常见约定解析图片，无需单独上传接口：

| 方式 | `image_url.url` 格式 | 适用场景 |
|------|----------------------|----------|
| **Base64 Data URL** | `data:image/jpeg;base64,<标准 Base64 载荷>`（亦可用 `png`、`webp` 等与模型/解码器支持的 MIME） | **纯内网、离线批处理**；图片由调用方读入并内嵌到 JSON，不依赖推理机访问外网 |
| **HTTP(S) URL** | `https://example.com/image.jpg` 等可被 **llama.cpp 容器内网络** 访问的地址 | 图片托管在对象存储或内网文件服务上时可减少请求体体积 |

**要点：**

- `messages` 里 `role: "user"` 的 `content` 为**数组**时，可同时包含多个 `image_url` 与 `text` 块；顺序一般建议 **先图后文**（与 OpenAI 多模态习惯一致）。
- 将本地文件转为 Base64 时，载荷部分 **不要** 再带 `data:image/...;base64,` 前缀——该前缀仅在最终写入 `url` 字段时使用一次。
- 若使用 URL 传图，请确认容器到该地址的网络与 DNS 可达；**离线环境优先使用 Base64**。

> **注意：** 若使用离线导入的镜像，需确保 `docker-compose.yml` 中的 `image` 名称与 `docker load` 后的镜像名一致。

---

## 📋 实施检查清单

| 阶段 | 检查项 | 验证命令 |
|------|--------|----------|
| 一 | 内核版本一致 | `uname -r` 与 `rpm -qa \| grep kernel-devel` 输出一致 |
| 二 | Nouveau 已禁用 | `lsmod \| grep nouveau` 无输出 |
| 三 | NVIDIA 驱动正常 | `nvidia-smi` 正常输出 |
| 四 | Docker 运行正常 | `docker run --rm hello-world` |
| 四 | GPU 透传可用 | `docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi` |
| 五 | 推理服务启动 | `docker-compose ps` 全部 Up |
| 五 | 页端可访问 | 浏览器打开 `http://<IP>:80/` |
| 五 | Gemma 可用 | `curl -X POST http://localhost:8080/v1/chat/completions ...` |
| 五 | Qwen 35B VLM 可用 | `curl -X POST http://localhost:8081/v1/chat/completions ...` |
| 五 | Qwen 9B VLM 可用 | `curl -X POST http://localhost:8082/v1/chat/completions ...` |
| 五 | 多模态图像输入可用 | 向 8081/8082 发送含 `image_url` 的请求（`data:image/...;base64,...` 或可达的 `http(s)://...`），确认返回图像描述 |

---

*本文档适用于麒麟 V10 SP1.1 操作系统，在纯离线物理机环境中实施。任何接手的工程师均可按阶段顺序执行，知其然，更知其所以然。*

---

## 📥 下载资源总览

| 类别 | 本地路径 | 资源 | 下载链接 |
|------|----------|------|----------|
| **模型** | `Models/` | Gemma 3 27B Q4_K_M GGUF | Hugging Face / 社区转换 |
| **模型** | `Models/` | Qwen3.5 35B A3B Q4_K_M GGUF | Hugging Face / 社区转换 |
| **模型** | `Models/` | Qwen3.5 9B Q4_K_M GGUF | Hugging Face / 社区转换 |
| **模型（mmproj）** | `Models/` | mmproj-F16_qwen35-35.gguf（Qwen 35B 视觉投影器）| Hugging Face / 社区转换 |
| **模型（mmproj）** | `Models/` | mmproj-F16_qwen35-9.gguf（Qwen 9B 视觉投影器）| Hugging Face / 社区转换 |
| **镜像** | `Images/` | llama.cpp server-cuda | `docker pull ghcr.io/ggml-org/llama.cpp:server-cuda` |
| **镜像** | `Images/` | Nginx Alpine | `docker pull nginx:alpine` |
| 一 | `Base/Kernel_Base/` | kernel / kernel-devel / kernel-headers | [麒麟 V10SP1.1 Packages](https://update.cs2c.com.cn/NS/V10/V10SP1.1/os/adv/lic/updates/x86_64/Packages/) |
| 三 | `Base/` | NVIDIA-Linux-x86_64-550.163.01.run | [NVIDIA 数据中心驱动](https://www.nvidia.com/download/driverResults.aspx/243537/en-us/) |
| 四 | `Base/Docker_Base/` | docker-24.0.9.tgz | [Docker 静态包](https://download.docker.com/linux/static/stable/x86_64/docker-24.0.9.tgz) |
| 四 | `Base/Docker_Base/` | docker-compose-linux-x86_64 | [Docker Compose](https://github.com/docker/compose/releases) |
| 四 | `Base/Docker_Base/` | NVIDIA Container Toolkit (4×RPM) | [NVIDIA 仓库](https://nvidia.github.io/libnvidia-container/) + `yum download` |
