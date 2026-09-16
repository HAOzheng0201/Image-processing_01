# 图像处理编程作业 01

本作业包含七道题，依次学习图像读写、数组索引、灰度转换、二维卷积、高斯核、高斯滤波和图像分块。各题独立放在 `questionXX/` 中，共用一套 Anaconda 环境和原始图片。

每题的 `README.md` 解释题意、公式、手算例子及结果；`main.py` 包含实现和中文注释；`results/` 只保存程序输出。阅读时可以先手算小例子，再对照代码，最后查看真实图片的结果。

## 题目导航

| 题目 | 核心内容 | 重点理解 | 输出 |
|---|---|---|---|
| [第 1 题](question01/README.md) | Pillow、scikit-image、OpenCV | 文件格式、数组形状、RGB/BGR | 5 张 PNG |
| [第 2 题](question02/README.md) | `scanLine4e` 灰度扫描 | 行列索引、切片、扫描曲线 | 2 张 PNG |
| [第 3 题](question03/README.md) | `rgb1gray` 灰度转换 | 平均法、NTSC 权重、浮点计算 | 6 张 PNG |
| [第 4 题](question04/README.md) | `twodConv` 二维卷积 | 核翻转、邻域乘加、边界填充 | 3 张 PNG |
| [第 5 题](question05/README.md) | `gaussKernel` 高斯核 | 标准差、离散采样、归一化 | 5 个 TXT、1 张 PNG |
| [第 6 题](question06/README.md) | 高斯滤波与比较 | 参数匹配、误差、边界效应 | 28 张 PNG |
| [第 7 题](question07/README.md) | `split_image` 图像分块 | 块尺寸、块顺序、补齐与重建 | 12 个 NPZ、14 张 PNG |

第 6 题直接导入第 3、4、5 题的函数，需要保留这些题的目录和代码，但不需要先运行它们。第 7 题沿用第 4 题的填充值规则，不调用卷积函数；彩色原图保留 RGB 通道。

## 目录和数据

```text
01_homework/
├─ README.md
├─ environment.yml                 # 七道题共用的依赖配置
├─ environment-lock.yml            # 已有 Windows 环境的版本记录
├─ 2026年秋季图像处理编程作业01.pdf  # 老师的题目和提交要求
├─ assignment01_images_2026/        # 原始图片
└─ question01/ … question07/        # 每题：README.md、main.py、results/
```

下表统一按 NumPy 的“高、宽、通道”顺序列出，不使用 Pillow 的“宽、高”顺序。

| 原图 | 数组形状 | 颜色与类型 |
|---|---|---|
| `cameraman.tif` | `(256, 256)` | 灰度，`uint8` |
| `einstein.tif` | `(679, 800)` | 灰度，`uint8` |
| `mandril_color.tif` | `(512, 512, 3)` | RGB，`uint8` |
| `runner.jpg` | `(321, 481, 3)` | RGB，`uint8` |

## 共用环境

七道题共用 Anaconda 环境 `imageproc_hw01`。在 Anaconda Prompt 中进入本目录，首次配置时运行：

```bat
conda env create -f environment.yml
```

本机已经创建该环境，无需重复安装。每次使用时激活：

```bat
conda activate imageproc_hw01
```

`environment.yml` 指定 Python 3.11，以及 NumPy、Pillow、Matplotlib、SciPy、scikit-image、OpenCV。`environment-lock.yml` 记录已有 Windows 环境的依赖版本，便于核对；它不是保证跨平台完全相同的通用锁文件。日常配置使用 `environment.yml`，各题不再创建独立环境。

在 VS Code 中，通过“Python: Select Interpreter”选择 `imageproc_hw01`。终端激活环境和编辑器选择解释器是两件事；出现找不到软件包时，先核对正在使用的解释器。第 6 题使用 SciPy 的 `gaussian_filter(..., radius=...)` 参数，如果旧环境提示不认识 `radius`，应核对 SciPy 版本和当前环境，而不是直接删掉参数，因为它决定比较时的核尺寸。

## 运行作业

在 Anaconda Prompt 中进入 `01_homework`，激活环境后，按需选择一条命令执行：

```bat
python question01/main.py
python question02/main.py
python question03/main.py
python question04/main.py
python question05/main.py
python question06/main.py
python question07/main.py
```

第 1～5 题保存结果后会显示 Matplotlib 窗口，关闭窗口后程序结束；第 6～7 题保存结果并关闭画布，不自动弹窗。当前程序没有 `--show` 参数，不需要添加它。第 6 题使用直接卷积，处理大图和较大高斯核时耗时会增加，终端会打印进度。

程序根据脚本位置寻找原图，将结果写入对应题目的 `results/`。重复运行会覆盖同名输出，不会改写原始图片。独立图像保持输入的空间尺寸；带标题、坐标轴的对比图是绘图画布，不能用画布尺寸判断算法是否改变图像尺寸。

## 如何阅读已有结果

当前目录已有七道题的程序输出。README 中的小矩阵、纯色像素和常数图例子用于手算；真实图片的观察应对照对应结果文件。第 5 题的 TXT 是核权重，第 7 题的 NPZ 是完整块数组，都属于计算结果。

有三个容易混淆的地方：

1. **显示相似不等于数值完全相同。** 第 6 题应看浮点 MAE 和最大绝对差，不能只比较取整后的 PNG。
2. **差值不总是错误。** 第 3 题比较两种不同灰度定义，第 6 题的边界实验比较不同延伸假设；出现差异有明确原因。
3. **满足一个性质不足以证明全部正确。** 高斯核和为 1 还需检查形状与对称性；分块重建一致还需核对块编号与原图坐标。

每题文档中的结果链接可直接打开。修改算法或参数后，应重新运行对应题目，再更新有关结论，避免旧图片与新代码混用。

## 实验报告与提交

各题 README 是学习说明，可以帮助组织报告，但不能替代题目要求的报告 PDF。报告中应交代方法、参数、代表性结果与解释，尤其是第 3 题两种灰度方法的差异，以及第 6 题库函数和边界方式的比较。现有 Word 报告的正文与版式不在本说明的核对范围内。

按作业 PDF，提交报告须为 PDF，压缩包命名为“作业01+学生姓名”，截止时间为 **2026 年 10 月 6 日 22:00**，通过课程平台提交。整理提交副本时保留必要的代码、数据、环境配置和结果，排除 `.git`、缓存、临时预览及 Word 临时锁文件；无需为打包删除工作目录中的文件。具体要求以老师发布的作业文件及后续通知为准。
