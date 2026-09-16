# 问题 1：图像处理软件包比较（10 分）

本题选择 **Pillow、scikit-image、OpenCV**。按照老师的要求，本文说明三个软件包支持的图像类型与文件格式、读取/显示/写入图片的方法，以及主要图像处理功能，并补充理解代码所需的基础知识。

## 一、先理解：图片读进 Python 后是什么？

### 1. 文件格式、颜色模式和数据类型是三个概念

| 概念 | 回答的问题 | 例子 |
|---|---|---|
| 文件格式 | 图片如何编码并存储在磁盘上？ | JPEG、PNG、TIFF |
| 颜色模式 | 一个像素由哪些分量表示？ | 灰度 L、彩色 RGB、带透明度的 RGBA |
| 数据类型 | 每个分量用什么数值类型保存？ | `uint8`、`uint16`、`float32` |

例如，一张 RGB PNG 图片读取后，可以成为形状为 `(512, 512, 3)`、类型为 `uint8` 的数组。**PNG 是文件格式，RGB 是颜色模式，uint8 是数组元素类型**，三者不能混为一谈。同一种文件格式可能支持多种颜色模式，但并非所有格式都支持所有模式和位深。

- JPEG：常用于照片，常见保存方式为有损压缩；重新保存可能改变像素。
- PNG：无损压缩，支持常见灰度、彩色及透明度表示，适合保存实验结果。
- TIFF：支持多种位深、压缩方式和多页结构，常用于科学图像。TIFF 本身不意味着一定无损，需看具体编码。

### 2. 灰度图和彩色图如何表示？

**灰度图**通常是二维数组，形状为 `(H, W)`。其中 H 是高度（行数），W 是宽度（列数）。对本作业的 8 位灰度图，0 表示黑，255 表示白，中间值表示不同程度的灰。

**RGB 彩色图**通常是三维数组，形状为 `(H, W, 3)`。最后一维依次存储红、绿、蓝三个分量。例如 8 位 RGB 中：

```python
[255, 0, 0]      # 红色
[0, 255, 0]      # 绿色
[0, 0, 255]      # 蓝色
[255, 255, 255]  # 白色
```

`RGBA` 比 RGB 多一个透明度通道。二值图只有两种取值，与包含许多灰度级的灰度图不同；后续题目中的“黑白图像”实际要求的是灰度图。

数组索引从 0 开始：

```python
gray[0, 0]       # 灰度图左上角像素：一个数
rgb[0, 0]        # RGB 图左上角像素：三个数
rgb[0, 0, 0]     # 左上角像素的红色分量
```

Pillow 的 `image.size` 返回 **(宽, 高)**，NumPy 的 `array.shape` 则通常是 **(高, 宽)** 或 **(高, 宽, 通道数)**。非正方形图片尤其容易把这两个顺序弄反。

### 3. uint8 有什么含义？

`uint8` 表示 8 位无符号整数，范围是 0～255。RGB 每通道 8 位，一共是每像素 24 位（不含透明度）。

数值范围不等于文件大小：磁盘上的图片可能经过压缩，而数组保存的是解码后的像素。直接用 `uint8` 数组进行加法等运算可能溢出，因此后续灰度转换与滤波通常应先转浮点数：

```python
work = rgb.astype(np.float64)
```

这里仅改变数值类型，**不会自动把 0～255 缩放成 0～1**。若需要归一化，还要显式除以 255，或使用有明确转换规则的库函数。

## 二、三个软件包支持哪些类型和格式？

下面列出常见能力，不是完整格式清单。实际读写能力取决于软件包版本、编解码器、图像模式和位深；支持一种格式也不意味着该格式的所有变体都支持。

| 比较项 | Pillow | scikit-image | OpenCV |
|---|---|---|---|
| 主要定位 | 图片读写、格式转换与基础编辑 | 科学图像处理与分析 | 图像、视频与计算机视觉 |
| 读取后的主要对象 | `PIL.Image.Image` | NumPy 数组 | NumPy 数组 |
| 常见图像类型 | 二值、灰度、调色板、RGB、RGBA、CMYK；也有部分整数和浮点模式 | 二值、灰度、RGB/RGBA、多维图像；不同算法对维度有要求 | 灰度、BGR/BGRA 及其他多通道图像；不同算子对类型有要求 |
| 常见数值类型 | 常见每通道 8 位，也有 `I`、`F`、`I;16` 等模式 | `bool`、`uint8`、`uint16`、浮点数组等 | `uint8`、`uint16`、`float32`、`float64` 等，依具体算子而定 |
| 常见文件格式 | JPEG、PNG、TIFF、BMP、GIF、WebP 等 | 通过 I/O 后端读取 PNG、JPEG、TIFF、BMP、GIF 等 | JPEG、PNG、TIFF、BMP、WebP 等；更多格式依构建配置而定 |
| 常见彩色通道顺序 | RGB | RGB | BGR |

scikit-image 的核心是数组上的图像算法，文件读写依赖相应后端。它的 `io.imread` 不会把所有输入一律归一化成浮点数；但部分处理函数可能转换类型或要求特定数值范围，使用前应查看函数文档。

本题代码使用 `mandril_color.tif`（RGB 彩色图）和 `cameraman.tif`（8 位灰度图）。文件扩展名都是 `.tif`，颜色模式却不同，正好说明格式与模式的区别。

## 三、如何读取、显示和写入图片？

以下短代码展示基本接口；它们是独立的教学片段。完整运行请使用本目录的 `main.py`。

### 1. Pillow：以图片对象为中心

```python
from PIL import Image

with Image.open("assignment01_images_2026/mandril_color.tif") as image:
    print(image.format)  # TIFF：源文件格式
    print(image.mode)    # RGB：颜色模式
    print(image.size)    # (512, 512)：宽、高
    image.show()        # 使用系统图片查看器显示
    image.save("pillow.png")
```

- `Image.open` 打开文件，像素通常按需加载。
- `with` 保证离开代码块时关闭文件。
- `show` 适合临时查看，依赖系统外部图片查看器。
- `save` 根据扩展名选择输出格式；这里将 TIFF 保存为 PNG，并不需要改变 RGB 模式。
- `np.array(image)` 可以把图片对象转成 NumPy 数组，方便后续数值计算。

### 2. scikit-image：直接使用数组

```python
from skimage import io
import matplotlib.pyplot as plt

image = io.imread("assignment01_images_2026/mandril_color.tif")
print(image.shape, image.dtype)
plt.imshow(image)
plt.axis("off")
plt.show()
io.imsave("scikit_image.png", image)
```

- `io.imread` 返回 NumPy 数组，可直接切片和计算。
- `io.imsave` 把数组编码为图片文件。
- 显示采用 Matplotlib，这是科学图像处理中常见的搭配；不依赖 scikit-image 的旧显示接口。

对于 8 位灰度图，应使用：

```python
plt.imshow(gray, cmap="gray", vmin=0, vmax=255)
```

`cmap="gray"` 指定灰色色图，否则二维数组可能显示成伪彩色。`vmin` 和 `vmax` 固定数值到亮度的映射，便于公平比较不同灰度图。这些参数控制**显示方式**，不会修改原数组。

### 3. OpenCV：注意 BGR 与中文路径

对于普通路径，基本接口如下：

```python
import cv2

image = cv2.imread("image.png", cv2.IMREAD_COLOR)
if image is None:
    raise OSError("读取失败，请检查路径和格式")

cv2.imshow("OpenCV", image)
cv2.waitKey(0)          # 等待按键，并让窗口处理显示事件
cv2.destroyAllWindows()
if not cv2.imwrite("opencv.png", image):
    raise OSError("保存失败")
```

`IMREAD_COLOR` 读取为三通道 BGR；`IMREAD_GRAYSCALE` 读取为灰度；`IMREAD_UNCHANGED` 尽量保留源图的通道及位深，具体仍取决于解码器。

OpenCV 的彩色顺序是 **BGR**。例如同一个红色像素，在 RGB 中是 `[255, 0, 0]`，在 BGR 中是 `[0, 0, 255]`。如果把 BGR 数组直接交给按 RGB 解释数据的 Matplotlib，红、蓝就会颠倒。转换方式是：

```python
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(rgb)
```

`cv2.imshow` 和常规 OpenCV 彩色编码接口本来就使用 BGR，因此不应在传给它们之前随意改成 RGB。

当前作业路径含中文，部分 Windows OpenCV 构建的 `imread`/`imwrite` 对这类路径支持不稳定。因此 `main.py` 使用：

```python
file_bytes = np.fromfile(path, dtype=np.uint8)
bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
success, encoded = cv2.imencode(".png", bgr)
encoded.tofile(output_path)
```

这里 `fromfile` 读到的是**文件编码字节**，并不是已经排列好的像素；`imdecode` 才把它解码为图片数组。反过来，`imencode` 把像素编码成 PNG，`tofile` 将编码结果写入磁盘。完整代码会检查解码和编码是否成功。

## 四、三个软件包提供哪些图像处理功能？

| 功能 | Pillow | scikit-image | OpenCV |
|---|---|---|---|
| 基础编辑与几何变换 | 裁剪、缩放、旋转、翻转、粘贴、文字绘制 | 缩放、旋转、几何变换、重采样 | 缩放、仿射/透视变换、绘图与文字 |
| 颜色与增强 | 模式转换、亮度/对比度/色彩增强 | 颜色空间转换、曝光调整、直方图均衡 | 颜色空间转换、直方图均衡、CLAHE |
| 滤波与边缘 | 常用模糊、锐化和边缘滤镜 | 高斯/中值滤波、去噪、Sobel/Canny 边缘检测 | 卷积、高斯/中值/双边滤波、Sobel/Canny |
| 分割与形态学 | 点运算和有限的相关滤镜 | 阈值分割、分水岭、膨胀/腐蚀、骨架化 | 阈值分割、形态学、连通域、轮廓分析 |
| 测量与视觉 | 偏重图片编辑和格式处理 | 区域属性测量、特征提取、配准 | 特征匹配、相机标定、光流、视频读写等 |

这些术语可以先这样理解：

- **滤波**：根据周围像素计算新像素，例如平滑噪声，后续卷积题会详细涉及。
- **边缘检测**：寻找亮度变化明显的位置，常对应物体边界。
- **分割**：把图像分成不同区域，例如区分前景与背景。
- **形态学**：利用局部形状规则处理区域，例如膨胀使区域扩大，腐蚀使区域收缩。
- **几何变换**：改变像素的位置关系，例如旋转和透视校正。

三者功能有重叠，选择依据是任务，而不是哪个库“绝对更好”：批量缩图、转换格式可优先考虑 Pillow；科学图像分析可考虑 scikit-image；视频和计算机视觉流程可考虑 OpenCV。本文没有做性能测试，因此不据此比较运行速度。

## 五、运行本题代码

七道题共用根目录配置的 `imageproc_hw01` 环境。本机已配置好时，在 Anaconda Prompt 中进入 `01_homework` 目录后运行：

```bat
conda activate imageproc_hw01
python question01/main.py
```

如果在其他机器首次安装环境，先在根目录执行 `conda env create -f environment.yml`。在 IDE 中运行时，也要选择 `imageproc_hw01` 的 Python 解释器。

程序会按顺序完成：

1. 用 Pillow 读取彩色图和灰度图，打印格式、模式及尺寸。
2. 用 scikit-image 和 OpenCV 读取同一张彩色图。
3. 分别保存三个库读写得到的 PNG 图片。
4. 在终端打印数组形状、类型及左上角像素。
5. 保存并显示彩色对比图和灰度图，关闭图形窗口后结束。

运行后 `results/` 中只有以下五张图片：

| 输出文件 | 含义 |
|---|---|
| `pillow.png` | Pillow 保存的彩色图 |
| `scikit_image.png` | scikit-image 保存的彩色图 |
| `opencv.png` | OpenCV 保存的彩色图 |
| `comparison.png` | 三个库读取结果的并排展示 |
| `grayscale.png` | cameraman 灰度显示图 |

三个库分别保存的彩色图保留源像素尺寸；`comparison.png` 和 `grayscale.png` 是带标题的绘图结果，尺寸由绘图设置决定。重复运行会覆盖这五个同名输出，不会修改原始图片。

现有结果可从 [三个库的显示对比](results/comparison.png) 和 [灰度图](results/grayscale.png) 查看。你此前的终端输出中，Pillow、scikit-image 的左上角 RGB 像素均为 `[166,139,62]`，OpenCV 的 BGR 像素为 `[62,139,166]`：红、蓝的存储位置互换，绿色仍在中间。将 OpenCV 结果转为 RGB 后，显示颜色一致。这个像素例子解释了通道顺序，但不能单凭一个像素就断定整幅数组逐元素一致。

### TIFF 的 Unknown field 警告是什么意思？

TIFF 除了像素，还可以保存带编号的标签，用于记录尺寸、编码信息或软件附加的元数据。此前日志中的 `Unknown field with tag 34016` 等提示表示底层 TIFF 解码器不认识某些标签；它不等于“图像像素读取失败”。`Null count` 提示也属于标签解析信息，需要结合后续返回结果判断。

此前这次运行继续输出了正常的数组形状、像素值和保存路径，说明程序完成了读取与保存。学习时先检查是否得到有效数组、尺寸是否合理、显示是否正常，不必为了去掉警告而修改原始 TIFF，也不建议直接屏蔽所有警告。若解码失败，代码会明确报错；若需要精确保留源文件元数据，则还要另外检查标签，不能仅凭图片显示正常作判断。

## 六、运行后如何判断自己理解了？

建议先预测，再核对终端和图片：

1. `cameraman.tif` 为什么只有二维数组，而 `mandril_color.tif` 有三维？
2. 为什么 `image.size` 和 `array.shape` 对非正方形图片可能看起来顺序相反？
3. 为什么 OpenCV 的一个像素数值顺序可能与另外两个库相反，而正确显示后颜色相同？
4. 将灰度显示的 `cmap="gray"` 去掉，颜色变化是否意味着原始数组变化？
5. 把 TIFF 保存成 PNG，改变的是文件编码还是 RGB 三个通道的含义？

正常情况下，三幅彩色显示应在视觉上相同。如果出现明显红蓝颠倒，首先检查通道顺序。单纯看起来相同不等于证明所有像素完全一致；本题以掌握接口和理解数据表示为目标，不额外构建逐像素测试系统。

## 七、官方参考文档

- [Pillow：图像模式和概念](https://pillow.readthedocs.io/en/stable/handbook/concepts.html)
- [Pillow：文件格式支持](https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html)
- [Pillow：Image 接口](https://pillow.readthedocs.io/en/stable/reference/Image.html)
- [scikit-image：图像 I/O](https://scikit-image.org/docs/stable/api/skimage.io.html)
- [scikit-image：数据类型与数值范围](https://scikit-image.org/docs/stable/user_guide/data_types.html)
- [scikit-image：API 总览](https://scikit-image.org/docs/stable/api/api.html)
- [OpenCV：官方文档入口](https://docs.opencv.org/)

查阅时优先关注函数的输入类型、通道顺序、输出范围和返回值，并选择与安装版本相符的文档。根目录 `environment-lock.yml` 记录了已有环境的依赖版本。
