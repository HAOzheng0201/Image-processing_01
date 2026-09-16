"""问题 1：使用 Pillow、scikit-image 和 OpenCV 读取、显示、保存图片。"""

# import 用于导入工具；from ... import ... 表示只导入其中的某个对象。
# Path 是 Python 自带的路径工具，能方便地拼接目录、文件名。
from pathlib import Path

# 安装的软件包名与导入名不一定相同：OpenCV 导入为 cv2，Pillow 导入为 PIL。
import cv2
# as 表示给模块起一个短名称，后面用 plt、np 调用它们的功能。
# pyplot 负责绘图；NumPy 负责数组，图片的像素可以存放在数组中。
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
# scikit-image 导入为 skimage；io 是其中负责图片输入/输出的模块。
from skimage import io


def main():
    # def 定义函数。这里把整个实验放进 main，直到文件末尾调用 main() 才执行。
    # Python 用缩进表示哪些语句属于同一个函数、循环或条件分支。

    # ---------- 1. 准备输入、输出路径 ----------
    # __file__ 是当前脚本的路径；resolve() 将其解析成绝对路径；parent 取上一级目录。
    # 因此 question_dir 指向 question01，而不取决于你从哪个终端目录启动程序。
    question_dir = Path(__file__).resolve().parent
    # Path 对象之间的 / 表示拼接路径，这里不是数学上的除法。
    # question01 的上一级是作业根目录，原始数据文件夹就在根目录下。
    data_dir = question_dir.parent / "assignment01_images_2026"
    result_dir = question_dir / "results"
    # mkdir 创建目录；exist_ok=True 表示目录已经存在时也不报错。
    result_dir.mkdir(exist_ok=True)
    # 这里只构造文件路径，还没有读取图片内容。
    color_path = data_dir / "mandril_color.tif"
    gray_path = data_dir / "cameraman.tif"

    # ---------- 2. 用 Pillow 读取和保存 ----------
    # Image.open 返回图片对象；with ... as image 把它命名为 image。
    # 离开 with 的缩进范围时会关闭源文件，即使中途发生异常也会清理文件资源。
    with Image.open(color_path) as image:
        # format：文件编码格式（TIFF）；mode：颜色模式（RGB）；size：(宽, 高)。
        # print 可以接收多个值，默认用空格分隔后显示在终端。
        print("Pillow 图片信息：", image.format, image.mode, image.size)
        # 将 Image 对象转成独立的 NumPy 数组，后续可以按行、列访问像素。
        # 本图得到 (512, 512, 3) 的数组，每个像素含 R、G、B 三个数值。
        # np.array 默认复制数据，因此关闭源文件后仍然可以使用 pillow_rgb。
        pillow_rgb = np.array(image)
        # save 根据 .png 扩展名选择 PNG 编码；只写输出文件，不修改源 TIFF。
        image.save(result_dir / "pillow.png")

    with Image.open(gray_path) as image:
        # L 表示 8 位灰度模式；本图每个像素只有一个值，所以数组是二维的。
        print("灰度图片信息：", image.format, image.mode, image.size)
        gray = np.array(image)

    # ---------- 3. 用 scikit-image 读取和保存 ----------
    # imread 直接返回 NumPy 数组，不需要像 Pillow 那样再调用 np.array。
    # 对本题彩色图，返回通道顺序为 RGB；imsave 把数组编码并写入文件。
    skimage_rgb = io.imread(color_path)
    io.imsave(result_dir / "scikit_image.png", skimage_rgb)

    # ---------- 4. 用 OpenCV 读取和保存 ----------
    # 普通路径可用 cv2.imread，但部分 Windows 构建对中文路径支持不稳定。
    # 这里分两步读取：先获得文件字节，再让 OpenCV 解码为像素数组。
    # uint8 是 8 位无符号整数，范围 0～255，适合表示一个字节。
    # 注意：file_bytes 是一维的文件编码数据，此时还不是 (高, 宽, 通道) 的图片。
    file_bytes = np.fromfile(color_path, dtype=np.uint8)
    # imdecode 负责解码；IMREAD_COLOR 要求读取为三通道 BGR 彩色图。
    opencv_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    # None 表示没有得到图片。raise 抛出异常并停止程序，避免后面继续处理空结果。
    # f"...{变量}..." 是格式化字符串，会把花括号中的值填入提示文字。
    if opencv_bgr is None:
        raise OSError(f"OpenCV 无法读取图片：{color_path}")

    # OpenCV 使用 BGR，Matplotlib 使用 RGB：同一个像素的数值排列顺序不同。
    # 例如 RGB [166, 139, 62] 对应 BGR [62, 139, 166]，表达的颜色相同。
    # cvtColor 生成转换结果；opencv_bgr 仍保留，opencv_rgb 专门用于 Matplotlib 显示。
    opencv_rgb = cv2.cvtColor(opencv_bgr, cv2.COLOR_BGR2RGB)

    # OpenCV 编码接口需要 BGR，所以这里使用 opencv_bgr，而不是 opencv_rgb。
    # imencode 返回两个值：success 是是否成功的布尔值；encoded 是 PNG 编码字节。
    # 左边用两个变量接收两个返回值，这种写法叫“解包”。
    success, encoded = cv2.imencode(".png", opencv_bgr)
    # not success 表示“没有成功”；编码失败时不继续保存。
    if not success:
        raise OSError("OpenCV 无法将图片编码为 PNG")
    # 把编码字节写入目标路径，这种方式也能处理当前的中文目录。
    encoded.tofile(result_dir / "opencv.png")

    # ---------- 5. 观察图片在内存中的表示 ----------
    # 下面每一对 (名称, 数组) 是一个元组；for 每次取一对，分别赋给 name 和 array。
    # 用循环可以对四个数组做同样的打印操作，而不用重复写四遍 print。
    # shape 是数组各维度长度：彩色图为 (高, 宽, 通道)，灰度图为 (高, 宽)。
    # dtype 是元素类型；本题读取结果为 uint8，但其他图片也可能是 uint16、浮点等。
    # 数组索引从 0 开始，[0, 0] 取左上角像素：彩色图得到三个值，灰度图得到一个值。
    for name, array in (
        ("Pillow RGB", pillow_rgb),
        ("scikit-image RGB", skimage_rgb),
        ("OpenCV BGR", opencv_bgr),
        ("灰度图", gray),
    ):
        print(f"{name}: shape={array.shape}, dtype={array.dtype}, "
              f"左上角像素={array[0, 0]}")

    # ---------- 6. 并排显示三个库读取的彩色图 ----------
    # 图中文字使用英文，避免未配置中文字体时出现方框；代码注释使用中文。
    # subplots(1, 3) 创建一行三列的子图。
    # fig 是整张画布，axes 包含三个子图对象，每个子图都能独立显示图片、设置标题。
    # figsize=(12, 4) 的单位是英寸；constrained 会自动调整间距，减少标题重叠。
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    # zip 按位置配对：第一个子图、第一段标题、第一张图片组成一组，依次类推。
    # for 每次把这三个对象分别取名为 ax、title、array，再执行下方的显示代码。
    for ax, title, array in zip(
        axes,
        ("Pillow", "scikit-image", "OpenCV: BGR to RGB"),
        (pillow_rgb, skimage_rgb, opencv_rgb),
    ):
        ax.imshow(array)      # 将 RGB 数组显示在当前子图上。
        ax.set_title(title)   # 设置当前子图的标题。
        ax.axis("off")       # 隐藏坐标轴刻度和边框，像素数据本身不变。
    # 保存的是带标题的整张画布，不是单张源图。dpi 表示每英寸输出的像素数。
    # 与前面的 Image.save/io.imsave 不同，画布尺寸由 figsize 和 dpi 决定。
    fig.savefig(result_dir / "comparison.png", dpi=150)

    # ---------- 7. 显示灰度图 ----------
    # 不指定行列数时，subplots 默认只创建一个子图，gray_ax 就是该子图对象。
    gray_fig, gray_ax = plt.subplots(figsize=(4, 4), layout="constrained")
    # 二维数组默认可能显示成伪彩色；cmap="gray" 指定用黑白灰来表示数值。
    # vmin=0、vmax=255 固定映射：0 显示为黑，255 显示为白，中间值显示为灰。
    # 这些参数只改变显示方式，不会修改 gray 数组中的值。
    gray_ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
    gray_ax.set_title("Grayscale: cameraman")
    gray_ax.axis("off")
    gray_fig.savefig(result_dir / "grayscale.png", dpi=150)

    print(f"图片已保存到：{result_dir}")
    # 先保存再 show，确保图片在打开窗口前已经写入磁盘。
    # 普通脚本运行时，show 通常会等待你关闭窗口；此时终端停住不是程序出错。
    plt.show()
    plt.close("all")  # 关闭剩余画布并释放绘图资源，不删除已经保存的图片。


# 直接运行这个文件时，Python 会将 __name__ 设为 "__main__"，于是调用 main()。
# 如果以后其他文件导入本文件，下面的条件不成立，就不会自动执行整套实验。
if __name__ == "__main__":
    main()
