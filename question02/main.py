"""问题 2：提取灰度图的中心行、中心列，并绘制灰度曲线。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def scanLine4e(f: np.ndarray, I: int, loc: str) -> np.ndarray:
    """返回灰度图中指定行或列的一维像素数组。

    参数：
        f：二维 NumPy 数组，形状为 (高, 宽)。
        I：从 0 开始的行号或列号，不接受负数索引。
        loc："row" 表示提取行，"column" 表示提取列。

    返回：
        指定行或列的副本，数据类型与输入一致。
        行序列长度等于图片宽度，列序列长度等于图片高度。

    上面的类型标注用于说明接口，不会自动检查参数，因此下方仍需检查。
    """
    # ndim 是数组的维度数。灰度图应有行、列两个维度，彩色图通常还有通道维度。
    # 先检查是不是数组，再访问 ndim，避免对其他对象访问不存在的属性。
    if not isinstance(f, np.ndarray):
        raise TypeError("f 必须是 NumPy 数组")
    if f.ndim != 2:
        raise ValueError("f 必须是二维灰度图，形状应为 (高, 宽)")

    # 同时接受 Python 整数和 NumPy 整数，例如 int、np.int64。
    # Python 的 bool 是 int 的子类，但 True/False 不适合当作本函数的行列编号。
    if isinstance(I, (bool, np.bool_)) or not isinstance(I, (int, np.integer)):
        raise TypeError("I 必须是整数行号或列号")
    if not isinstance(loc, str) or loc not in ("row", "column"):
        raise ValueError('loc 只能是 "row" 或 "column"，注意大小写')

    height, width = f.shape
    if height == 0 or width == 0:
        raise ValueError("输入图像不能为空")

    if loc == "row":
        # 有 height 行时，合法行号为 0 到 height - 1。
        if not 0 <= I < height:
            raise IndexError(f"行号 {I} 越界，合法范围为 0～{height - 1}")
        # 固定行号 I，冒号表示取所有列：从左到右获得这一行的像素。
        return f[I, :].copy()

    # 前面已检查 loc，执行到这里时只能是 "column"。
    if not 0 <= I < width:
        raise IndexError(f"列号 {I} 越界，合法范围为 0～{width - 1}")
    # 固定列号 I，取所有行：从上到下获得这一列的像素。
    # NumPy 切片通常与原数组共享数据；copy() 让返回值独立，避免误改原图。
    return f[:, I].copy()


def main():
    # __file__ 是本脚本的路径，parent 是上一级目录。
    # 使用脚本位置定位数据，即使从其他工作目录启动，也能找到原图。
    question_dir = Path(__file__).resolve().parent
    data_dir = question_dir.parent / "assignment01_images_2026"
    result_dir = question_dir / "results"
    result_dir.mkdir(exist_ok=True)  # 第一次运行时创建；已有目录则继续使用。

    # 两张原图本身就是灰度图，不做颜色转换或缩放，以保留原始灰度值。
    for filename in ("cameraman.tif", "einstein.tif"):
        with Image.open(data_dir / filename) as image:
            f = np.array(image)  # 复制像素数据，关闭文件后仍能使用。

        if f.ndim != 2:
            raise ValueError(f"{filename} 应为二维灰度图，实际形状为 {f.shape}")
        height, width = f.shape  # NumPy 顺序为 (高, 宽)，不是 Pillow 的 (宽, 高)。

        # // 表示整除。奇数尺寸取唯一中心；偶数尺寸取两个居中位置中靠下/右的一个。
        # 例如 256 行的两个居中索引为 127、128，本程序取 128。
        center_row = height // 2
        center_column = width // 2
        row_values = scanLine4e(f, center_row, "row")
        column_values = scanLine4e(f, center_column, "column")

        print(f"{filename}：形状={f.shape}")
        print(f"  中心行索引={center_row}，行序列长度={row_values.size}")
        print(f"  中心列索引={center_column}，列序列长度={column_values.size}")

        # 一张画布上放三个子图：原图、中心行曲线、中心列曲线。
        # 图内使用英文，避免没有配置中文字体时出现方框。
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
        row_color = "tab:orange"
        column_color = "tab:blue"

        # origin="upper" 明确第 0 行在顶部，行号从上到下增加。
        # axhline/axvline 只在画布上叠加线条，不会修改数组 f 中的任何像素。
        axes[0].imshow(f, cmap="gray", vmin=0, vmax=255, origin="upper")
        axes[0].axhline(center_row, color=row_color, linewidth=1,
                       label=f"Row {center_row}")
        axes[0].axvline(center_column, color=column_color, linewidth=1,
                       linestyle="--", label=f"Column {center_column}")
        axes[0].set_title(Path(filename).stem)
        axes[0].set_xlabel("Column index (left to right)")
        axes[0].set_ylabel("Row index (top to bottom)")
        axes[0].legend(loc="upper right", fontsize=8)

        # 一行有 width 个像素。arange(width) 生成 0, 1, ..., width - 1，作为横坐标。
        # 曲线连接离散像素的灰度值，连接线只是帮助观察变化，并非新增像素。
        axes[1].plot(np.arange(width), row_values, color=row_color, linewidth=1)
        axes[1].set_title(f"Center row: {center_row}")
        axes[1].set_xlabel("Column index (left to right)")
        axes[1].set_xlim(0, width - 1)

        # 一列有 height 个像素；这里横轴是行号，表示沿原图从上向下扫描的位置。
        axes[2].plot(np.arange(height), column_values, color=column_color,
                     linewidth=1, linestyle="--")
        axes[2].set_title(f"Center column: {center_column}")
        axes[2].set_xlabel("Row index (top to bottom)")
        axes[2].set_xlim(0, height - 1)

        # 本题两张图片都是 uint8 灰度图，统一纵轴为 0～255，便于对比亮暗程度。
        # 这些显示设置不改变扫描函数的返回值，也没有对数据归一化或平滑。
        for ax in axes[1:]:
            ax.set_ylabel("Gray value")
            ax.set_ylim(0, 255)
            ax.grid(True, alpha=0.3)  # alpha 控制透明度，让网格不遮挡曲线。

        # stem 去掉文件扩展名；savefig 保存整张画布，包括原图、扫描线和曲线。
        output_path = result_dir / f"{Path(filename).stem}_scanlines.png"
        fig.savefig(output_path, dpi=150)
        print(f"  已保存：{output_path}")

    # 先保存两张结果图，再一起显示。普通脚本中，关闭所有窗口后程序才继续结束。
    plt.show()
    plt.close("all")  # 释放绘图资源，不会删除已保存的文件。


# 直接运行本文件时执行实验；被其他文件导入时，不会自动读取图片或弹窗。
if __name__ == "__main__":
    main()
