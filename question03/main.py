"""问题 3：分别用平均法和 NTSC 加权法将 RGB 图像转换为灰度图。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def rgb1gray(f: np.ndarray, method: str = "NTSC") -> np.ndarray:
    """将 24 位 RGB 图像转换为二维浮点灰度图。

    参数：
        f：形状为 (高, 宽, 3) 的 uint8 数组，通道顺序必须是 RGB。
        method："average" 为三个通道取平均；"NTSC" 为题目指定的加权和。

    返回：
        形状为 (高, 宽) 的 float64 数组，保留计算得到的小数。
        保存为普通 8 位灰度图片时，再单独取整并转换为 uint8。
    """
    # 24 位 RGB 指每个像素有 R、G、B 三个通道，每通道 8 位。
    # 先检查是不是数组，再检查维度、通道数、尺寸和数据类型。
    if not isinstance(f, np.ndarray):
        raise TypeError("f 必须是 NumPy 数组")
    if f.ndim != 3 or f.shape[2] != 3:
        raise ValueError("f 必须是 RGB 图像，形状应为 (高, 宽, 3)")
    if f.shape[0] == 0 or f.shape[1] == 0:
        raise ValueError("输入图像不能为空")
    if f.dtype != np.uint8:
        raise TypeError("本函数输入应为每通道 8 位的 RGB 图像，即 uint8 数组")
    if not isinstance(method, str) or method not in ("average", "NTSC"):
        raise ValueError('method 只能是 "average" 或 "NTSC"，注意大小写')

    # uint8 数组相加可能溢出：例如 200 + 100 无法用 0～255 表示。
    # 因此必须先转成浮点数，再做加法；计算完之后才转浮点已经来不及。
    # astype 只改变类型，不会自动除以 255；这里仍使用 0～255 的数值尺度。
    rgb = f.astype(np.float64)

    # 前两个冒号表示取全部行和列，最后的索引选择通道。
    # 每个通道都是一张 (高, 宽) 的二维数组，并非只取一个像素。
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]

    # NumPy 对数组逐元素运算，一次计算所有像素，不需要编写两层循环。
    if method == "average":
        return (red + green + blue) / 3.0

    # 使用题目原样给出的系数，不调用现成的灰度转换函数代替本题实现。
    # 三个系数之和为 0.9999；这里不重新归一化，保留题目指定公式。
    return 0.2989 * red + 0.5870 * green + 0.1140 * blue


def main():
    # 根据本脚本定位输入与输出，不依赖启动程序时终端所在的目录。
    question_dir = Path(__file__).resolve().parent
    data_dir = question_dir.parent / "assignment01_images_2026"
    result_dir = question_dir / "results"
    result_dir.mkdir(exist_ok=True)

    for filename in ("mandril_color.tif", "runner.jpg"):
        with Image.open(data_dir / filename) as image:
            # 两张作业原图均为 RGB。显式检查，避免把其他模式的数据当作 RGB 使用。
            if image.mode != "RGB":
                raise ValueError(f"{filename} 应为 RGB 模式，实际为 {image.mode}")
            f = np.array(image)

        average = rgb1gray(f, "average")
        ntsc = rgb1gray(f)  # 不传 method，使用函数定义中的默认值 "NTSC"。
        name = Path(filename).stem  # 去掉扩展名，用于组成输出文件名。

        # 保存前才量化为 uint8：rint 取最近整数（恰好半整数时取最近偶数），
        # clip 限制范围，astype 转为 8 位整数。直接 astype 会截去小数部分。
        # 不覆盖 average 和 ntsc，使后面的绘图、比较仍使用未取整的计算值。
        for method, gray in (("average", average), ("NTSC", ntsc)):
            gray_uint8 = np.clip(np.rint(gray), 0, 255).astype(np.uint8)
            Image.fromarray(gray_uint8).save(result_dir / f"{name}_{method}.png")

        # 同一画布展示原图和两种灰度结果。固定显示范围，避免自动拉伸影响比较。
        # 图中使用英文标题，避免未配置中文字体时出现方框。
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), layout="constrained")
        axes[0].imshow(f)
        axes[0].set_title("Original RGB")
        axes[1].imshow(average, cmap="gray", vmin=0, vmax=255)
        axes[1].set_title("Average")
        axes[2].imshow(ntsc, cmap="gray", vmin=0, vmax=255)
        axes[2].set_title("NTSC")
        for ax in axes:
            ax.axis("off")  # 只隐藏坐标轴，不修改像素。
        fig.suptitle(name)
        fig.savefig(result_dir / f"{name}_comparison.png", dpi=150)

        # 绝对差衡量两种方法的差异大小，不表示哪一种更正确。
        # 这里比较浮点结果，不让保存时的取整掩盖小幅差异。
        difference = np.abs(average - ntsc)
        print(f"{filename}：输入形状={f.shape}，输出形状={ntsc.shape}")
        print(f"  灰度计算类型={ntsc.dtype}")
        print(f"  两种方法的平均绝对差={difference.mean():.4f}")
        print(f"  两种方法的最大绝对差={difference.max():.4f}")

    print(f"结果已保存到：{result_dir}")
    # 两张对比图保存后一起显示。普通脚本中，关闭窗口后程序继续结束。
    plt.show()
    plt.close("all")


# 直接运行时执行实验；其他题目导入 rgb1gray 时，不会自动读取文件和弹窗。
if __name__ == "__main__":
    main()
