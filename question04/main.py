"""问题 4：实现同尺寸二维卷积，支持补零和边界像素复制。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def twodConv(f: np.ndarray, w: np.ndarray, padding: str = "zero") -> np.ndarray:
    """对二维灰度图进行卷积，返回与输入图像同尺寸的 float64 数组。

    f：非空的二维实数数组，表示灰度图。
    w：非空的二维实数数组，表示矩形卷积核，不要求为正方形。
    padding："zero" 为补零（默认），"replicate" 为边界像素复制。

    对 m×n 的核，翻转后核的锚点取 (m//2, n//2)。
    行、列分别确定锚点：奇数长度取中心，偶数长度取两个居中索引中较大的一个。
    函数不取整、不裁剪，也不自动归一化卷积核。
    """
    # 先验证输入，使错误尽量发生在具体的参数检查处，而不是后续数组运算中。
    for name, array in (("f", f), ("w", w)):
        if not isinstance(array, np.ndarray):
            raise TypeError(f"{name} 必须是 NumPy 数组")
        if array.ndim != 2 or array.size == 0:
            raise ValueError(f"{name} 必须是非空二维数组")
        # kind 为 u/i/f 分别表示无符号整数、有符号整数、浮点数。
        # 本题处理实数像素，拒绝字符串、复数和布尔数组。
        if array.dtype.kind not in "uif":
            raise TypeError(f"{name} 必须是整数或浮点数组")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} 不能包含 NaN 或无穷大")
    if not isinstance(padding, str) or padding not in ("zero", "replicate"):
        raise ValueError('padding 只能是 "zero" 或 "replicate"')

    # 转浮点后再乘加，避免 uint8 溢出，并保留负数和小数结果。
    image = f.astype(np.float64)
    kernel = w.astype(np.float64)
    height, width = image.shape
    kernel_height, kernel_width = kernel.shape

    # 输出与输入同尺寸时，一共需要补 m-1 行、n-1 列。
    # 分别看两个方向：核高为奇数时上下对称，为偶数时上方比下方多补一个。
    # 核宽同理：奇数时左右对称，偶数时左方比右方多补一个。
    top = kernel_height // 2
    bottom = kernel_height - 1 - top
    left = kernel_width // 2
    right = kernel_width - 1 - left
    pad_width = ((top, bottom), (left, right))

    # np.pad 只负责构造边界扩展，不进行卷积。
    # constant 填指定常数；edge 用最近的边界值扩展，对应题目的 replicate。
    if padding == "zero":
        padded = np.pad(image, pad_width, mode="constant", constant_values=0)
    else:
        padded = np.pad(image, pad_width, mode="edge")

    # ::-1 表示逆序取值。两维都逆序等价于将核旋转 180°，是卷积的重要步骤。
    # 不翻转核而直接滑动乘加，计算的是相关运算。
    flipped = kernel[::-1, ::-1]
    g = np.zeros((height, width), dtype=np.float64)

    # 数学上：每个输出像素等于一个局部窗口与 flipped 逐元素相乘后求和。
    # 这里按“核的位置”循环，每次同时更新整张输出，避免 Python 逐像素循环过慢。
    # padded[i:i+height, j:j+width] 是该核位置对应的全部图像取样点，
    # 形状始终为 (height, width)，可以与 g 逐元素相加。
    for i in range(kernel_height):
        for j in range(kernel_width):
            g += flipped[i, j] * padded[i:i + height, j:j + width]

    return g


def main():
    # 先用可手算的小矩阵观察卷积结果。非对称核能暴露“忘记翻转”的错误。
    sample = np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]], dtype=np.float64)
    asymmetric_kernel = np.array([[1, 2, 0],
                                  [0, 0, 0],
                                  [0, 0, 0]], dtype=np.float64)
    print("小矩阵输入：\n", sample)
    print("非对称卷积核：\n", asymmetric_kernel)
    for padding in ("zero", "replicate"):
        result = twodConv(sample, asymmetric_kernel, padding)
        print(f"{padding} 卷积结果：\n{result}")
        # 中心像素无需使用填充值。翻转后的核底行为 [0, 2, 1]，故结果为 2×8+9=25。
        print(f"  中心值={result[1, 1]}，按手算应为 25")

    question_dir = Path(__file__).resolve().parent
    data_dir = question_dir.parent / "assignment01_images_2026"
    result_dir = question_dir / "results"
    result_dir.mkdir(exist_ok=True)
    with Image.open(data_dir / "cameraman.tif") as image:
        f = np.array(image)

    # 用 9×15 的矩形均值核展示平滑和边界效果，同时演示非正方形核的使用。
    # 核的 135 个系数相等且总和为 1；归一化是这个示例核的选择，不是卷积的要求。
    mean_kernel = np.ones((9, 15), dtype=np.float64) / (9 * 15)
    zero = twodConv(f, mean_kernel)  # 省略第三个参数，默认补零。
    replicate = twodConv(f, mean_kernel, "replicate")
    print(f"真实图像：输入={f.shape}，核={mean_kernel.shape}")
    print(f"输出：zero={zero.shape}，replicate={replicate.shape}")

    # 均值核的结果在 0～255 内，可以保存为普通 8 位灰度 PNG。
    # 一般卷积可能输出负数或超过 255，不能把这种保存方式当作卷积函数本身的步骤。
    for name, gray in (("zero", zero), ("replicate", replicate)):
        saved = np.clip(np.rint(gray), 0, 255).astype(np.uint8)
        Image.fromarray(saved).save(result_dir / f"cameraman_{name}.png")

    # 两行三列：第一行看全图，第二行看左上角同一块区域，便于观察边缘变暗。
    # 裁取结果中的前 40 行、前 40 列，不是把裁剪块单独进行卷积。
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), layout="constrained")
    for column, (title, gray) in enumerate((
        ("Original", f), ("Zero padding", zero), ("Replicate padding", replicate)
    )):
        axes[0, column].imshow(gray, cmap="gray", vmin=0, vmax=255)
        axes[0, column].set_title(title)
        axes[1, column].imshow(gray[:40, :40], cmap="gray", vmin=0, vmax=255,
                              interpolation="nearest")
        axes[1, column].set_title(f"{title}: top-left 40 x 40")
        axes[0, column].axis("off")
        axes[1, column].axis("off")
    fig.suptitle("2D convolution with a 9 x 15 mean kernel")
    fig.savefig(result_dir / "padding_comparison.png", dpi=150)

    print(f"图片已保存到：{result_dir}")
    plt.show()  # 普通脚本中，关闭窗口后程序继续结束。
    plt.close("all")


# 后续题目可以导入 twodConv，导入时不会自动执行这里的演示。
if __name__ == "__main__":
    main()
