"""问题 5：构造归一化二维高斯核，观察 sigma 和核尺寸的作用。"""

import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def gaussKernel(sig: float, m: int | None = None) -> np.ndarray:
    """返回形状为 (m, m)、元素和约为 1 的 float64 高斯核。

    sig：正的有限实数，表示标准差 sigma，单位为像素。
    m：正整数核尺寸；省略时使用 2 * ceil(3 * sig) + 1。
       显式尺寸小于建议尺寸时发出警告，但仍按用户给定的尺寸计算。
       奇数尺寸以中心像素为原点，偶数尺寸以中间四个像素之间的位置为原点。
    """
    # bool 在 Python 中属于整数的子类，但 True/False 不适合作为标准差。
    if isinstance(sig, (bool, np.bool_)) or not isinstance(
        sig, (int, float, np.integer, np.floating)
    ):
        raise TypeError("sig 必须是正的实数")
    sig = float(sig)
    if not math.isfinite(sig) or sig <= 0:
        raise ValueError("sig 必须大于 0，且不能是 NaN 或无穷大")

    # 默认每个方向覆盖约正负 3 sigma。ceil 向上取整，保证半径不小于 3 sigma。
    # 先 ceil(sig)，不是这里采用的规则；应对完整的 3*sig 向上取整。
    recommended_m = 2 * math.ceil(3 * sig) + 1
    if m is None:
        m = recommended_m
    else:
        if isinstance(m, (bool, np.bool_)) or not isinstance(m, (int, np.integer)):
            raise TypeError("m 必须是正整数")
        m = int(m)
        if m <= 0:
            raise ValueError("m 必须大于 0")
        if m < recommended_m:
            # 警告不同于异常：提醒截断范围偏小，然后继续返回指定尺寸的核。
            # stacklevel=2 让警告定位到调用函数的位置，便于知道是哪次调用触发的。
            warnings.warn(
                f"m={m} 小于 sigma={sig:g} 的建议尺寸 {recommended_m}，"
                "高斯尾部截断较多；仍按指定尺寸生成并归一化。",
                UserWarning,
                stacklevel=2,
            )

    # arange(m) 生成 0 到 m-1；减去 (m-1)/2 后，以核的几何中心为原点。
    # m=3 时为 [-1, 0, 1]；m=4 时为 [-1.5, -0.5, 0.5, 1.5]。
    coordinates = np.arange(m, dtype=np.float64) - (m - 1) / 2
    # meshgrid 将一维坐标展开成二维网格。
    # x 每一行都是横坐标序列，y 每一列都是纵坐标序列。
    x, y = np.meshgrid(coordinates, coordinates)
    distance_squared = x ** 2 + y ** 2

    # 理论上的未归一化权重为 exp(-(x^2+y^2)/(2*sig^2))。
    # 减去最小平方距离，相当于给所有权重乘同一个常数，归一化后结果不变。
    # 奇数 m 时最小距离是 0；偶数 m 且 sigma 很小时，这一步能防止所有权重
    # 一起下溢为 0。最近的点指数始终为 0，因此至少一个权重等于 1。
    offset_distance = distance_squared - distance_squared.min()
    # 连续除以 sig，避免极小 sig 的平方先变为 0。
    # 非中心位置在极小 sig 下可能得到无穷大的衰减量，此时 exp(-inf)=0，符合极限。
    with np.errstate(over="ignore", under="ignore"):
        weights = np.exp(-0.5 * (offset_distance / sig) / sig)

    # 按离散元素和归一化，而不是只套用连续高斯公式的 1/(2*pi*sig^2)。
    # 浮点运算可能有极小的舍入误差，因此元素和应理解为在数值精度内等于 1。
    w = weights / weights.sum()
    return w


def main():
    question_dir = Path(__file__).resolve().parent
    result_dir = question_dir / "results"
    result_dir.mkdir(exist_ok=True)

    # 小例子便于对照 README 手算。sigma=1 建议 m=7，这里故意指定 m=3，
    # 所以出现“核尺寸偏小”的警告是预期行为，不代表程序失败。
    small_kernel = gaussKernel(1, 3)
    print("sigma=1、m=3 的高斯核（显示六位小数）：")
    print(np.array2string(small_kernel, precision=6, suppress_small=True))
    print(f"元素和：{small_kernel.sum():.15f}")
    np.savetxt(result_dir / "gaussian_sigma1_m3.txt", small_kernel, fmt="%.12f")

    # 使用后续第 6 题所需的四个 sigma；m 省略，由函数自动确定。
    sigmas = (1, 2, 3, 5)
    kernels = [gaussKernel(sig) for sig in sigmas]
    # 四幅热力图共用同一色标，颜色相同才代表权重相同。
    # 大 sigma 的核中心权重较小，因此不能给每幅图单独自动拉伸色标再比较亮度。
    vmax = max(kernel.max() for kernel in kernels)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), layout="constrained")

    for column, (sig, kernel) in enumerate(zip(sigmas, kernels)):
        m = kernel.shape[0]
        radius = m // 2  # 这里都是自动生成的奇数尺寸核。
        coordinate = np.arange(m) - radius
        print(f"sigma={sig}：尺寸={kernel.shape}，"
              f"元素和={kernel.sum():.15f}，中心权重={kernel[radius, radius]:.6f}")
        # 保存数值矩阵供查看和复用；文件只包含程序输出，不生成额外汇总文档。
        np.savetxt(result_dir / f"gaussian_sigma{sig}_m{m}.txt", kernel, fmt="%.12f")

        # extent 指定像素外边界，使整数坐标对应权重单元的中心。
        heatmap = axes[0, column].imshow(
            kernel, cmap="viridis", vmin=0, vmax=vmax, interpolation="nearest",
            origin="lower", extent=(-radius - 0.5, radius + 0.5,
                                     -radius - 0.5, radius + 0.5),
        )
        axes[0, column].set_title(f"sigma={sig}, size={m} x {m}")
        axes[0, column].set_xlabel("x (pixels)")
        axes[0, column].set_ylabel("y (pixels)")

        # 中心行是二维核的一条截面，不是单独归一化的一维高斯核。
        axes[1, column].plot(coordinate, kernel[radius, :], marker="o", markersize=3)
        axes[1, column].set_title("Center row of the 2D kernel")
        axes[1, column].set_xlabel("x (pixels)")
        axes[1, column].set_ylabel("Weight")
        axes[1, column].set_xlim(-15, 15)  # 所有截面使用相同坐标范围便于比较宽度。
        axes[1, column].set_ylim(0, vmax * 1.05)
        axes[1, column].grid(True, alpha=0.3)

    fig.colorbar(heatmap, ax=axes[0, :].tolist(), label="Weight", shrink=0.85)
    fig.suptitle("Normalized 2D Gaussian kernels")
    fig.savefig(result_dir / "gaussian_kernels.png", dpi=150)
    print(f"结果已保存到：{result_dir}")
    plt.show()
    plt.close("all")


# 导入 gaussKernel 时不自动运行演示，方便第 6 题调用。
if __name__ == "__main__":
    main()
