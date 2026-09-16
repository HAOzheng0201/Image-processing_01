"""问题 7：将灰度图或 RGB 图分成不重叠的图像块，并通过拼接检查结果。"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image


def split_image(f: np.ndarray, block_size: int, padding: str = "zero") -> np.ndarray:
    """按从上到下、从左到右的顺序分块，不足一块时在底部和右侧填充。

    f：非空灰度数组 (H, W)，或 RGB 数组 (H, W, 3)。
    block_size：正整数块边长 B。本题演示 8、16、32，函数也允许其他正整数。
    padding："zero" 补零，"replicate" 复制最近边界像素，默认补零。

    返回灰度块 (N, B, B)，或彩色块 (N, B, B, 3)。
    N = ceil(H/B) * ceil(W/B)。返回数组独立于原图，保留原图的数据类型。
    """
    if not isinstance(f, np.ndarray):
        raise TypeError("f 必须是 NumPy 数组")
    if f.ndim not in (2, 3):
        raise ValueError("f 应为二维灰度图或三维 RGB 图")
    if f.ndim == 3 and f.shape[2] != 3:
        raise ValueError("彩色输入必须有三个 RGB 通道")
    if f.shape[0] == 0 or f.shape[1] == 0:
        raise ValueError("输入图像不能为空")
    if f.dtype.kind not in "uif" or not np.isfinite(f).all():
        raise ValueError("像素必须是有限的整数或浮点数")
    if isinstance(block_size, (bool, np.bool_)) or not isinstance(
        block_size, (int, np.integer)
    ):
        raise TypeError("block_size 必须是正整数")
    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError("block_size 必须大于 0")
    if not isinstance(padding, str) or padding not in ("zero", "replicate"):
        raise ValueError('padding 只能是 "zero" 或 "replicate"')

    height, width = f.shape[:2]
    # 向上取整得到块网格的行列数，不必先转成浮点数再调用 ceil。
    rows = (height + block_size - 1) // block_size
    columns = (width + block_size - 1) // block_size
    padded_height = rows * block_size
    padded_width = columns * block_size

    # 只在底部和右侧补齐，左上角仍然是原图 [0, 0]。
    # 第 4 题也使用 zero/replicate，但为了卷积在四周填充；这里的填充位置不同。
    pad_width = ((0, padded_height - height), (0, padded_width - width))
    if f.ndim == 3:
        # 第三个维度是颜色通道，不能往通道维度补像素。
        pad_width += ((0, 0),)
    if padding == "zero":
        padded = np.pad(f, pad_width, mode="constant", constant_values=0)
    else:
        padded = np.pad(f, pad_width, mode="edge")

    # f.shape[2:]：灰度图为 ()，RGB 图为 (3,)，从而用同一写法构造输出形状。
    patches = np.empty(
        (rows * columns, block_size, block_size) + f.shape[2:], dtype=f.dtype
    )
    # 外层逐行、内层逐列，顺序相当于阅读文字：先排完一行，再进入下一行。
    for row in range(rows):
        for column in range(columns):
            index = row * columns + column
            top = row * block_size
            left = column * block_size
            # 切片不包括右端点，所以每次正好取 block_size 行、block_size 列。
            # 若有 RGB 通道，没有写出的第三维会全部保留。
            patches[index] = padded[top:top + block_size, left:left + block_size]
    return patches


def reconstruct_image(patches: np.ndarray, original_shape: tuple) -> np.ndarray:
    """按相同顺序拼回图像，再裁去底部和右侧填充；用于检查分块是否丢失数据。"""
    if not isinstance(patches, np.ndarray) or patches.ndim not in (3, 4):
        raise ValueError("patches 应为 (N,B,B) 或 (N,B,B,3) 数组")
    if patches.shape[0] == 0 or patches.shape[1] == 0:
        raise ValueError("图像块不能为空")
    if patches.shape[1] != patches.shape[2]:
        raise ValueError("图像块必须是正方形")
    if patches.ndim == 4 and patches.shape[3] != 3:
        raise ValueError("彩色图像块应有三个通道")
    original_shape = tuple(original_shape)
    if len(original_shape) not in (2, 3) or any(
        isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v <= 0
        for v in original_shape
    ):
        raise ValueError("original_shape 必须是由正整数构成的原图形状")
    if original_shape[2:] != patches.shape[3:]:
        raise ValueError("原图的通道形状与图像块不一致")

    height, width = original_shape[:2]
    block_size = patches.shape[1]
    rows = (height + block_size - 1) // block_size
    columns = (width + block_size - 1) // block_size
    if patches.shape[0] != rows * columns:
        raise ValueError("图像块数量与原图形状不匹配")

    canvas = np.empty(
        (rows * block_size, columns * block_size) + original_shape[2:],
        dtype=patches.dtype,
    )
    for row in range(rows):
        for column in range(columns):
            index = row * columns + column
            top, left = row * block_size, column * block_size
            canvas[top:top + block_size, left:left + block_size] = patches[index]
    # 通过原图形状区分“原来就是黑色”的像素与“后来补零”的像素。
    return canvas[:height, :width].copy()


def show_image(ax, image, title):
    """灰度图固定显示范围，RGB 图按原通道顺序显示。"""
    if image.ndim == 2:
        ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    else:
        ax.imshow(image, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")


def save_overview(f, padded, patches, block_size, path):
    """展示分块位置、中央相邻六块和最后一块；不是只在原图上画网格。"""
    height, width = f.shape[:2]
    rows = padded.shape[0] // block_size
    columns = padded.shape[1] // block_size
    # 从网格中央附近选择两行三列，保持它们在原图中的相邻关系。
    # 四张作业图片即使 B=32，网格也至少为 8×8，因此这些位置都合法且不含填充。
    start_row = rows // 2 - 1
    start_column = columns // 2 - 1

    # GridSpec 允许左边的总览占多格，右边六个小图分开显示实际块数据。
    fig = plt.figure(figsize=(14, 9), layout="constrained")
    layout = fig.add_gridspec(3, 5, width_ratios=(1.2, 1.2, 1, 1, 1))
    overview = fig.add_subplot(layout[:, :2])
    show_image(overview, padded, f"Full grid: {rows} x {columns}")
    for y in range(block_size, padded.shape[0], block_size):
        overview.axhline(y - 0.5, color="yellow", linewidth=0.35, alpha=0.65)
    for x in range(block_size, padded.shape[1], block_size):
        overview.axvline(x - 0.5, color="yellow", linewidth=0.35, alpha=0.65)
    if padded.shape[0] > height:
        overview.axhline(height - 0.5, color="red", linestyle="--", linewidth=1)
    if padded.shape[1] > width:
        overview.axvline(width - 0.5, color="red", linestyle="--", linewidth=1)

    # 青色框圈出右上方展示的六块。坐标减 0.5，是因为像素中心位于整数坐标。
    overview.add_patch(Rectangle(
        (start_column * block_size - 0.5, start_row * block_size - 0.5),
        3 * block_size, 2 * block_size, fill=False, edgecolor="cyan", linewidth=2,
    ))
    for local_row in range(2):
        for local_column in range(3):
            row = start_row + local_row
            column = start_column + local_column
            index = row * columns + column
            top, left = row * block_size, column * block_size
            # 直接与原图中的对应切片比较，避免只靠拆分、重建的相互抵消判断正确性。
            expected = f[top:top + block_size, left:left + block_size]
            if not np.array_equal(patches[index], expected):
                raise RuntimeError(f"第 {index} 块与原图对应区域不一致")
            ax = fig.add_subplot(layout[local_row, local_column + 2])
            show_image(ax, patches[index],
                       f"Patch {index}: grid ({row}, {column})\n"
                       f"rows [{top}:{top + block_size}), cols [{left}:{left + block_size})")
            ax.title.set_fontsize(9)

    # 最后一块单独放在底部，不与中央六块混排，以免误以为它们空间上相邻。
    last_top = (rows - 1) * block_size
    last_left = (columns - 1) * block_size
    valid_height = height - last_top
    valid_width = width - last_left
    overview.add_patch(Rectangle(
        (last_left - 0.5, last_top - 0.5), block_size, block_size,
        fill=False, edgecolor="orange", linewidth=2,
    ))
    last_ax = fig.add_subplot(layout[2, 2])
    show_image(last_ax, patches[-1], f"Last patch: {len(patches) - 1}")
    information = fig.add_subplot(layout[2, 3:])
    information.axis("off")
    information.text(
        0, 0.95,
        f"Each patch: {block_size} x {block_size} pixels\n"
        "Cyan box: six neighboring patches above\n"
        "Orange box: last patch at bottom right\n"
        "Red dashed line: input / padding boundary\n\n"
        f"Last patch starts at ({last_top}, {last_left})\n"
        f"Original pixels in last patch: {valid_height} x {valid_width}\n"
        f"Padded pixels: {block_size ** 2 - valid_height * valid_width}\n"
        "All indices start at 0; slice ends are excluded.",
        va="top", fontsize=10, linespacing=1.6,
    )
    fig.suptitle(f"{path.stem.removesuffix('_overview')}: actual patches, zero padding")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    question_dir = Path(__file__).resolve().parent
    data_dir = question_dir.parent / "assignment01_images_2026"
    output = question_dir / "results"
    output.mkdir(exist_ok=True)
    filenames = ("cameraman.tif", "einstein.tif", "mandril_color.tif", "runner.jpg")

    for filename in filenames:
        with Image.open(data_dir / filename) as image:
            if image.mode not in ("L", "RGB"):
                raise ValueError(f"{filename} 应为 L 或 RGB 模式")
            f = np.array(image)  # 第 7 题保留彩色原图，不复用第 6 题的灰度结果。
        name = Path(filename).stem
        height, width = f.shape[:2]

        # 原题列举三种块尺寸，但没有明确要求三种都展示；这里全部运行便于比较。
        # 主任务是得到 patches。重建、保存和展示只是帮助核对、阅读结果。
        for block_size in (8, 16, 32):
            patches = split_image(f, block_size, "zero")
            restored = reconstruct_image(patches, f.shape)
            # 本题只移动和复制像素，没有浮点计算，应当逐元素完全一致。
            if not np.array_equal(restored, f):
                raise RuntimeError(f"{filename} 的 {block_size}×{block_size} 分块重建不一致")

            rows = (height + block_size - 1) // block_size
            columns = (width + block_size - 1) // block_size
            full_shape = (rows * block_size, columns * block_size) + f.shape[2:]
            # 将原图形状换成填充后形状，即可拼出包含填充像素的完整画布。
            padded = reconstruct_image(patches, full_shape)
            print(f"{filename}，B={block_size}：网格={rows}×{columns}，"
                  f"块数组={patches.shape}，下补={full_shape[0] - height}，"
                  f"右补={full_shape[1] - width}，重建完全一致")

            # 不把几千个块分别写成文件；一个 npz 保存完整块数组及必要的布局信息。
            # 压缩是无损的，不会改变像素值；这里保存的都是普通数值/字符串数组。
            np.savez_compressed(
                output / f"{name}_b{block_size}_patches.npz",
                patches=patches, original_shape=np.array(f.shape),
                grid_shape=np.array([rows, columns]), block_size=np.array(block_size),
                padding=np.array("zero"),
            )

            save_overview(f, padded, patches, block_size,
                          output / f"{name}_b{block_size}_overview.png")

            # 以下是辅助演示，不是老师另行要求的比较实验。
            # 对尺寸不能整除的两张图，再用 B=32 展示复制与补零的区别。
            if block_size == 32 and name in ("einstein", "runner"):
                replicated = split_image(f, block_size, "replicate")
                if not np.array_equal(reconstruct_image(replicated, f.shape), f):
                    raise RuntimeError(f"{filename} 的 replicate 分块重建不一致")
                fig, axes = plt.subplots(1, 2, figsize=(8, 4), layout="constrained")
                show_image(axes[0], patches[-1], "Last patch: zero")
                show_image(axes[1], replicated[-1], "Last patch: replicate")
                fig.suptitle(f"{name}: bottom-right patch, B=32")
                fig.savefig(output / f"{name}_b32_padding_comparison.png", dpi=150)
                plt.close(fig)
                print(f"  {filename}：replicate、B=32 的重建也完全一致")

    print(f"结果已保存到：{output}")


if __name__ == "__main__":
    main()
