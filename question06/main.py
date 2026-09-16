"""问题 6：高斯滤波、与 SciPy 比较，以及两种边界填充的比较。"""

# sys 提供 Python 运行时信息，这里只用它调整模块搜索路径。
import sys
# Path 用来拼接文件路径；/ 用在 Path 对象之间时表示连接目录，而不是除法。
from pathlib import Path

# plt 负责绘图；np 负责数组运算；Image 负责读写图片。
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
# 这是库提供的参考实现，只用于比较，不代替我们自己的主实验算法。
from scipy.ndimage import gaussian_filter

# 直接执行 python question06/main.py 时，Python 首先搜索 question06 目录。
# 将作业根目录加入搜索路径，才能导入相邻题目中的函数，不必复制已有算法。
ROOT = Path(__file__).resolve().parent.parent
# __file__ 是当前脚本；resolve() 取得绝对路径；两次 parent 到达作业根目录。
# sys.path 是 Python 查找模块时搜索的目录列表，需要存放字符串形式的路径。
if str(ROOT) not in sys.path:
    # 在列表索引 0 处插入，优先从作业根目录寻找 question03 等目录。
    # 只影响本次 Python 进程，不会修改系统环境变量，也不会移动文件。
    sys.path.insert(0, str(ROOT))

# question03.main 表示 question03 目录下的 main.py；后面是从文件中导入的函数名。
# 前几题用 if __name__ == "__main__" 保护实验入口，因此导入不会自动运行那些实验。
from question03.main import rgb1gray
from question04.main import twodConv
from question05.main import gaussKernel


def show_gray(ax, image, title):
    """统一灰度显示范围，避免各子图自动拉伸亮度影响比较。"""
    # ax 是一个子图对象；image 是二维数组；title 是显示在该子图上方的文字。
    # cmap="gray" 用黑白灰表示数值，0 对应黑、255 对应白。
    # interpolation="nearest" 不在显示时额外平滑像素，便于观察算法本身的效果。
    # 这些参数只控制画法，不修改 image 中的数值。
    ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")


def save_gray(path, image):
    """仅保存时量化，原来的浮点数组仍供后续比较使用。"""
    # 从内向外执行：rint 取最近整数（半整数取最近偶数），clip 限制到 0～255，
    # astype 转为 8 位整数。比如 100.2 保存成 100，100.8 保存成 101。
    # 结果放入新变量 saved，不会覆盖传入的浮点 image。
    saved = np.clip(np.rint(image), 0, 255).astype(np.uint8)
    # 将二维 uint8 数组变成 Pillow 灰度图片，再按路径扩展名保存为 PNG。
    Image.fromarray(saved).save(path)


def main():
    # results 与本脚本处于同一个 question06 目录下。
    output = Path(__file__).resolve().parent / "results"
    output.mkdir(exist_ok=True)  # 目录不存在时创建；已经存在时不报错。
    sigmas = (1, 2, 3, 5)
    # 每个 sigma 的核只生成一次，四张图片复用；字典以 sigma 为键保存核。
    # 这是字典推导式，等价于先 kernels={}，再循环执行 kernels[sig]=gaussKernel(sig)。
    # kernels[1] 是 sigma=1 的 7×7 核，kernels[5] 是 sigma=5 的 31×31 核。
    # 方括号中的数字是字典键，不是“第几个元素”的数组下标。
    kernels = {sig: gaussKernel(sig) for sig in sigmas}
    filenames = ("cameraman.tif", "einstein.tif", "mandril_color.tif", "runner.jpg")

    for filename in filenames:
        name = Path(filename).stem  # 例如 runner.jpg 取出 runner，供输出文件命名。
        # with 在离开代码块时关闭源文件；np.array 复制出的像素仍可继续使用。
        with Image.open(ROOT / "assignment01_images_2026" / filename) as image:
            array = np.array(image)
            if image.mode == "L":
                # L 为 8 位灰度，形状是 (H,W)。转换类型后仍保留 0～255 的尺度。
                f = array.astype(np.float64)
            elif image.mode == "RGB":
                # 使用第 3 题自己的 NTSC 公式；不读取以前保存并取整过的灰度 PNG。
                # 输入是 (H,W,3)，输出变为 (H,W)，类型为 float64，保留灰度小数。
                f = rgb1gray(array, "NTSC")
            else:
                raise ValueError(f"{filename} 的模式应为 L 或 RGB，实际为 {image.mode}")

        # 第一部分：四个 sigma 都从同一幅 f 开始，不能在前一次模糊结果上继续滤波。
        # 主实验统一用 replicate；这不会修改 twodConv 默认补零的接口。
        filtered = {}  # 为当前图片创建空字典；下一张图片会重新创建，不混用结果。
        for sig in sigmas:
            # f"...{变量}..." 会将变量值嵌入文字；flush=True 让进度及时显示。
            print(f"正在处理 {filename}：sigma={sig}，核={kernels[sig].shape}", flush=True)
            # 字典的值是整张二维滤波结果。f 不变，所以四次都是对原始灰度图滤波。
            filtered[sig] = twodConv(f, kernels[sig], "replicate")
            save_gray(output / f"{name}_sigma{sig}_replicate.png", filtered[sig])

        # fig 是整张画布，axes 是五个子图组成的一维数组；figsize 的单位为英寸。
        # constrained 自动调整子图间距；一个输入加四个滤波结果，一共五列。
        fig, axes = plt.subplots(1, 5, figsize=(17, 4.5), layout="constrained")
        show_gray(axes[0], f, "Input grayscale")
        # axes[1:] 取后四个子图；zip 把它们与 sigma=1、2、3、5 按顺序配对。
        for ax, sig in zip(axes[1:], sigmas):
            show_gray(ax, filtered[sig], f"sigma={sig}")
        fig.suptitle(f"{name}: Gaussian filtering, replicate padding")  # 整张画布的总标题。
        # savefig 保存带标题的画布，dpi 为每英寸像素数，与独立灰度 PNG 的尺寸不同。
        fig.savefig(output / f"{name}_sigma_comparison.png", dpi=150)
        plt.close(fig)  # 本题图较多，保存后关闭画布；直接在 results 中查看。

        # 第二部分：sigma=1 时与库函数比较。半径 3 对应 7×7 核。
        # SciPy 的 nearest 对应我们的 replicate，不要使用默认的 reflect。
        # radius 显式指定核范围；output=float64 防止中间运算以整数存储。
        # shape[0] 取得核的行数，7//2=3；默认生成的核是正方形奇数尺寸。
        radius = kernels[1].shape[0] // 2
        # sigma=1 作用于二维图像的两个方向；order=0 表示平滑，不求高斯导数。
        # radius=3 指定每个方向取 -3～3 的 7 个位置，匹配自己生成的 7×7 核。
        # 这里的 output 是 SciPy 的命名参数，指定返回类型；不是上面的输出目录变量。
        reference = gaussian_filter(f, sigma=1, order=0, radius=radius,
                                    mode="nearest", output=np.float64)
        custom = filtered[1]  # 直接引用已经算好的 sigma=1 结果，不再卷积一次。
        # 两个数组都是 (H,W)，减法按相同像素位置进行；abs 取绝对值，避免正负抵消。
        # difference 仍为 (H,W)，每个位置记录该像素的绝对误差。
        difference = np.abs(custom - reference)
        mae = difference.mean()     # 所有像素误差的平均值，是一个数。
        max_error = difference.max()  # 所有像素中最大的误差，也是一个数。
        # rtol=0 表示只看绝对差；1e-10 是本实验的数值一致性判断阈值，非题目要求。
        # allclose 一般检查 |a-b| <= atol + rtol*|b|；本处简化为 |a-b| <= 1e-10。
        # 所有位置都满足才返回 True，不能只凭平均误差小就认定每个位置都一致。
        matches = np.allclose(custom, reference, rtol=0, atol=1e-10)
        # :.6e 表示科学计数法并保留小数点后六位，例如 1.000000e-13 表示 10 的 -13 次方。
        print(f"  与 SciPy 比较：MAE={mae:.6e}，最大绝对差={max_error:.6e}，"
              f"绝对误差阈值 1e-10 内一致={matches}")

        fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), layout="constrained")
        show_gray(axes[0], custom, "Our convolution, sigma=1")
        show_gray(axes[1], reference, "SciPy, sigma=1")
        # 差值不转 uint8，否则微小误差会全变成 0。用热力图显示真实浮点差值。
        # 色标上限按本图最大差设置；若恰好全为零，用 1e-15 避免零宽色标。
        # max(a,b) 选较大的数，这里只调整显示范围，没有修改 difference 数组。
        # magma 是颜色映射；亮色只表示在本图范围内较大，要看色标才能知道实际误差。
        # heatmap 是 imshow 返回的绘图对象，用于告诉 colorbar 使用哪套颜色和值的映射。
        heatmap = axes[2].imshow(difference, cmap="magma", vmin=0,
                                 vmax=max(float(max_error), 1e-15),
                                 interpolation="nearest")
        axes[2].set_title(f"Absolute difference\nmax={max_error:.2e}")  # \n 在标题中换行。
        axes[2].axis("off")
        # colorbar 添加颜色与数值的对应标尺；shrink=0.8 将标尺长度缩为默认的 80%。
        fig.colorbar(heatmap, ax=axes[2], label="Gray-value difference", shrink=0.8)
        fig.suptitle(f"{name}: MAE={mae:.2e}; atol=1e-10: {matches}")
        fig.savefig(output / f"{name}_library_comparison.png", dpi=150)
        plt.close(fig)

        # 第三部分：按题意任选两张图，固定 sigma=5、31×31 核，只改变填充方式。
        # in 判断图片名是否属于这个元组；另外两张图会跳过此分支。
        if name in ("cameraman", "runner"):
            replicate = filtered[5]  # 复用主实验结果，不重复计算。
            zero = twodConv(f, kernels[5], "zero")
            save_gray(output / f"{name}_sigma5_zero.png", zero)
            boundary_difference = np.abs(replicate - zero)
            radius = kernels[5].shape[0] // 2  # 31×31 核对应半径 15。

            # True 标记可能使用图外填充值的区域；False 为完全不接触图外的内部。
            # 四张作业图片均大于 31×31，因此这个内部区域非空。
            # mask（掩码）可以理解为一张与图片同尺寸的“选择表”，只存 True/False。
            # np.ones(..., dtype=bool) 先把每个位置都设为 True，即暂时选中整张图。
            boundary_mask = np.ones(f.shape, dtype=bool)
            # radius=15；[15:-15] 去掉前后各 15 个位置，选择中间不受填充影响的区域。
            # 对 256×256 图像，这里选择行列索引 15～240，右端点 241 不包含在切片内。
            # 把内部设成 False 后，四周宽 15 的边界带仍然为 True。
            boundary_mask[radius:-radius, radius:-radius] = False
            # 布尔索引只取 True 对应位置，结果是一维数组，不再保留二维排列。
            # 例如 [10,20,30] 用 [True,False,True] 选择后得到 [10,30]。
            # 因此下面的 mean 只统计边界带，不让大片相同的内部像素稀释平均值。
            boundary_mae = boundary_difference[boundary_mask].mean()
            # ~ 对布尔数组逐元素取反：True 变 False，False 变 True。
            # ~boundary_mask 就只选内部。这里不能用 not，not 不是数组逐元素取反。
            interior_max = boundary_difference[~boundary_mask].max()
            # :.6f 是普通小数格式，保留六位小数；:.6e 便于显示非常小的数。
            print(f"  两种填充比较（sigma=5）：边界带 MAE={boundary_mae:.6f}，"
                  f"内部最大绝对差={interior_max:.6e}")

            # 两行三列时 axes 是二维数组，用 axes[行号, 列号] 选择子图，索引从 0 开始。
            fig, axes = plt.subplots(2, 3, figsize=(12, 8), layout="constrained")
            show_gray(axes[0, 0], zero, "Zero padding")
            show_gray(axes[0, 1], replicate, "Replicate padding")
            heatmap = axes[0, 2].imshow(
                boundary_difference, cmap="magma", vmin=0,
                vmax=max(float(boundary_difference.max()), 1e-15),
                interpolation="nearest",
            )
            axes[0, 2].set_title("Absolute difference")
            axes[0, 2].axis("off")
            fig.colorbar(heatmap, ax=axes[0, 2], label="Gray-value difference", shrink=0.8)
            # 同一个左上角区域并排展示。先对整图滤波，再裁剪，不对小块重新滤波。
            # [:50, :50] 取第 0～49 行、第 0～49 列，共 50×50 个像素。
            # 如果先裁小块再滤波，小块的下方和右方会变成新边界，得到不同结果。
            show_gray(axes[1, 0], zero[:50, :50], "Zero: top-left 50 x 50")
            show_gray(axes[1, 1], replicate[:50, :50], "Replicate: top-left 50 x 50")
            show_gray(axes[1, 2], f[:50, :50], "Input: top-left 50 x 50")
            fig.suptitle(f"{name}: sigma=5, kernel=31 x 31; boundary MAE={boundary_mae:.3f}")
            fig.savefig(output / f"{name}_boundary_comparison.png", dpi=150)
            plt.close(fig)

    print(f"计算结束，图片已保存到：{output}")


# 直接运行本文件时调用 main；被其他文件导入时，不自动执行实验。
if __name__ == "__main__":
    main()
