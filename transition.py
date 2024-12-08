import argparse
import subprocess
import os
import sys
import platform
from PIL import Image, ExifTags
from reportlab.lib.pagesizes import A4, LETTER
from reportlab.pdfgen import canvas
from tempfile import TemporaryDirectory

def get_audiveris_command(audiveris_script, args):
    """
    构建Audiveris命令，根据操作系统选择适当的脚本。
    """
    if platform.system() == "Windows":
        command = [audiveris_script] + args
    else:
        command = ["bash", audiveris_script] + args
    return command

def correct_image_orientation(img):
    """
    根据EXIF数据修正图像方向。
    """
    try:
        exif = img._getexif()
        if exif is not None:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            exif_orientation = exif.get(orientation, None)
            if exif_orientation == 3:
                img = img.rotate(180, expand=True)
            elif exif_orientation == 6:
                img = img.rotate(270, expand=True)
            elif exif_orientation == 8:
                img = img.rotate(90, expand=True)
    except Exception as e:
        print(f"无法修正图像方向: {e}")
    return img

def resize_image(img, max_pixels=20000000):
    """
    调整图像大小，确保总像素数不超过max_pixels，保持宽高比。
    """
    width, height = img.size
    total_pixels = width * height
    if total_pixels > max_pixels:
        scale_factor = (max_pixels / total_pixels) ** 0.5
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        img = img.resize((new_width, new_height), Image.LANCZOS)
        print(f"缩放图片到: {new_width}x{new_height} pixels")
    else:
        print("无需缩放图片")
    return img

def preprocess_image(image_path, temp_dir, max_pixels=20000000):
    """
    预处理图片：修正方向，调整尺寸。
    将处理后的图片保存到临时目录中，避免在原文件夹中留下处理后的文件。
    返回预处理后的图片路径。
    """
    try:
        with Image.open(image_path) as img:
            # 修正图像方向
            img = correct_image_orientation(img)

            # 调整图像尺寸
            img = resize_image(img, max_pixels)

            # 保存预处理后的图片到临时目录，保持PNG格式
            preprocessed_filename = f"preprocessed_{os.path.splitext(os.path.basename(image_path))[0]}.png"
            preprocessed_path = os.path.join(temp_dir, preprocessed_filename)
            img.save(preprocessed_path, format='PNG')
            print(f"预处理并保存图片: {preprocessed_path}")

            return preprocessed_path

    except Exception as e:
        print(f"预处理图片失败: {image_path}, 错误: {e}")
        return None

def merge_images_to_pdf_reportlab(images, output_pdf_path, page_size=A4, margin=50):
    """
    使用ReportLab将图片列表合并为一个多页PDF文件，控制每张图片的尺寸。
    """
    try:
        print(f"正在合并 {len(images)} 张图片为PDF: {output_pdf_path} (页面大小: {page_size})")
        c = canvas.Canvas(output_pdf_path, pagesize=page_size)
        page_width, page_height = page_size

        for img_path in images:
            try:
                with Image.open(img_path) as img:
                    img_width, img_height = img.size
                    aspect = img_width / img_height

                    # 计算图片在PDF中的尺寸，保持比例，适应页面大小减去边距
                    available_width = page_width - 2 * margin
                    available_height = page_height - 2 * margin
                    if aspect > 1:
                        # 宽图
                        display_width = min(available_width, img_width)
                        display_height = display_width / aspect
                    else:
                        # 高图
                        display_height = min(available_height, img_height)
                        display_width = display_height * aspect

                    # 计算图片插入位置
                    x = (page_width - display_width) / 2
                    y = (page_height - display_height) / 2

                    # 在PDF中绘制图片
                    c.drawImage(img_path, x, y, width=display_width, height=display_height)
                    c.showPage()
            except Exception as img_e:
                print(f"插入图片失败: {img_path}, 错误: {img_e}")

        c.save()
        print(f"图片成功合并为PDF: {output_pdf_path}")
    except Exception as e:
        print(f"合并图片为PDF失败: {e}")
        sys.exit(1)

def convert_pdf_to_musicxml(audiveris_script, pdf_path, output_dir, verbose=False):
    """
    使用Audiveris将PDF转换为MusicXML。
    """
    if not os.path.isfile(pdf_path):
        print(f"错误：PDF文件不存在 - {pdf_path}")
        sys.exit(1)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    args = [
        "-batch",
        "-export",
        "-output", output_dir,
        "--", pdf_path
    ]

    if verbose:
        args.insert(-2, "-verbose")

    command = get_audiveris_command(audiveris_script, args)

    try:
        print(f"正在转换PDF为MusicXML: {pdf_path}")
        subprocess.run(command, check=True)
        print(f"转换成功，文件保存在 {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"转换失败: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="使用Audiveris将PDF或图片转换为MusicXML格式的XML文件。"
    )

    parser.add_argument(
        '--audiveris',
        required=True,
        help="Audiveris的启动脚本路径，例如 Windows: C:\\path\\to\\Audiveris.bat, Unix-like: /path/to/Audiveris.sh"
    )

    parser.add_argument(
        '--mode',
        required=True,
        choices=['pdf', 'image-merge'],
        help="转换模式：'pdf'（PDF转XML）、'image-merge'（合并图片为PDF后转XML）"
    )

    parser.add_argument(
        '--input',
        required=True,
        help="输入文件路径（PDF文件）或输入文件夹路径（图片文件夹）"
    )

    parser.add_argument(
        '--output',
        required=True,
        help="输出目录路径，用于保存生成的XML文件"
    )

    parser.add_argument(
        '--dpi',
        type=int,
        default=350,
        help="合并PDF时的DPI（默认：300）"
    )

    parser.add_argument(
        '--page-size',
        type=str,
        choices=['A4', 'LETTER'],
        default='A4',
        help="PDF页面大小（默认：A4）"
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help="启用详细日志输出"
    )

    args = parser.parse_args()

    audiveris_script = args.audiveris
    mode = args.mode
    input_path = args.input
    output_dir = args.output
    dpi = args.dpi
    page_size_input = args.page_size
    verbose = args.verbose

    # 设置页面大小
    if page_size_input == 'A4':
        PAGE_SIZE = A4
    elif page_size_input == 'LETTER':
        PAGE_SIZE = LETTER
    else:
        PAGE_SIZE = A4  # 默认

    # 检查Audiveris启动脚本是否存在
    if not os.path.isfile(audiveris_script):
        print(f"错误：Audiveris启动脚本不存在 - {audiveris_script}")
        sys.exit(1)

    # 根据操作系统设置脚本执行权限（Unix-like系统）
    if platform.system() != "Windows":
        if not os.access(audiveris_script, os.X_OK):
            print(f"设置可执行权限给脚本: {audiveris_script}")
            os.chmod(audiveris_script, 0o755)

    if mode == 'pdf':
        # 直接转换PDF为MusicXML
        convert_pdf_to_musicxml(audiveris_script, input_path, output_dir, verbose)

    elif mode == 'image-merge':
        # 使用临时目录存放预处理后的图片和合并的PDF
        with TemporaryDirectory() as temp_dir:
            supported_formats = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp')
            images = sorted([
                f for f in os.listdir(input_path)
                if f.lower().endswith(supported_formats)
            ])

            if not images:
                print(f"错误：目录中没有支持的图片文件 - {input_path}")
                sys.exit(1)

            preprocessed_images = []
            for image in images:
                image_path = os.path.join(input_path, image)
                preprocessed_path = preprocess_image(image_path, temp_dir)
                if preprocessed_path:
                    preprocessed_images.append(preprocessed_path)

            if not preprocessed_images:
                print("错误：所有图片预处理失败。")
                sys.exit(1)

            # 合并预处理后的图片为PDF
            output_pdf_path = os.path.join(temp_dir, "merged.pdf")
            merge_images_to_pdf_reportlab(preprocessed_images, output_pdf_path, page_size=PAGE_SIZE)

            # 转换合并后的PDF为MusicXML
            convert_pdf_to_musicxml(audiveris_script, output_pdf_path, output_dir, verbose)

    else:
        print("错误：未知的转换模式")
        sys.exit(1)

if __name__ == "__main__":
    main()
