import argparse
import subprocess
import os
import sys


def convert_pdf_to_musicxml(audiveris_path, pdf_path, output_dir):
    """
    使用Audiveris将PDF转换为MusicXML。
    """
    if not os.path.isfile(pdf_path):
        print(f"错误：PDF文件不存在 - {pdf_path}")
        sys.exit(1)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    command = [
        'java',
        '-jar',
        audiveris_path,
        '-batch',
        pdf_path,
        '-export',
        output_dir
    ]

    try:
        print(f"正在转换PDF: {pdf_path}")
        subprocess.run(command, check=True)
        print(f"转换成功，文件保存在 {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"转换失败: {e}")
        sys.exit(1)


def convert_images_to_musicxml(audiveris_path, images_folder, output_dir):
    """
    使用Audiveris将图片文件夹中的图片转换为MusicXML。
    """
    if not os.path.isdir(images_folder):
        print(f"错误：图片文件夹不存在 - {images_folder}")
        sys.exit(1)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    command = [
        'java',
        '-jar',
        audiveris_path,
        '-batch',
        images_folder,
        '-export',
        output_dir
    ]

    try:
        print(f"正在转换图片文件夹: {images_folder}")
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
        help="Audiveris的JAR文件路径，例如 /path/to/audiveris.jar"
    )

    parser.add_argument(
        '--mode',
        required=True,
        choices=['pdf', 'image'],
        help="转换模式：'pdf' 或 'image'"
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

    args = parser.parse_args()

    audiveris_path = args.audiveris
    mode = args.mode
    input_path = args.input
    output_dir = args.output

    # 检查Audiveris JAR文件是否存在
    if not os.path.isfile(audiveris_path):
        print(f"错误：Audiveris JAR文件不存在 - {audiveris_path}")
        sys.exit(1)

    if mode == 'pdf':
        convert_pdf_to_musicxml(audiveris_path, input_path, output_dir)
    elif mode == 'image':
        convert_images_to_musicxml(audiveris_path, input_path, output_dir)
    else:
        print("错误：未知的转换模式")
        sys.exit(1)


if __name__ == "__main__":
    main()
