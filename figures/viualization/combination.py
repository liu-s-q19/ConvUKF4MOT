from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def merge_images_to_pdf(image_paths, output_pdf):
    # 创建一个 PDF 画布
    c = canvas.Canvas(output_pdf, pagesize=letter)

    # 设置 PDF 的尺寸
    pdf_width, pdf_height = letter

    # 计算每个图像在 PDF 中的位置和大小
    image_width = pdf_width
    image_height = pdf_height / len(image_paths)

    # 遍历图像路径并将其添加到 PDF 中
    for i, image_path in enumerate(image_paths):
        # 打开图像
        image = Image.open(image_path)

        # 缩放图像以适应 PDF 页面
        image.thumbnail((image_width, image_height))

        # 计算图像在 PDF 中的位置
        x = 0
        y = pdf_height - (i + 1) * image_height

        # 将图像添加到 PDF 中
        c.drawImage(image_path, x, y, width=image.width, height=image.height, preserveAspectRatio=True)

        # 如果不是最后一张图像，则添加新的页面
        if i < len(image_paths) - 1:
            c.showPage()

    # 保存 PDF
    c.save()

# 要合成的图像路径列表
image_paths = ["convukf/000113.jpg", "convukf/000121.jpg", "convukf/000129.jpg"]

# 输出 PDF 的文件名
output_pdf = "convukf_vual.pdf"

# 合成图像并保存为 PDF
merge_images_to_pdf(image_paths, output_pdf)

print("PDF 已生成：{}".format(output_pdf))
