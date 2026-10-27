from PIL import Image
import numpy as np
import os

# 使用 for 循环遍历 0 到 9
for i in range(10):
    input_file = f'{i}.png'       # 动态生成输入文件名: 0.png, 1.png...
    output_file = f'image{i}.bin' # 动态生成输出文件名: image0.bin, image1.bin...

    # 加个保险：检查图片文件是否存在
    if not os.path.exists(input_file):
        print(f"⚠️ 找不到图片 {input_file}，已跳过。")
        continue

    # 1. 打开你的 PNG 图片并强制转换为单通道灰度图 ('L')
    img = Image.open(input_file).convert('L')

    # 2. 转换为 numpy 数组，数据类型指定为 float32
    img_array = np.array(img, dtype=np.float32)

    # 3. 归一化：将 0~255 的像素值映射到 0.0~1.0 之间
    img_array = img_array / 255.0

    # 4. 直接导出为纯二进制文件
    img_array.tofile(output_file)

    print(f"✅ {input_file} 转换成功 -> {output_file} (3136 字节)")

print("\n🎉 全部 10 张图片处理完毕！")