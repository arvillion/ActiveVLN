import cv2
from PIL import Image, ImageDraw, ImageFont
import textwrap
import numpy as np
import av

def pad_frame_bottom(frame_np, target_height, bg_color=(255, 255, 255)):
    h, w, _ = frame_np.shape
    if h >= target_height:
        return frame_np
    pad_bottom = target_height - h
    padded = cv2.copyMakeBorder(frame_np, 0, pad_bottom, 0, 0,
                                cv2.BORDER_CONSTANT, value=bg_color)
    return padded


def estimate_wrap_width(canvas_width, font_size, left_margin=10, right_margin=10):
    avg_char_width = font_size * 0.6
    usable_width = canvas_width - left_margin - right_margin
    return int(usable_width / avg_char_width)

def render_frame_pil(image_np, status_info, instruction, user_text, assistant_text,
                     image_width=400, image_height=400,
                     text_width=400, font_path="arial.ttf", font_size=16,
                     bold_font_path=None):

    # 加载字体
    font = ImageFont.truetype(font_path, font_size)
    if bold_font_path is None:
        bold_font = ImageFont.truetype(font_path, font_size)
    else:
        bold_font = ImageFont.truetype(bold_font_path, font_size)
    line_height = font_size + 6

    # 1. 状态栏文本
    status_text = "   ".join([f"{k}: {v}" for k, v in status_info.items()])
    status_height = line_height + 4

    # 2. Resize 图像区域
    image_np_resized = cv2.resize(image_np, (image_width, image_height))
    image_pil = Image.fromarray(image_np_resized)

    # 在图像顶部绘制状态栏
    draw_img = ImageDraw.Draw(image_pil)
    draw_img.rectangle([(0, 0), (image_width, status_height)], fill=(0, 0, 0))
    draw_img.text((5, 2), status_text, font=bold_font, fill=(255, 255, 255))

    # 3. 包装文字内容（右侧区域）
    wrap_width = estimate_wrap_width(text_width, font_size)

    # Instruction
    instr_prefix = "Instruction: "
    instr_lines = textwrap.wrap(instruction, width=wrap_width)

    # User & Assistant
    user_lines = textwrap.wrap(user_text, width=wrap_width)
    assistant_lines = textwrap.wrap(assistant_text, width=wrap_width)

    # 4. 估计右侧区域总高度
    text_block_height = (
        line_height +  # Instruction prefix
        len(instr_lines) * line_height +
        line_height + len(user_lines) * line_height +
        line_height + len(assistant_lines) * line_height +
        10
    )

    # 5. 画布高度 = max(图像高度, 文字高度)
    canvas_height = max(image_height, text_block_height)
    canvas_width = image_width + text_width

    # 6. 创建总画布
    canvas = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))
    canvas.paste(image_pil, (0, 0))
    draw = ImageDraw.Draw(canvas)

    # 7. 绘制右侧文字
    x0 = image_width + 10
    y = 10

    # Instruction
    draw.text((x0, y), instr_prefix, font=bold_font, fill=(100, 0, 150))
    y += line_height
    for line in instr_lines:
        draw.text((x0, y), line, font=font, fill=(0, 0, 0))
        y += line_height

    # User
    if user_text:
        draw.text((x0, y), "User:", font=bold_font, fill=(0, 0, 200))
        y += line_height
        for line in user_lines:
            draw.text((x0 + 10, y), line, font=font, fill=(0, 0, 0))
            y += line_height

    # Assistant
    if assistant_text:
        draw.text((x0, y), "Assistant:", font=bold_font, fill=(0, 120, 0))
        y += line_height
        for line in assistant_lines:
            draw.text((x0 + 10, y), line, font=font, fill=(0, 0, 0))
            y += line_height

    return np.array(canvas)


def save_video(frames_np_list, output_path, fps=1):
    # 统一高度
    max_height = max(f.shape[0] for f in frames_np_list)
    width = frames_np_list[0].shape[1]

    # 填充每帧
    padded_frames = [pad_frame_bottom(f, max_height) for f in frames_np_list]

    # 打开输出文件并添加 H264 视频流
    container = av.open(output_path, mode='w')
    stream = container.add_stream('libx264', rate=fps)
    stream.width = width
    stream.height = max_height
    stream.pix_fmt = 'yuv420p'  # H.264 要求的格式

    for frame in padded_frames:
        # 转成 RGB -> BGR -> PyAV 识别的格式
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_frame = av.VideoFrame.from_ndarray(bgr_frame, format='bgr24')
        for packet in stream.encode(video_frame):
            container.mux(packet)

    # 刷新剩余数据
    for packet in stream.encode():
        container.mux(packet)

    container.close()