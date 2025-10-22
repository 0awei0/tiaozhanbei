# -*- coding: utf-8 -*-
import os
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------- 页面与样式 ----------------------
st.set_page_config(page_title="ChatBot • Streamlit x Ark", page_icon="💬", layout="centered")

CUSTOM_CSS = """
<style>
.block-container {max-width: 780px; padding-bottom: 160px !important;}
.stChatMessage {padding: 0.35rem 0.2rem;}
.stChatMessage .stMarkdown, .stChatMessage p {font-size: 1rem; line-height: 1.65;}
.chat-image {border-radius: 12px; box-shadow: 0 3px 14px rgba(0,0,0,0.08);} 
[data-testid="stChatInputContainer"] {
    position: fixed; bottom: 0; left: 0; right: 0;
    background-color: var(--background-color);
    padding: 0.6rem 1rem 1rem 1rem;
    box-shadow: 0 -2px 6px rgba(0,0,0,0.06); z-index: 9999;
}
[data-testid="stPopover"] {
    position: fixed !important;
    left: 65rem; bottom: 6.5em; z-index: 1000; width: 5%;
}
.small-uploader [data-testid="stFileUploader"] section div{ padding: 6px 0 !important; }
.small-uploader [data-testid="stFileUploader"] label{ display: none; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------- 环境初始化 ----------------------
load_dotenv()
st.sidebar.header("⚙️ 设置")
default_model = "doubao-seed-1-6-vision-250815"
base_url = st.sidebar.text_input("Ark Base URL", value="https://ark.cn-beijing.volces.com/api/v3")
model = st.sidebar.text_input("Model", value=default_model)
max_rounds = st.sidebar.number_input(
    "用于构造上下文的最近轮数", min_value=1, max_value=10, value=3, step=1,
    help="仅带入最近 N 轮（用户+助手配对）的文本作为上下文；图片不会带入下一轮。",
)
st.sidebar.caption("在项目根目录放置 .env，并写入 ARK_API_KEY=xxxxx")

client = OpenAI(base_url=base_url, api_key=os.environ.get("ARK_API_KEY"))

# ---------------------- 会话状态 ----------------------
if "messages" not in st.session_state:
    # 历史仅存文本
    st.session_state.messages: List[Dict[str, Any]] = []
if "pending_image" not in st.session_state:
    st.session_state.pending_image: Optional[Tuple[bytes, str]] = None  # (bytes, mime)
# 用于强制重置 file_uploader 的递增 key
if "upload_rev" not in st.session_state:
    st.session_state.upload_rev = 0


# ---------------------- 辅助函数 ----------------------
def image_to_data_url(file_bytes: bytes, mime: str) -> str:
    import base64 as _b64
    b64 = _b64.b64encode(file_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def build_context_messages(history: List[Dict[str, Any]], max_pairs: int) -> List[Dict[str, Any]]:
    text_only: List[Dict[str, Any]] = []
    for m in history:
        if m.get("text"):
            text_only.append({"role": m["role"], "content": [{"type": "text", "text": m["text"]}]})
    if max_pairs > 0:
        text_only = text_only[-(max_pairs * 2):]
    return text_only


# ---------------------- 历史渲染（仅文本） ----------------------
st.title("💬 ChatBot (Streamlit × Ark)")
st.caption("支持图片上传与流式输出 · **图片仅在当前轮显示，不进入下一轮历史**")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m.get("text"):
            st.markdown(m["text"])

# ---------------------- 右下角固定的「＋」上传 Popover ----------------------
with st.popover("＋", use_container_width=False):
    st.markdown("**上传图片**（仅用于当前轮，不进入下一轮上下文）")
    with st.container():
        st.write("")
        # 关键：使用递增的 key，确保每次发送后控件被“重建”，不会把旧文件再次注入
        uploader_key = f"uploader_{st.session_state.upload_rev}"
        up = st.file_uploader(
            " ", type=["png", "jpg", "jpeg", "webp", "bmp"],
            accept_multiple_files=False, key=uploader_key
        )
        if up is not None:
            mime = up.type or "image/png"
            st.session_state.pending_image = (up.read(), mime)
            st.success("图片已就绪，本轮发送时会一并提交。")

# ---------------------- 聊天输入（固定底部） ----------------------
user_input = st.chat_input("发送消息（可先点右下角＋上传图片）…")

# ---------------------- 处理提交（含流式输出） ----------------------
if user_input is not None:
    # 读取并清空“待发送图片”，保证不进入下一轮
    img_bytes, img_mime = None, None
    if st.session_state.pending_image is not None:
        img_bytes, img_mime = st.session_state.pending_image
    st.session_state.pending_image = None  # 清理缓存

    # 发送后：递增 rev，下一次重绘时 file_uploader 会是全新控件（没有旧文件）
    st.session_state.upload_rev += 1

    # 本轮用户气泡：显示文本 +（若有）图片，但**不写入历史的图片字段**
    with st.chat_message("user"):
        if user_input.strip():
            st.markdown(user_input)
        if img_bytes:
            st.image(img_bytes, use_container_width=True)

    # 历史仅存文本
    st.session_state.messages.append({"role": "user", "text": user_input})

    # 组装上下文（仅文本）
    context_msgs = build_context_messages(st.session_state.messages, max_rounds)

    # 当前用户消息内容（图 + 文本）——图像只随本轮一起发送给模型
    user_content: List[Dict[str, Any]] = []
    if img_bytes and img_mime:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": image_to_data_url(img_bytes, img_mime)},
        })
    if user_input.strip():
        user_content.append({"type": "text", "text": user_input})

    # 助手气泡：流式输出
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=context_msgs + [{"role": "user", "content": user_content}],
                stream=True,
            )
            for chunk in stream:
                try:
                    delta = chunk.choices[0].delta
                    piece = getattr(delta, "content", None)
                except Exception:
                    piece = getattr(chunk.choices[0], "text", None)
                if piece:
                    full_response += piece
                    placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response if full_response else "（空响应）")
        except Exception as e:
            full_response = f"调用模型出错：{e}"
            placeholder.error(full_response)

    # 写入历史（仅文本）
    st.session_state.messages.append({"role": "assistant", "text": full_response})

# ---------------------- 页脚 ----------------------
st.markdown(
"""
<hr/>
<p style="font-size:0.9rem;opacity:0.7;">
提示：侧边栏可调整 <em>用于构造上下文的最近轮数</em>（默认 3 轮）。图片仅参与当前轮对话，不会被记入历史上下文；每次发送后上传控件都会被重置。
</p>
""",
    unsafe_allow_html=True
)
