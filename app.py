"""
Meeting Recap Automation App — Online Edition
----------------------------------------------
Pipeline: Google Drive → Download → Extract Audio
          → OpenAI Whisper (STT) → Claude (Email Gen) → Gmail SMTP
"""

import os
import re
import smtplib
import ssl
import tempfile
import uuid
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import gdown
import streamlit as st
from anthropic import Anthropic
from moviepy.editor import VideoFileClip
from openai import OpenAI

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

# Mono + 64kbps -> ~50 phút audio dưới 25MB (giới hạn Whisper)
AUDIO_BITRATE = "64k"

CLAUDE_MODEL = "claude-sonnet-4-5"
WHISPER_MODEL = "whisper-1"
SYSTEM_PROMPT = """Bạn là một trợ lý chuyên viết email tổng kết (Recap) chuyên nghiệp
sau các buổi Onboarding khách hàng.

YÊU CẦU BẮT BUỘC:
- Ngôn ngữ: Tiếng Việt.
- Văn phong: Súc tích, chuyên nghiệp, lịch sự, đi thẳng vào vấn đề.
- Định dạng: HTML sạch (dùng <p>, <ul>, <li>, <strong>, <br>) để nhúng
  trực tiếp vào body email. KHÔNG bọc trong \`\`\`html ... \`\`\` markdown.
- KHÔNG thêm thẻ <html>, <head>, <body>. Chỉ trả về phần nội dung HTML.

CẤU TRÚC EMAIL BẮT BUỘC:
1. Lời chào (sử dụng tên người nhận) + cảm ơn đã tham gia buổi Onboarding.
2. Mục tiêu đã đạt được (1-2 câu tóm tắt mục tiêu chính của buổi họp).
3. Nội dung chính (Key Takeaways) - dạng bullet points.
4. Các bước tiếp theo / Action Items - nêu rõ ai phụ trách việc gì.
5. Lời kết & thông tin liên hệ hỗ trợ.
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_temp_paths():
    """Tạo unique temp paths cho mỗi session (multi-user safe)."""
    session_id = uuid.uuid4().hex[:8]
    tmp_dir = tempfile.gettempdir()
    return (
        os.path.join(tmp_dir, f"video_{session_id}.mp4"),
        os.path.join(tmp_dir, f"audio_{session_id}.mp3"),
    )


def extract_drive_file_id(url: str) -> str | None:
    if not url:
        return None
    patterns = [
        r"/file/d/([a-zA-Z0-9_-]+)",
        r"[?&]id=([a-zA-Z0-9_-]+)",
        r"/d/([a-zA-Z0-9_-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def download_video_from_drive(file_id: str, output_path: str) -> None:
    download_url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(download_url, output_path, quiet=True, fuzzy=True)
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError(
            "Không tải được file từ Google Drive. "
            "Hãy chắc chắn link đã được chia sẻ ở chế độ 'Anyone with the link'."
        )


def extract_audio(video_path: str, audio_path: str) -> None:
    clip = None
    try:
        clip = VideoFileClip(video_path)
        if clip.audio is None:
            raise RuntimeError("Video không chứa audio track.")
        clip.audio.write_audiofile(
            audio_path,
            bitrate=AUDIO_BITRATE,
            ffmpeg_params=["-ac", "1"],  # mono
            logger=None,
        )
    finally:
        if clip is not None:
            try:
                clip.close()
            except Exception:
                pass


def transcribe_audio(openai_client: OpenAI, audio_path: str) -> str:
    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    if file_size_mb > 25:
        raise RuntimeError(
            f"File audio sau khi nén vẫn nặng {file_size_mb:.1f}MB "
            "(vượt giới hạn 25MB của Whisper). Hãy thử với buổi họp ngắn hơn."
        )
    with open(audio_path, "rb") as audio_file:
        transcript = openai_client.audio.transcriptions.create(
            model=WHISPER_MODEL,
            file=audio_file,
        )
    return transcript.text


def generate_recap_email(
    anthropic_client: Anthropic, transcript: str, recipient_name: str
) -> str:
    user_prompt = (
        f"Tên người nhận email: {recipient_name}\n\n"
        f"Dưới đây là transcript đầy đủ của buổi Onboarding. "
        f"Hãy viết email recap theo đúng cấu trúc đã quy định:\n\n"
        f"---\n{transcript}\n---"
    )
    response = anthropic_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2048,
        temperature=0.4,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    html_body = "".join(
        block.text for block in response.content if block.type == "text"
    ).strip()
    html_body = re.sub(r"^\`\`\`(?:html)?\s*", "", html_body)
    html_body = re.sub(r"\s*\`\`\`$", "", html_body)
    return html_body


def send_email(
    sender_email: str,
    sender_password: str,
    recipient_email: str,
    subject: str,
    html_body: str,
) -> None:
    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["From"] = sender_email
    message["To"] = recipient_email
    message.attach(MIMEText(html_body, "html", "utf-8"))

    context = ssl.create_default_context()
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.ehlo()
        server.starttls(context=context)
        server.ehlo()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient_email, message.as_string())


def cleanup_temp_files(*paths: str) -> None:
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except OSError:
            pass


def get_secret(key: str, default: str = "") -> str:
    """Đọc từ st.secrets nếu có (production), fallback về empty (dev)."""
    try:
        return st.secrets.get(key, default)
    except (FileNotFoundError, KeyError):
        return default


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Meeting Recap Automation",
    page_icon="📧",
    layout="centered",
    menu_items={
        "About": "Tự động hoá email recap sau Onboarding • Powered by Claude + Whisper",
    },
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
    }
    .main-header h1 { color: white !important; margin: 0; font-size: 2rem; }
    .main-header p { color: rgba(255,255,255,0.9); margin: 0.5rem 0 0; }
    div[data-testid="stSidebar"] { background: #f8f9fb; }
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        font-weight: 600;
        padding: 0.6rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>📧 Meeting Recap Automation</h1>
    <p>Tự động viết & gửi email tổng kết sau buổi Onboarding</p>
</div>
""", unsafe_allow_html=True)

# ----- Sidebar -----
with st.sidebar:
    st.header("⚙️ Configuration")

    # Đọc secrets nếu admin đã cấu hình sẵn ở Streamlit Cloud
    default_anthropic = get_secret("ANTHROPIC_API_KEY")
    default_openai = get_secret("OPENAI_API_KEY")
    default_gmail = get_secret("GMAIL_ADDRESS")
    default_gmail_pwd = get_secret("GMAIL_APP_PASSWORD")

    has_secrets = bool(default_anthropic and default_openai and default_gmail and default_gmail_pwd)

    if has_secrets:
        st.success("✅ Đã cấu hình sẵn từ admin")
        use_custom = st.checkbox("Dùng credentials riêng", value=False)
    else:
        use_custom = True

    if use_custom:
        st.markdown("**🔑 API Keys**")
        anthropic_api_key = st.text_input(
            "Anthropic API Key", type="password",
            value=default_anthropic,
            help="Lấy tại: https://console.anthropic.com/settings/keys",
        )
        openai_api_key = st.text_input(
            "OpenAI API Key", type="password",
            value=default_openai,
            help="Dùng cho Whisper STT. Lấy tại: https://platform.openai.com/api-keys",
        )

        st.markdown("**📧 Gmail SMTP**")
        gmail_address = st.text_input(
            "Gmail Address", value=default_gmail, placeholder="you@gmail.com",
        )
        gmail_app_password = st.text_input(
            "Gmail App Password", type="password",
            value=default_gmail_pwd,
            help="Tạo tại: myaccount.google.com/apppasswords (16 ký tự).",
        )
    else:
        anthropic_api_key = default_anthropic
        openai_api_key = default_openai
        gmail_address = default_gmail
        gmail_app_password = default_gmail_pwd

    st.divider()
    with st.expander("ℹ️ Hướng dẫn"):
        st.caption(
            "**Drive permission**: file phải share **Anyone with the link – Viewer**.\n\n"
            "**Gmail**: dùng **App Password 16 ký tự**, không phải mật khẩu thường.\n\n"
            "**Audio limit**: Whisper giới hạn 25MB (~50 phút mono).\n\n"
            "🤖 Claude viết email • Whisper làm STT."
        )


# ----- Main form -----
with st.form("meeting_form", clear_on_submit=False):
    st.subheader("📝 Meeting Information")

    col1, col2 = st.columns(2)
    with col1:
        recipient_name = st.text_input("Recipient Name *", placeholder="VD: Anh Minh")
    with col2:
        recipient_email = st.text_input("Recipient Email *", placeholder="client@company.com")

    email_subject = st.text_input(
        "Email Subject *",
        placeholder="VD: [Recap] Onboarding session - Công ty ABC",
    )
    drive_url = st.text_input(
        "Google Drive URL *",
        placeholder="https://drive.google.com/file/d/.../view?usp=sharing",
    )
    submitted = st.form_submit_button(
        "🚀 Generate & Send Recap Email", type="primary", use_container_width=True
    )


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------
if submitted:
    missing = []
    if not anthropic_api_key: missing.append("Anthropic API Key")
    if not openai_api_key: missing.append("OpenAI API Key")
    if not gmail_address: missing.append("Gmail Address")
    if not gmail_app_password: missing.append("Gmail App Password")
    if not email_subject: missing.append("Email Subject")
    if not recipient_name: missing.append("Recipient Name")
    if not recipient_email: missing.append("Recipient Email")
    if not drive_url: missing.append("Google Drive URL")

    if missing:
        st.error(f"⛔ Vui lòng nhập đầy đủ: **{', '.join(missing)}**")
        st.stop()

    file_id = extract_drive_file_id(drive_url)
    if not file_id:
        st.error("⛔ Google Drive URL không hợp lệ. Không tách được File ID.")
        st.stop()

    anthropic_client = Anthropic(api_key=anthropic_api_key)
    openai_client = OpenAI(api_key=openai_api_key)

    temp_video, temp_audio = get_temp_paths()
    html_email_body = None
    transcript = None

    try:
        with st.status("🔄 Đang xử lý...", expanded=True) as status:
            st.write("📥 **Step 1/5** — Đang tải video từ Google Drive...")
            try:
                download_video_from_drive(file_id, temp_video)
            except Exception as e:
                raise RuntimeError(f"Lỗi khi tải video: {e}") from e
            size_mb = os.path.getsize(temp_video) / (1024 * 1024)
            st.write(f"✅ Tải xong ({size_mb:.1f}MB).")

            st.write("🎵 **Step 2/5** — Đang trích xuất audio (mono, 64kbps)...")
            try:
                extract_audio(temp_video, temp_audio)
            except Exception as e:
                raise RuntimeError(f"Lỗi khi trích xuất audio: {e}") from e
            audio_size_mb = os.path.getsize(temp_audio) / (1024 * 1024)
            st.write(f"✅ Audio extracted ({audio_size_mb:.1f}MB).")

            st.write("📝 **Step 3/5** — Đang transcribe bằng Whisper...")
            try:
                transcript = transcribe_audio(openai_client, temp_audio)
            except Exception as e:
                raise RuntimeError(f"Lỗi Whisper API: {e}") from e
            if not transcript.strip():
                raise RuntimeError("Transcript rỗng. Hãy kiểm tra lại file audio.")
            st.write(f"✅ Transcript xong ({len(transcript)} ký tự).")

            st.write(f"🤖 **Step 4/5** — Đang sinh email recap (`{CLAUDE_MODEL}`)...")
            try:
                html_email_body = generate_recap_email(
                    anthropic_client, transcript, recipient_name
                )
            except Exception as e:
                raise RuntimeError(f"Lỗi Anthropic API: {e}") from e
            st.write("✅ Email content đã sẵn sàng.")

            st.write(f"📤 **Step 5/5** — Đang gửi email tới `{recipient_email}`...")
            try:
                send_email(
                    sender_email=gmail_address,
                    sender_password=gmail_app_password,
                    recipient_email=recipient_email,
                    subject=email_subject,
                    html_body=html_email_body,
                )
            except smtplib.SMTPAuthenticationError as e:
                raise RuntimeError(
                    "Sai Gmail / App Password. Hãy đảm bảo bạn dùng "
                    "App Password (16 ký tự) chứ không phải mật khẩu Gmail thường."
                ) from e
            except Exception as e:
                raise RuntimeError(f"Lỗi khi gửi email: {e}") from e

            status.update(label="✅ Hoàn tất!", state="complete", expanded=False)

        st.success(f"🎉 Đã gửi email recap tới **{recipient_email}** thành công!")

        with st.expander("👀 Xem trước nội dung email đã gửi", expanded=True):
            st.markdown(html_email_body, unsafe_allow_html=True)

        with st.expander("📜 Xem transcript gốc"):
            st.text(transcript)

        st.download_button(
            "💾 Tải HTML email", data=html_email_body,
            file_name=f"recap_{recipient_name.replace(' ', '_')}.html",
            mime="text/html",
        )

    except Exception as e:
        st.error(f"❌ Quá trình xử lý thất bại: {e}")

    finally:
        cleanup_temp_files(temp_video, temp_audio)

st.markdown("---")
st.caption("🤖 Powered by **Claude Sonnet 4.5** + **OpenAI Whisper** • Built with Streamlit")
