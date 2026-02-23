import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import mediapipe as mp
from io import BytesIO
from PIL import Image
import traceback
import time

# --- 1. 曜石黑金 UI 架构 ---
st.set_page_config(page_title="GolfAsistant | Black Gold", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .stApp { background-color: #121212; color: #FFFFFF; }
    [data-testid="stSidebar"] { background-color: #080808 !important; border-right: 3px solid #D4AF37 !important; }
    [data-testid="stSidebar"] .stMarkdown h1, [data-testid="stSidebar"] .stMarkdown h2, 
    [data-testid="stSidebar"] .stMarkdown h3, [data-testid="stSidebar"] label {
        color: #D4AF37 !important; font-weight: 800 !important;
    }
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #D4AF37 0%, #8A6D3B 100%) !important;
        color: #000000 !important; border: 2px solid #FFD700 !important;
        border-radius: 4px !important; width: 100% !important;
    }
    .report-box { border: 1px solid #D4AF37; border-radius: 8px; padding: 30px; background: #1A1A1A; margin-bottom: 30px; }
    section[data-testid="stFileUploadDropzone"] { background-color: #222222 !important; border: 1px dashed #D4AF37 !important; }
    [data-testid="stMetric"] { background-color: #000000 !important; border: 1px solid #D4AF37 !important; }
    </style>
    """, unsafe_allow_html=True)

TEMP_DIR = "temp_output"
if not os.path.exists(TEMP_DIR): 
    os.makedirs(TEMP_DIR)

# --- 2. AI 深度分析核心引擎 ---

def get_action_data(video_path):
    import mediapipe as mp
    from mediapipe.python.solutions import pose as mp_pose
    
    y_coords = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    # 增加 model_complexity 以提升 Pro 视频识别率
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)
            if results.pose_landmarks:
                # 抓取右手腕坐标
                y_coords.append(results.pose_landmarks.landmark[16].y)
            else:
                y_coords.append(np.nan)
    cap.release()

    arr = np.array(y_coords)
    if len(arr) == 0 or np.all(np.isnan(arr)):
        return np.zeros(100), [0, 20, 40, 60, 80, 99], (0, 99), fps

    # 插值处理 NaN，防止曲线断裂
    mask = np.isnan(arr)
    if np.any(mask):
        arr[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), arr[~mask])

    if len(arr) < 3:
        arr = np.repeat(arr, 10) if len(arr) > 0 else np.zeros(10)

    dy = np.abs(np.gradient(arr))
    moving_indices = np.where(dy > (np.max(dy) * 0.1))[0]
    start_f, end_f = (moving_indices[0], moving_indices[-1]) if len(moving_indices) > 0 else (0, len(arr) - 1)
    key_indices = np.linspace(start_f, end_f, 6).astype(int)
    return arr, key_indices, (start_f, end_f), fps

def render_premium_video(video_path, y_data, swing_window, fps):
    cap = cv2.VideoCapture(video_path)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 使用唯一文件名，防止多用户冲突导致文件被占用
    ts = int(time.time())
    raw_out = os.path.join(TEMP_DIR, f"raw_{ts}.mp4")
    final_out = os.path.join(TEMP_DIR, f"final_{ts}.mp4")
    
    out = cv2.VideoWriter(raw_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w + 400, h))
    
    for i in range(len(y_data)):
        ret, frame = cap.read()
        if not ret: break
        fig, ax = plt.subplots(figsize=(4, h/100), dpi=100)
        fig.patch.set_facecolor('#000000')
        ax.plot(y_data, color='#D4AF37', linewidth=3)
        ax.axvline(x=i, color='#FFFFFF', linewidth=2)
        ax.invert_yaxis()
        ax.axis('off')
        fig.canvas.draw()
        graph_img = cv2.cvtColor(np.array(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
        plt.close(fig)
        out.write(np.hstack((frame, cv2.resize(graph_img, (400, h)))))
    
    cap.release()
    out.release()
    
    # 尝试调用 FFmpeg
    os.system(f'ffmpeg -y -i "{raw_out}" -vcodec libx264 -crf 28 "{final_out}"')
    
    # 检查 FFmpeg 是否生成成功，如果不成功，回退使用 raw 视频防止报错
    return final_out if os.path.exists(final_out) else raw_out

def get_pose_frame(video_path, frame_idx):
    import mediapipe as mp
    from mediapipe.python.solutions import pose as mp_pose
    from mediapipe.python.solutions import drawing_utils as mp_drawing

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read(); cap.release()
    if not ret: return None
    
    with mp_pose.Pose(static_image_mode=True, model_complexity=1) as pose:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if res.pose_landmarks:
            mp_drawing.draw_landmarks(rgb, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    return rgb

# --- 3. 页面逻辑 ---

with st.sidebar:
    st.title("🏆 GolfAsistant")
    st.markdown("尊享级 AI 深度分析对比系统")
    st.markdown("---")
    u_file = st.file_uploader("学员练习视频 (High Quality)", type=["mp4", "mov"])
    p_file = st.file_uploader("职业对标视频 (PGA Pro)", type=["mp4", "mov"])
    st.markdown("---")
    analyze_btn = st.button("开启 AI 深度分析 ⚡")

if u_file and p_file:
    if analyze_btn:
        try:
            with st.status("正在启动 AI 深度分析引擎...", expanded=True) as status:
                # 唯一化临时视频路径
                ts = int(time.time())
                u_p = os.path.join(TEMP_DIR, f"u_{ts}.mp4")
                p_p = os.path.join(TEMP_DIR, f"p_{ts}.mp4")
                
                with open(u_p, "wb") as f: f.write(u_file.getbuffer())
                with open(p_p, "wb") as f: f.write(p_file.getbuffer())

                u_data, u_idx, u_win, u_fps = get_action_data(u_p)
                p_data, p_idx, p_win, p_fps = get_action_data(p_p)

                # 模块1: 指标
                c1, c2, c3 = st.columns(3)
                c1.metric("学员挥杆时长", f"{u_win[1]-u_win[0]} Frames")
                c2.metric("职业选手时长", f"{p_win[1]-p_win[0]} Frames")
                match = max(0, 100-abs((u_win[1]-u_win[0])-(p_win[1]-p_win[0])))
                c3.metric("AI 对齐匹配度", f"{match}%")

                # 模块2: 轨迹图
                st.markdown('<div class="report-box"><h3>📊 手腕 AI 轨迹对比分析</h3>', unsafe_allow_html=True)
                fig_t, ax = plt.subplots(figsize=(12, 4))
                fig_t.patch.set_facecolor('#1A1A1A')
                ax.plot(np.linspace(0, 100, len(u_data)), u_data, label="Student", color="#FFFFFF", linewidth=3)
                ax.plot(np.linspace(0, 100, len(p_data)), p_data, label="Pro", color="#D4AF37", linestyle="--", linewidth=3)
                ax.invert_yaxis()
                ax.set_facecolor('#1A1A1A')
                ax.tick_params(colors='#D4AF37')
                ax.legend(facecolor='#000000', edgecolor='#D4AF37', labelcolor='white')
                st.pyplot(fig_t)
                buf_track = BytesIO(); fig_t.savefig(buf_track, format="png"); plt.close(fig_t)
                st.markdown('</div>', unsafe_allow_html=True)

                # 模块3: 对齐矩阵
                st.markdown('<div class="report-box"><h3>📸 AI 关键阶段对比 (Stage 1-6)</h3>', unsafe_allow_html=True)
                m_imgs = []
                blank_img = np.zeros((500, 350, 3), dtype=np.uint8) 
                
                for i in range(6):
                    img_u = get_pose_frame(u_p, u_idx[i])
                    img_p = get_pose_frame(p_p, p_idx[i])
                    
                    # 鲁棒性：黑图占位，防止 np.hstack 崩溃
                    res_u = cv2.resize(img_u, (350, 500)) if img_u is not None else blank_img
                    res_p = cv2.resize(img_p, (350, 500)) if img_p is not None else blank_img
                    
                    m_imgs.append(np.hstack((res_u, res_p)))
                
                # 确保拼接逻辑安全
                r1, r2 = np.hstack(m_imgs[:3]), np.hstack(m_imgs[3:])
                full_m = np.vstack((r1, r2))
                st.image(full_m, use_container_width=True)
                buf_matrix = BytesIO(); Image.fromarray(full_m).save(buf_matrix, format="png")
                st.markdown('</div>', unsafe_allow_html=True)

                # 模块4: 动态视频
                v_path = render_premium_video(u_p, u_data, u_win, u_fps)
                st.video(v_path)
                
                status.update(label="✅ AI 深度分析报告就绪", state="complete")

            with st.sidebar:
                st.markdown("---")
                st.subheader("📥 导出分析数据")
                st.download_button("📊 导出轨迹曲线图", buf_track.getvalue(), "track.png", use_container_width=True)
                st.download_button("📸 导出对比快照", buf_matrix.getvalue(), "matrix.png", use_container_width=True)
                with open(v_path, "rb") as f:
                    st.download_button("🎥 导出分析录影", f, "video.mp4", use_container_width=True)

        except Exception as e:
            st.error(f"分析引擎中断: {e}")
            st.code(traceback.format_exc())
else:
    st.info("💎 请在左侧侧边栏上传素材。系统将自动启动尊享级 AI 深度分析流程。")
