import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import mediapipe as mp
from io import BytesIO
from PIL import Image
import traceback

# --- 1. 曜石黑金 UI 架构 ---
st.set_page_config(page_title="GolfAsistant | Black Gold", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    /* 全局背景：极简深灰，衬托主体 */
    .stApp { background-color: #121212; color: #FFFFFF; }
    
    /* 侧边栏：曜石黑 + 黄金分割线 */
    [data-testid="stSidebar"] {
        background-color: #080808 !important;
        border-right: 3px solid #D4AF37 !important;
    }
    
    /* 侧边栏高对比度文字 */
    [data-testid="stSidebar"] .stMarkdown h1, 
    [data-testid="stSidebar"] .stMarkdown h2, 
    [data-testid="stSidebar"] .stMarkdown h3,
    [data-testid="stSidebar"] label {
        color: #D4AF37 !important; /* 黄金色 */
        font-weight: 800 !important;
    }
    [data-testid="stSidebar"] .stMarkdown p {
        color: #FFFFFF !important;
        font-size: 1.1rem;
    }

    /* AI 深度分析按钮：黑金重金属质感 */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #D4AF37 0%, #8A6D3B 100%) !important;
        color: #000000 !important; /* 黑字更显尊贵 */
        border: 2px solid #FFD700 !important;
        border-radius: 4px !important;
        height: 3em !important;
        width: 100% !important;
        font-size: 1.2rem !important;
        text-transform: uppercase;
        box-shadow: 0 4px 20px rgba(212, 175, 55, 0.4);
    }

    /* 报告容器：深邃背景 + 金色边框 */
    .report-box {
        border: 1px solid #D4AF37;
        border-radius: 8px;
        padding: 30px;
        background: #1A1A1A;
        margin-bottom: 30px;
    }
    
    /* 修正上传组件在深色模式下的显示 */
    section[data-testid="stFileUploadDropzone"] {
        background-color: #222222 !important;
        border: 1px dashed #D4AF37 !important;
        color: #FFFFFF !important;
    }
    
    /* 数据卡片高对比度 */
    [data-testid="stMetric"] {
        background-color: #000000 !important;
        border: 1px solid #D4AF37 !important;
        border-radius: 5px;
    }
    [data-testid="stMetricLabel"] { color: #D4AF37 !important; }
    [data-testid="stMetricValue"] { color: #FFFFFF !important; }
    </style>
    """, unsafe_allow_html=True)

TEMP_DIR = "temp_output"
if not os.path.exists(TEMP_DIR): 
    os.makedirs(TEMP_DIR)

# --- 2. AI 深度分析核心引擎 ---

def get_action_data(video_path):
    """AI 深度分析：运动特征提取"""
    mp_pose = mp.solutions.pose
    y_coords = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    with mp_pose.Pose(min_detection_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            y_coords.append(res.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y if res.pose_landmarks else np.nan)
    cap.release()
    
    arr = np.array(y_coords)
    mask = np.isnan(arr)
    if np.any(mask) and not np.all(mask):
        arr[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), arr[~mask])
    
    dy = np.abs(np.gradient(arr))
    moving_indices = np.where(dy > (np.max(dy) * 0.1))[0]
    start_f = moving_indices[0] if len(moving_indices) > 0 else 0
    end_f = moving_indices[-1] if len(moving_indices) > 0 else total_frames - 1
    return arr, np.linspace(start_f, end_f, 6).astype(int), (start_f, end_f), fps

def render_premium_video(video_path, y_data, swing_window, fps):
    """AI 深度分析：流媒体影像合成"""
    cap = cv2.VideoCapture(video_path)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw_out, final_out = os.path.join(TEMP_DIR, "raw_tmp.mp4"), os.path.join(TEMP_DIR, "video_final.mp4")
    out = cv2.VideoWriter(raw_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w + 400, h))
    
    for i in range(len(y_data)):
        ret, frame = cap.read()
        if not ret: break
        fig, ax = plt.subplots(figsize=(4, h/100), dpi=100)
        fig.patch.set_facecolor('#000000') # 绘图区背景黑色
        ax.plot(y_data, color='#D4AF37', linewidth=3) # 金色轨迹
        ax.axvline(x=i, color='#FFFFFF', linewidth=2)
        ax.axvspan(swing_window[0], swing_window[1], color='#D4AF37', alpha=0.15)
        ax.invert_yaxis()
        ax.axis('off')
        fig.canvas.draw()
        graph_img = cv2.cvtColor(np.array(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
        plt.close(fig)
        out.write(np.hstack((frame, cv2.resize(graph_img, (400, h)))))
    cap.release(); out.release()
    os.system(f'ffmpeg -y -i {raw_out} -vcodec libx264 -crf 28 {final_out}')
    return final_out

def get_pose_frame(video_path, frame_idx):
    """AI 深度分析：高光时刻对齐"""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read(); cap.release()
    if not ret: return None
    with mp.solutions.pose.Pose(static_image_mode=True) as pose:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if res.pose_landmarks:
            # 这里的连线在输出图中会自动绘制
            mp.solutions.drawing_utils.draw_landmarks(rgb, res.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
    return rgb

# --- 3. 尊享版页面结构 ---

with st.sidebar:
    st.title("🏆 GolfAsistant")
    st.markdown("尊享级 AI 深度分析对比系统")
    st.markdown("---")
    
    st.subheader("📽️ 影像素材库")
    u_file = st.file_uploader("学员练习视频 (High Quality)", type=["mp4", "mov"])
    p_file = st.file_uploader("职业对标视频 (PGA Pro)", type=["mp4", "mov"])
    st.markdown("---")
    
    analyze_btn = st.button("开启 AI 深度分析 ⚡")

if u_file and p_file:
    if analyze_btn:
        try:
            with st.status("正在启动 AI 深度分析引擎...", expanded=True) as status:
                u_p, p_p = os.path.join(TEMP_DIR, "u.mp4"), os.path.join(TEMP_DIR, "p.mp4")
                with open(u_p, "wb") as f: f.write(u_file.getbuffer())
                with open(p_p, "wb") as f: f.write(p_file.getbuffer())

                u_data, u_idx, u_win, u_fps = get_action_data(u_p)
                p_data, p_idx, p_win, p_fps = get_action_data(p_p)

                # 模块1: 动力学对齐指标
                c1, c2, c3 = st.columns(3)
                c1.metric("学员挥杆时长", f"{u_win[1]-u_win[0]} Frames")
                c2.metric("职业选手时长", f"{p_win[1]-p_win[0]} Frames")
                match = max(0, 100-abs((u_win[1]-u_win[0])-(p_win[1]-p_win[0])))
                c3.metric("AI 对齐匹配度", f"{match}%")

                # 模块2: 物理轨迹
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
                for i in range(6):
                    img_u, img_p = get_pose_frame(u_p, u_idx[i]), get_pose_frame(p_p, p_idx[i])
                    if img_u is not None and img_p is not None:
                        comb = np.hstack((cv2.resize(img_u, (350, 500)), cv2.resize(img_p, (350, 500))))
                        m_imgs.append(comb)
                r1, r2 = np.hstack(m_imgs[:3]), np.hstack(m_imgs[3:])
                full_m = np.vstack((r1, r2))
                st.image(full_m, use_container_width=True)
                buf_matrix = BytesIO(); Image.fromarray(full_m).save(buf_matrix, format="png")
                st.markdown('</div>', unsafe_allow_html=True)

                # 模块4: 动态分析
                st.markdown('<div class="report-box"><h3>📺 AI 动态追踪分析录影</h3>', unsafe_allow_html=True)
                v_path = render_premium_video(u_p, u_data, u_win, u_fps)
                st.video(v_path)
                st.markdown('</div>', unsafe_allow_html=True)
                
                status.update(label="✅ AI 深度分析报告就绪", state="complete")

            # 导出面板
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
