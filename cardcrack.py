# cardcrack.py
# 콘크리트 균열 자동 진단 V8.0 - 모바일 친화 원스텝
# streamlit-webrtc로 카메라 스트림에 가이드박스를 직접 오버레이
# 촬영 버튼 한 번에 바로 분석까지 자동 진행
 
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import threading
 
st.set_page_config(page_title="균열 자동 진단 V8", layout="wide")
st.title("🔍 콘크리트 균열 자동 진단 V8")
st.caption("💳 카드를 가이드박스에 맞춰 한 번에 촬영 → 자동 분석")
 
# ════════════════════════════════════════════════════════════════
# 상수
# ════════════════════════════════════════════════════════════════
CARD_W_MM = 85.60
CARD_H_MM = 53.98
CARD_ASPECT = CARD_W_MM / CARD_H_MM
GUIDE_RATIO = 0.40  # 가이드박스가 차지하는 화면 너비 비율
 
# ════════════════════════════════════════════════════════════════
# YOLO 모델
# ════════════════════════════════════════════════════════════════
@st.cache_resource
def load_yolo():
    return YOLO("bestcrack.pt")
 
# ════════════════════════════════════════════════════════════════
# 카드 자동 검출
# ════════════════════════════════════════════════════════════════
def detect_card(img_np):
    H, W = img_np.shape[:2]
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    edged = cv2.dilate(edged, kernel, iterations=2)
 
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
 
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    best_card = None
    best_score = 0
 
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (W * H * 0.02) or area > (W * H * 0.8):
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), angle = rect
        if w == 0 or h == 0:
            continue
        long_side = max(w, h)
        short_side = min(w, h)
        aspect = long_side / short_side
        aspect_diff = abs(aspect - CARD_ASPECT)
        if aspect_diff > 0.25:
            continue
        score = area / (1 + aspect_diff * 10)
        if score > best_score:
            best_score = score
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            best_card = {
                "long_px": long_side,
                "short_px": short_side,
                "center": (cx, cy),
                "box": box,
                "angle": angle
            }
    return best_card
 
# ════════════════════════════════════════════════════════════════
# 가이드박스 오버레이 그리기
# ════════════════════════════════════════════════════════════════
def draw_guide_box(frame):
    """매 프레임에 가이드박스 그리기"""
    H, W = frame.shape[:2]
    cx, cy = W // 2, H // 2
 
    # 가이드박스 크기 (가로 기준)
    box_w = int(W * GUIDE_RATIO)
    box_h = int(box_w / CARD_ASPECT)
 
    x1, y1 = cx - box_w // 2, cy - box_h // 2
    x2, y2 = cx + box_w // 2, cy + box_h // 2
 
    # 반투명 어두운 배경 (가이드박스 외부)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, H), (0, 0, 0), -1)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
    # 가이드박스 안은 다시 원본으로 (마스크 방식)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
 
    # 점선 박스 (대시 4개씩)
    color = (102, 255, 102)  # 녹색
    dash_len = 20
    gap = 10
    for i in range(x1, x2, dash_len + gap):
        cv2.line(frame, (i, y1), (min(i + dash_len, x2), y1), color, 3)
        cv2.line(frame, (i, y2), (min(i + dash_len, x2), y2), color, 3)
    for i in range(y1, y2, dash_len + gap):
        cv2.line(frame, (x1, i), (x1, min(i + dash_len, y2)), color, 3)
        cv2.line(frame, (x2, i), (x2, min(i + dash_len, y2)), color, 3)
 
    # 네 모서리 강조
    corner_len = 25
    corner_thick = 6
    # 좌상
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, corner_thick)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, corner_thick)
    # 우상
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, corner_thick)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, corner_thick)
    # 좌하
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, corner_thick)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, corner_thick)
    # 우하
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, corner_thick)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, corner_thick)
 
    # 중앙 십자 (빨강)
    cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (255, 0, 0), 3)
    cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (255, 0, 0), 3)
 
    # 상단 안내 텍스트 배경
    text = "Place card in green box"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 0.7, 2)
    text_y = max(y1 - 15, th + 10)
    cv2.rectangle(frame, (cx - tw // 2 - 8, text_y - th - 8),
                  (cx + tw // 2 + 8, text_y + 8), color, -1)
    cv2.putText(frame, text, (cx - tw // 2, text_y),
                font, 0.7, (0, 0, 0), 2)
 
    return frame
 
# ════════════════════════════════════════════════════════════════
# 비디오 프로세서 (프레임마다 가이드박스 오버레이 + 최신 프레임 저장)
# ════════════════════════════════════════════════════════════════
class VideoProcessor:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_frame = None  # 가이드박스 없는 원본 프레임
 
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        # 원본 프레임 저장 (오버레이 없는 깨끗한 프레임)
        with self.lock:
            self.latest_frame = img.copy()
        # 가이드박스를 그려서 화면에는 표시
        img_with_guide = draw_guide_box(img)
        return av.VideoFrame.from_ndarray(img_with_guide, format="bgr24")
 
# ════════════════════════════════════════════════════════════════
# 사이드바
# ════════════════════════════════════════════════════════════════
st.sidebar.header("⚙️ 옵션")
conf_thres = st.sidebar.slider("YOLO 신뢰도", 0.05, 0.9, 0.25, 0.05)
st.sidebar.markdown("---")
st.sidebar.markdown("""
**📋 사용법**
1. 카메라 권한 허용
2. 균열 위에 카드 놓기
3. 카드를 녹색 박스에 맞춤
4. **📸 촬영 및 분석** 클릭
5. 자동으로 결과 표시
""")
 
# 세션 상태
if "captured_img" not in st.session_state:
    st.session_state.captured_img = None
if "analyze" not in st.session_state:
    st.session_state.analyze = False
 
# ════════════════════════════════════════════════════════════════
# 1. 라이브 카메라
# ════════════════════════════════════════════════════════════════
st.markdown("### 📷 카메라")
st.markdown("**카드를 균열 위에 올리고, 녹색 가이드박스에 맞춰주세요.**")
 
RTC_CONFIG = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})
 
ctx = webrtc_streamer(
    key="cardcrack",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIG,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": {"facingMode": {"ideal": "environment"}},
        "audio": False
    },
    async_processing=True,
)
 
# 촬영 버튼
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    capture = st.button(
        "📸 촬영 및 분석",
        use_container_width=True,
        type="primary",
        disabled=(ctx.video_processor is None)
    )
 
if capture and ctx.video_processor is not None:
    with ctx.video_processor.lock:
        if ctx.video_processor.latest_frame is not None:
            # BGR → RGB로 변환해서 저장
            frame_rgb = cv2.cvtColor(ctx.video_processor.latest_frame, cv2.COLOR_BGR2RGB)
            st.session_state.captured_img = frame_rgb
            st.session_state.analyze = True
 
# ════════════════════════════════════════════════════════════════
# 파일 업로드 대안 (PC 사용자 또는 카메라 안 되는 경우)
# ════════════════════════════════════════════════════════════════
with st.expander("📁 또는 파일 업로드로 분석"):
    img_upload = st.file_uploader(
        "카드와 균열이 함께 있는 사진",
        type=["jpg", "jpeg", "png"]
    )
    if img_upload is not None:
        try:
            pil_img = Image.open(img_upload).convert("RGB")
            st.session_state.captured_img = np.array(pil_img)
            st.session_state.analyze = True
        except Exception as e:
            st.error(f"❌ 이미지 로드 실패: {e}")
 
# ════════════════════════════════════════════════════════════════
# 2. 분석 (촬영되면 자동 실행)
# ════════════════════════════════════════════════════════════════
if not st.session_state.analyze or st.session_state.captured_img is None:
    st.info("👆 카메라 권한을 허용하고, 카드를 가이드박스에 맞춘 뒤 **📸 촬영 및 분석** 버튼을 누르세요.")
    st.stop()
 
img_np = st.session_state.captured_img
H, W = img_np.shape[:2]
 
st.markdown("---")
st.markdown("### 🎯 분석 결과")
 
# 다시 찍기 버튼
if st.button("🔁 다시 촬영"):
    st.session_state.captured_img = None
    st.session_state.analyze = False
    st.rerun()
 
# 카드 검출
with st.spinner("🔍 카드 자동 검출 중..."):
    card_info = detect_card(img_np)
 
if card_info is None:
    st.error(
        "❌ **카드를 찾지 못했습니다.**\n\n"
        "• 카드가 사진에 명확히 보이는지\n"
        "• 카드와 배경의 색 대비가 충분한지\n"
        "• 카드가 너무 기울어지지 않았는지\n"
        "• 조명이 균일한지\n\n"
        "위 조건을 확인 후 다시 촬영해주세요."
    )
    st.image(img_np, caption="촬영된 사진", use_container_width=True)
    st.stop()
 
# mm/pixel 계산
scale = CARD_W_MM / card_info["long_px"]
 
# 카드 검출 시각화
img_with_card = img_np.copy()
cv2.drawContours(img_with_card, [card_info["box"]], 0, (0, 255, 0), 5)
 
col1, col2 = st.columns(2)
with col1:
    st.markdown("**📷 카드 검출**")
    st.image(img_with_card, use_container_width=True)
with col2:
    st.markdown("**📐 측정 기준**")
    st.write(f"📏 카드 픽셀: **{card_info['long_px']:.0f} × {card_info['short_px']:.0f} px**")
    st.write(f"🔬 1 픽셀 = **{scale:.4f} mm**")
    st.write(f"🖼️ 이미지: **{W} × {H} px**")
    st.success("✅ 카드 자동 검출 성공")
 
# 균열 검출
with st.spinner("🔍 균열 탐지 중..."):
    yolo = load_yolo()
    results = yolo.predict(img_np, conf=conf_thres, verbose=False)
 
if not results or results[0].masks is None:
    st.error("❌ 균열을 찾지 못했습니다. 사이드바에서 신뢰도를 낮춰보세요.")
    st.stop()
 
masks = results[0].masks.data.cpu().numpy()
full_mask = np.zeros((H, W), dtype=np.uint8)
for m in masks:
    mr = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    full_mask = np.maximum(full_mask, (mr > 0.5).astype(np.uint8))
 
# 카드 영역은 균열에서 제외
card_mask = np.zeros((H, W), dtype=np.uint8)
cv2.fillPoly(card_mask, [card_info["box"]], 1)
full_mask = full_mask * (1 - card_mask)
 
if full_mask.sum() == 0:
    st.warning("⚠️ 균열 마스크가 비어있습니다.")
    st.stop()
 
# 측정
pixel_cnt = int(full_mask.sum())
area_cm2 = (pixel_cnt * scale * scale) / 100.0
dt = cv2.distanceTransform(full_mask, cv2.DIST_L2, 5)
max_width_mm = 2 * float(dt.max()) * scale
 
c1, c2, c3 = st.columns(3)
c1.metric("📏 mm/pixel", f"{scale:.4f}")
c2.metric("📐 균열 면적", f"{area_cm2:.2f} cm²")
c3.metric("📏 최대 균열 폭", f"{max_width_mm:.2f} mm")
 
# 시각화
overlay = img_np.copy()
overlay[full_mask > 0] = [255, 50, 50]
blended = cv2.addWeighted(img_np, 0.55, overlay, 0.45, 0)
cv2.drawContours(blended, [card_info["box"]], 0, (0, 255, 0), 4)
 
st.image(blended, caption="🎯 검출 결과 (녹색: 카드 / 빨강: 균열)", use_container_width=True)
 
st.success(
    f"✅ 측정 완료 — 카드 기반 자동 보정\n\n"
    f"📊 총 균열 픽셀: {pixel_cnt:,}개 | 1px = {scale:.4f}mm"
)