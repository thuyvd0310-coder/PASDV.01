# python.py

import streamlit as st
import pandas as pd
import numpy as np
from google import genai
from google.genai.errors import APIError

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Thẩm định Phương án Vốn Vay",
    layout="wide"
)

st.title("Ứng dụng Thẩm định Dự án Đầu tư (DCF) 💰")

# --- Khởi tạo và Cấu hình Gemini Client (Global) ---
try:
    API_KEY = st.secrets.get("GEMINI_API_KEY")
    if API_KEY:
        GEMINI_CLIENT = genai.Client(api_key=API_KEY)
    else:
        st.error("Lỗi: Không tìm thấy Khóa API 'GEMINI_API_KEY'. Vui lòng cấu hình Streamlit Secrets.")
        GEMINI_CLIENT = None
except Exception as e:
    st.error(f"Lỗi khởi tạo Gemini Client: {e}")
    GEMINI_CLIENT = None

# --- Hàm gọi API Gemini (Cho AI Insights và Chatbot) ---
def generate_ai_response(prompt_text):
    """Gửi prompt đến Gemini API và nhận nhận xét."""
    if GEMINI_CLIENT is None:
        return "Lỗi: Gemini API không được cấu hình. Vui lòng kiểm tra API Key."
    
    try:
        model_name = 'gemini-2.5-flash' 
        response = GEMINI_CLIENT.models.generate_content(
            model=model_name,
            contents=prompt_text
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"

# --- Hàm tính toán Dòng tiền và Chỉ số DCF ---
@st.cache_data
def calculate_dcf(
    total_investment, 
    n_years, 
    wacc, 
    annual_revenue, 
    annual_cost, 
    tax_rate
):
    """Tính toán FCF, NPV và IRR."""
    
    # Giả định đơn giản: Khấu hao = 0 để phù hợp với dữ liệu đầu vào (Lãi suất đã được xử lý trong WACC)
    EBIT = annual_revenue - annual_cost
    TAX = EBIT * tax_rate
    EAT = EBIT - TAX
    
    # FCF (Dòng tiền Tự do) = EAT + Khấu hao - Thay đổi NWC - Capex (Ở đây: FCF = EAT)
    FCF_yearly = EAT
    
    # Tạo Dòng tiền
    cash_flows = [-total_investment] + [FCF_yearly] * n_years
    
    # Tính NPV (Giá trị Hiện tại Thuần)
    # np.npv(rate, values)
    NPV = np.npv(wacc, cash_flows)
    
    # Tính IRR (Tỷ suất Sinh lời Nội tại)
    # np.irr(values)
    IRR = np.irr(cash_flows) if np.irr(cash_flows) != np.nan else 0

    return FCF_yearly, NPV, IRR, cash_flows

# --- Cấu hình Ứng dụng theo Module ---

# ----------------------------------------------------
# MODULE 1: NHẬP LIỆU DỰ ÁN
# ----------------------------------------------------
with st.expander("📝 1. Nhập Liệu Dự Án và Thông số Tài chính", expanded=True):
    col1, col2, col3 = st.columns(3)
    
    # Input Vốn
    TOTAL_INV = col1.number_input("Tổng Vốn Đầu tư (tỷ VNĐ)", value=30.0, min_value=1.0, step=1.0)
    INV_DEBT_RATIO = col2.slider("Tỷ lệ Vay Vốn (%)", value=80, min_value=0, max_value=100) / 100
    LTV_TSBD = col3.number_input("Giá trị Tài sản Đảm bảo (tỷ VNĐ)", value=70.0, min_value=1.0, step=1.0)

    col4, col5 = st.columns(2)
    # Input Tài chính
    WACC = col4.number_input("WACC của Doanh nghiệp (%)", value=13.0, min_value=1.0, step=0.1) / 100
    TAX_RATE = col5.number_input("Thuế suất TNDN (%)", value=20.0, min_value=1.0, step=1.0) / 100
    
    # Input Dòng tiền
    st.subheader("Dự kiến Dòng tiền Hoạt động Hàng năm")
    col6, col7, col8 = st.columns(3)
    ANNUAL_REV = col6.number_input("Doanh thu Hàng năm (tỷ VNĐ)", value=3.5, min_value=0.1, step=0.1)
    ANNUAL_COST = col7.number_input("Chi phí Hàng năm (tỷ VNĐ)", value=2.0, min_value=0.1, step=0.1)
    N_YEARS = col8.number_input("Vòng đời Dự án (năm)", value=10, min_value=1, step=1)
    
    # Tính toán cơ bản
    VAY_VON = TOTAL_INV * INV_DEBT_RATIO
    VON_TU_CO = TOTAL_INV * (1 - INV_DEBT_RATIO)

# --- Tính toán DCF ---
try:
    FCF, NPV, IRR, cash_flows_full = calculate_dcf(
        TOTAL_INV, N_YEARS, WACC, ANNUAL_REV, ANNUAL_COST, TAX_RATE
    )
    
    # ----------------------------------------------------
    # MODULE 2: KẾT QUẢ VÀ CHỈ SỐ DCF
    # ----------------------------------------------------
    st.header("📈 2. Hiệu quả Tài chính và Khả năng Trả nợ")
    
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    col_k1.metric("Vốn Vay Dự kiến", f"{VAY_VON:,.0f} tỷ VNĐ")
    col_k2.metric("Lợi nhuận Sau Thuế/năm (FCF)", f"{FCF:,.2f} tỷ VNĐ")
    col_k3.metric("NPV (Giá trị Hiện tại Thuần)", f"{NPV:,.2f} tỷ VNĐ", delta="Đạt" if NPV > 0 else "Không đạt")
    col_k4.metric("IRR (Tỷ suất Sinh lời)", f"{IRR*100:,.2f}%", delta="> WACC" if IRR > WACC else "< WACC")
    
    # ----------------------------------------------------
    # MODULE 3: PHÂN TÍCH ĐỘ NHẠY VÀ RỦI RO
    # ----------------------------------------------------
    st.subheader("Phân tích Độ nhạy (Kịch bản Xấu nhất)")
    
    # Kịch bản Xấu nhất: Doanh thu giảm 15%, Chi phí tăng 10%
    DT_WORST = ANNUAL_REV * 0.85
    CP_WORST = ANNUAL_COST * 1.10
    
    FCF_W, NPV_W, IRR_W, _ = calculate_dcf(
        TOTAL_INV, N_YEARS, WACC, DT_WORST, CP_WORST, TAX_RATE
    )
    
    col_r1, col_r2 = st.columns(2)
    col_r1.metric("NPV (Kịch bản Xấu nhất)", f"{NPV_W:,.2f} tỷ VNĐ", delta="Vẫn dương" if NPV_W > 0 else "Đã âm")
    col_r2.metric("LTV (Cho vay/TSBĐ)", f"{(VAY_VON / LTV_TSBD) * 100:,.2f}%", delta="Rất an toàn")

    # ----------------------------------------------------
    # MODULE 4: AI INSIGHTS - NHẬN ĐỊNH CHUYÊN SÂU
    # ----------------------------------------------------
    st.header("🧠 3. AI Insights - Nhận định Chuyên sâu")
    
    if st.button("Tạo Báo cáo Thẩm định AI"):
        
        # Tạo prompt chi tiết dựa trên các kết quả
        prompt_ai = f"""
        Bạn là một chuyên gia thẩm định tài chính cấp cao. Hãy đưa ra nhận định chuyên sâu (khoảng 4-5 đoạn) về phương án đầu tư dây chuyền bánh mì này. 
        Tập trung vào 3 khía cạnh: Hiệu quả tài chính, Rủi ro (Độ nhạy), và Khả năng đảm bảo nợ cho ngân hàng.

        Dữ liệu đầu vào:
        - Tổng Vốn: {TOTAL_INV} tỷ VNĐ | Vốn Vay: {VAY_VON} tỷ VNĐ | WACC: {WACC*100}% | Thuế: {TAX_RATE*100}% | TSBĐ: {LTV_TSBD} tỷ VNĐ
        - FCF Hàng năm: {FCF:.2f} tỷ VNĐ | NPV Cơ sở: {NPV:.2f} tỷ VNĐ | IRR Cơ sở: {IRR*100:.2f}%

        Phân tích rủi ro (Kịch bản Xấu nhất - Doanh thu -15%, Chi phí +10%):
        - NPV Kịch bản Xấu nhất: {NPV_W:.2f} tỷ VNĐ
        - LTV (Loan-to-Value): {(VAY_VON / LTV_TSBD) * 100:.2f}%

        Hãy đánh giá mức độ chấp nhận rủi ro và đưa ra kết luận về việc cấp vốn.
        """
        
        with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
            ai_result = generate_ai_response(prompt_ai)
            st.markdown("**Kết quả Phân tích từ Gemini AI:**")
            st.info(ai_result)

    # ----------------------------------------------------
    # MODULE 5: KHUNG HỎI - ĐÁP CHUYÊN GIA
    # ----------------------------------------------------
    st.header("💬 4. Hỏi - Đáp Chuyên gia với Gemini")
    
    # 1. Khởi tạo Lịch sử Hội thoại
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        if GEMINI_CLIENT:
            st.session_state.chat_history.append({"role": "assistant", "content": "Xin chào! Tôi là chuyên gia thẩm định AI. Hãy hỏi tôi về tính toán NPV, IRR, hoặc các rủi ro của Phương án sử dụng vốn này."})

    # Khung tải tệp (Chỉ để bổ sung bối cảnh)
    uploaded_file = st.file_uploader(
        "📎 Tải thêm tệp (PDF/Excel) để bổ sung bối cảnh phân tích:", 
        type=["pdf", "xlsx", "csv"], 
        key="chat_file_uploader"
    )

    # 2. Hiển thị Lịch sử Hội thoại
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # 3. Xử lý Đầu vào (Input) của người dùng
    if prompt := st.chat_input("Nhập câu hỏi của bạn về dự án này..."):
        
        if GEMINI_CLIENT is None:
            st.error("Lỗi: Không thể khởi tạo Chatbot do thiếu Khóa API.")
        else:
            # Lưu và hiển thị câu hỏi của người dùng
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Chuẩn bị bối cảnh (contextual prompt)
            context = f"Context Dự án: Tổng Vốn {TOTAL_INV} tỷ VNĐ, NPV: {NPV:.2f} tỷ VNĐ, IRR: {IRR*100:.2f}%. | "
            if uploaded_file is not None:
                context += f"Người dùng đã tải tệp: {uploaded_file.name}. Vui lòng tham khảo bối cảnh này."
            
            full_prompt = (
                f"Bạn là chuyên gia thẩm định, hãy trả lời câu hỏi sau của người dùng, sử dụng bối cảnh dự án sau đây:\n\n"
                f"{context}\n\n"
                f"Câu hỏi: {prompt}"
            )
            
            with st.spinner("Gemini đang phân tích..."):
                ai_response = generate_ai_response(full_prompt)
            
            # Lưu và hiển thị phản hồi của AI
            with st.chat_message("assistant"):
                st.markdown(ai_response)
            st.session_state.chat_history.append({"role": "assistant", "content": ai_response})

except NameError:
    st.error("Vui lòng kiểm tra lại các giá trị đầu vào.")
except Exception as e:
    st.error(f"Đã xảy ra lỗi không xác định trong quá trình tính toán: {e}")
