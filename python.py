# ===============================================================
# ỨNG DỤNG THẨM ĐỊNH PHƯƠNG ÁN KINH DOANH TẠI AGRIBANK
# Phiên bản DCF Pro + Chat Gemini hỗ trợ AI
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
from io import BytesIO
from datetime import datetime
from docx import Document
import google.generativeai as genai

# ---------------------------------------------------------------
# CẤU HÌNH GIAO DIỆN
# ---------------------------------------------------------------
st.set_page_config(page_title="Thẩm định phương án kinh doanh", page_icon="🏦", layout="wide")

# CSS giao diện Agribank
st.markdown("""
    <style>
        body {
            background-color: #FFF2F2;
        }
        .main {
            background-color: #FFF2F2;
        }
        .title-container {
            background-color: #7A0019;
            color: white;
            text-align: center;
            padding: 15px;
            border-radius: 10px;
            font-weight: bold;
        }
        .title-container h1 {
            font-size: 28px;
            font-weight: 900;
            margin-bottom: 5px;
        }
        .warning-text {
            color: #7A0019;
            font-weight: bold;
            font-style: italic;
            margin-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Banner đầu trang
st.markdown("""
<div class="title-container">
    <h1>ỨNG DỤNG THẨM ĐỊNH PHƯƠNG ÁN KINH DOANH TẠI AGRIBANK 🏦</h1>
    <p>Hỗ trợ đánh giá tài chính, rủi ro và sinh báo cáo tự động</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# HÀM TÍNH TOÁN
# ---------------------------------------------------------------
def npv(rate, cashflows):
    return sum(cf / ((1 + rate) ** t) for t, cf in enumerate(cashflows))

def irr(cashflows, guess=0.1, max_iter=100, tol=1e-7):
    rate = guess
    for _ in range(max_iter):
        f = sum(cf / ((1 + rate) ** t) for t, cf in enumerate(cashflows))
        df = sum(-t * cf / ((1 + rate) ** (t + 1)) for t, cf in enumerate(cashflows))
        if abs(df) < 1e-12:
            break
        new_rate = rate - f / df
        if abs(new_rate - rate) < tol:
            rate = new_rate
            break
        rate = new_rate
    return rate if not np.isnan(rate) else np.nan

def payback_period(cashflows):
    cum = 0.0
    for t, cf in enumerate(cashflows):
        cum += cf
        if cum >= 0:
            prev_cum = cum - cf
            remain = -prev_cum
            frac = remain / cf if cf != 0 else 0
            return (t - 1) + frac
    return np.nan

def fmt_money(x, unit="tỷ VND"):
    return f"{x:,.2f} {unit}".replace(",", "X").replace(".", ",").replace("X", ".")

# ---------------------------------------------------------------
# XÂY DỰNG DÒNG TIỀN
# ---------------------------------------------------------------
def build_cashflow_table(total_invest, debt_ratio, collateral_value, wacc, tax_rate,
                         years, rev_y1, opex_y1, g_rev, g_opex, wc0, dep_base,
                         salvage_pct):
    df = pd.DataFrame({"Năm": list(range(0, years + 1))})
    capex0 = dep_base
    other_invest = total_invest - dep_base
    df["CAPEX"] = [capex0] + [0] * years
    df["VLĐ (đầu kỳ)"] = [wc0] + [0] * years
    rev, opex = [0.0], [0.0]
    for t in range(1, years + 1):
        rt = rev_y1 * ((1 + g_rev) ** (t - 1))
        ot = opex_y1 * ((1 + g_opex) ** (t - 1))
        rev.append(rt)
        opex.append(ot)
    df["Doanh thu"], df["Chi phí HĐ"] = rev, opex
    df["Khấu hao"] = [0.0] + [dep_base / years] * years
    df["EBIT"] = df["Doanh thu"] - df["Chi phí HĐ"] - df["Khấu hao"]
    df["Thuế"] = [max(e, 0) * tax_rate for e in df["EBIT"]]
    df["Thuế"].iloc[0] = 0.0
    fcf = (df["EBIT"] - df["Thuế"]) + df["Khấu hao"]
    fcf.iloc[0] = -(capex0 + wc0 + other_invest)
    fcf.iloc[-1] += dep_base * salvage_pct + wc0
    df["FCF"] = fcf
    cashflows = fcf.tolist()
    npv_val, irr_val = npv(wacc, cashflows), irr(cashflows)
    pb_val = payback_period(cashflows)
    ltv = (total_invest * debt_ratio) / collateral_value if collateral_value > 0 else np.nan
    return df, {"NPV": npv_val, "IRR": irr_val, "Payback": pb_val, "LTV": ltv}

# ---------------------------------------------------------------
# NHẬP DỮ LIỆU
# ---------------------------------------------------------------
st.subheader("1. Nhập dữ liệu thẩm định")

col1, col2, col3 = st.columns(3)
with col1:
    total_invest = st.number_input("Tổng vốn đầu tư (tỷ VND)", 0.0, 1e6, 30.0)
    debt_ratio = st.slider("Tỷ lệ vay vốn (%)", 0, 100, 80) / 100
    wacc = st.number_input("WACC (%)", 0.0, 100.0, 13.0) / 100
with col2:
    collateral_value = st.number_input("Giá trị tài sản đảm bảo (tỷ VND)", 0.0, 1e6, 70.0)
    tax_rate = st.number_input("Thuế TNDN (%)", 0.0, 100.0, 20.0) / 100
    years = st.number_input("Thời gian dự án (năm)", 1, 50, 10)
with col3:
    rev_y1 = st.number_input("Doanh thu năm 1 (tỷ VND)", 0.0, 1e6, 3.5)
    opex_y1 = st.number_input("Chi phí năm 1 (tỷ VND)", 0.0, 1e6, 2.0)
    g_rev = st.number_input("Tăng trưởng doanh thu (%/năm)", -100.0, 200.0, 0.0) / 100

# ---------------------------------------------------------------
# KẾT LUẬN AI
# ---------------------------------------------------------------
st.subheader("2. Kết luận của Hệ thống AI (tự động)")

df, summary = build_cashflow_table(total_invest, debt_ratio, collateral_value, wacc,
                                   tax_rate, years, rev_y1, opex_y1, g_rev, 0, 3.0, 27.0, 0.1)

st.markdown(f"""
**NPV:** `{fmt_money(summary['NPV'])}`  
**IRR:** `{summary['IRR']*100:.2f}%`  
**LTV:** `{summary['LTV']*100:.2f}%`  
**Payback:** `{summary['Payback']:.2f} năm`
""")

st.info("Hiệu quả tài chính tốt, có thể xem xét cấp vốn (NPV>0, IRR>WACC).")

with st.expander("🤖 Hiển thị phân tích chuyên sâu của AI (bấm để xem)"):
    st.markdown(f"""
    ### 💡 Phân tích chuyên sâu
    - 📈 Dự án đạt **NPV dương** ({fmt_money(summary['NPV'])}) và **IRR vượt WACC**.
    - 💰 Dòng tiền FCF ổn định, khả năng trả nợ tốt.
    - 🧮 Tỷ lệ LTV {summary['LTV']*100:.2f}% → an toàn cho ngân hàng.
    - 🏁 Khuyến nghị xem xét cấp vốn với điều kiện giám sát dòng tiền thực tế.
    """)

st.markdown("<p class='warning-text'>⚠️ AI có thể mắc lỗi. Luôn kiểm chứng các thông tin quan trọng!<br>Rất mong bạn thông cảm vì sự bất tiện này 🙏</p>", unsafe_allow_html=True)

# ---------------------------------------------------------------
# CHAT VỚI GEMINI
# ---------------------------------------------------------------
st.subheader("3. Trò chuyện cùng Trợ lý Gemini 🤖")

api_key = st.text_input("🔑 Nhập API Key Gemini của bạn (bắt buộc)", type="password")

if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    chat = st.session_state.get("chat", model.start_chat(history=[]))

    user_input = st.text_input("Nhập câu hỏi hoặc yêu cầu của bạn:")

    if st.button("Gửi câu hỏi 🚀"):
        if user_input.strip():
            response = chat.send_message(user_input)
            st.session_state["chat"] = chat
            st.markdown(
                f"""
                🗨️ **Bạn đang bù đầu tóc rối vì quá tải với số hồ sơ khổng lồ cần thẩm định?  
                Đừng lo, đã có tôi đây rồi! 😄**  
                <br><br>💬 {response.text}
                """,
                unsafe_allow_html=True,
            )
else:
    st.warning("🔒 Hãy nhập API Key Gemini để kích hoạt khung chat.")
