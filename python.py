# app.py
# DCF Pro - Tham dinh du an dau tu (Tieng Viet) - 3 kich ban
# Tac gia: ChatGPT (thiet ke cho Shelbi)
# Yeu cau: streamlit, numpy, pandas, matplotlib (tich hop san trong Streamlit Cloud)

import streamlit as st
import numpy as np
import pandas as pd
from io import BytesIO
import base64
import math

st.set_page_config(page_title="Thẩm định Dự án Đầu tư (DCF Pro)", page_icon="💰", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
def npv(rate, cashflows):
    # cashflows: list, CF0 .. CFn
    return sum(cf / ((1 + rate) ** t) for t, cf in enumerate(cashflows))

def irr(cashflows, guess=0.1, max_iter=100, tol=1e-7):
    # Newton-Raphson
    rate = guess
    for _ in range(max_iter):
        # NPV
        f = sum(cf / ((1 + rate) ** t) for t, cf in enumerate(cashflows))
        # dNPV/dr
        df = sum(-t * cf / ((1 + rate) ** (t + 1)) for t, cf in enumerate(cashflows))
        if abs(df) < 1e-12:
            break
        new_rate = rate - f / df
        if abs(new_rate - rate) < tol:
            rate = new_rate
            break
        rate = new_rate
    return rate if not (math.isnan(rate) or math.isinf(rate)) else np.nan

def payback_period(cashflows):
    # CF0 am. Tra ve nam + phan nam khi tong CF tich luy >= 0
    cum = 0.0
    for t, cf in enumerate(cashflows):
        cum += cf
        if cum >= 0:
            if t == 0:
                return 0.0
            # noi suy tuyen tinh trong nam t
            prev_cum = cum - cf
            remain = -prev_cum
            frac = remain / cf if cf != 0 else np.nan
            return (t - 1) + frac
    return np.nan  # khong hoan von

def fmt_money(x, unit="tỷ VND"):
    return f"{x:,.2f} {unit}".replace(",", "X").replace(".", ",").replace("X", ".")

def build_cashflow_table(
    total_invest,
    debt_ratio,
    collateral_value,
    wacc,
    tax_rate,
    years,
    rev_y1,
    opex_y1,
    g_rev,
    g_opex,
    wc0,
    dep_base,
    salvage_pct,
    scenario_name="Base",
    booster_rev=0.0,
    booster_opex=0.0,
):
    """
    - dep_base: co so khau hao (ty VND) (vi du: TSCĐ = Tong đầu tu - VLĐ)
    - salvage_pct: % gia tri thanh ly TSCĐ cuoi ky (0..1)
    - booster_rev/opex: dieu chinh theo kich ban (vi du +0.15, -0.10)
    """

    # Khoi tao bang
    years_idx = list(range(0, years + 1))  # 0..N
    df = pd.DataFrame({"Năm": years_idx})

    # Nam 0: giai doan dau tu
    capex0 = dep_base  # gia su capex ~ co so khau hao
    other_invest = total_invest - dep_base  # phan con lai (neu co)
    # VLĐ ban đầu ghi nhận tách bạch
    df["CAPEX"] = [capex0] + [0] * years
    df["VLĐ (đầu kỳ)"] = [wc0] + [0] * years

    # Doanh thu & Opex
    rev = [0.0]
    opex = [0.0]
    for t in range(1, years + 1):
        rt = rev_y1 * ((1 + g_rev) ** (t - 1))
        ot = opex_y1 * ((1 + g_opex) ** (t - 1))
        # booster theo kich ban
        rt *= (1 + booster_rev)
        ot *= (1 + booster_opex)
        rev.append(rt)
        opex.append(ot)
    df["Doanh thu"] = rev
    df["Chi phí HĐ"] = opex

    # Khau hao tu dep_base theo duong thang
    dep = [0.0] + [dep_base / years] * years
    df["Khấu hao"] = dep

    # EBIT = Rev - Opex - Dep
    ebit = [rev[i] - opex[i] - dep[i] for i in range(years + 1)]
    df["EBIT"] = ebit

    # Thuế TNDN (chi theo tiền mặt) = max(EBIT, 0) * tax_rate
    tax_cash = [max(e, 0) * tax_rate for e in ebit]
    tax_cash[0] = 0.0
    df["Thuế (tiền mặt)"] = tax_cash

    # FCF = (EBIT - Thuế) + Khấu hao  (bo qua ΔVLĐ hàng năm để đơn giản)
    fcf = [(ebit[i] - tax_cash[i]) + dep[i] for i in range(years + 1)]
    # Nam 0: outflow capex + VLĐ
    fcf[0] = -(capex0 + wc0 + other_invest)

    # Cuoi ky: thu hoi VLĐ + thanh ly TSCĐ
    salvage_value = dep_base * salvage_pct
    fcf[-1] += salvage_value + wc0

    df["FCF"] = fcf

    # Tong hop chi so
    cashflows = fcf
    project_npv = npv(wacc, cashflows)
    project_irr = irr(cashflows)
    pp = payback_period(cashflows)
    loan_amount = total_invest * debt_ratio
    ltv = loan_amount / collateral_value if collateral_value > 0 else np.nan

    kq = {
        "Kịch bản": scenario_name,
        "NPV": project_npv,
        "IRR": project_irr,
        "Payback (năm)": pp,
        "Vay dự kiến": loan_amount,
        "LTV": ltv,
    }
    return df, kq, cashflows


# -----------------------------
# Sidebar - Cau hinh kich ban
# -----------------------------
with st.sidebar:
    st.header("⚙️ Cấu hình kịch bản")
    sb_best_rev = st.slider("Tăng Doanh thu kịch bản Tốt (%)", 0, 50, 15) / 100
    sb_best_opex = -st.slider("Giảm Chi phí kịch bản Tốt (%)", 0, 50, 10) / 100
    sb_worst_rev = -st.slider("Giảm Doanh thu kịch bản Xấu (%)", 0, 50, 15) / 100
    sb_worst_opex = st.slider("Tăng Chi phí kịch bản Xấu (%)", 0, 50, 10) / 100

st.title("Ứng dụng Thẩm định Dự án Đầu tư (DCF) 💰")
st.markdown("**Phiên bản DCF Pro – 3 kịch bản & báo cáo tự động**")

# -----------------------------
# 1. Nhap lieu
# -----------------------------
st.subheader("1. Nhập Liệu Dự Án và Thông số tài chính")

col1, col2, col3 = st.columns(3)
with col1:
    total_invest = st.number_input("Tổng Vốn Đầu tư (tỷ VND)", 0.0, 1e6, 30.0, step=0.5, format="%.2f")
    debt_ratio = st.slider("Tỷ lệ Vay Vốn (%)", 0, 100, 80) / 100
    wacc = st.number_input("WACC của Doanh nghiệp (%)", 0.0, 100.0, 13.0, step=0.25, format="%.2f") / 100

with col2:
    collateral_value = st.number_input("Giá trị Tài sản Đảm bảo (tỷ VND)", 0.0, 1e6, 70.0, step=0.5, format="%.2f")
    tax_rate = st.number_input("Thuế suất TNDN (%)", 0.0, 100.0, 20.0, step=0.5, format="%.2f") / 100
    years = st.number_input("Vòng đời Dự án (năm)", 1, 50, 10, step=1)

with col3:
    rev_y1 = st.number_input("Doanh thu Năm 1 (tỷ VND)", 0.0, 1e6, 3.50, step=0.1, format="%.2f")
    opex_y1 = st.number_input("Chi phí Năm 1 (tỷ VND)", 0.0, 1e6, 2.00, step=0.1, format="%.2f")
    g_rev = st.number_input("Tăng trưởng Doanh thu (%/năm)", -100.0, 200.0, 0.0, step=1.0, format="%.2f") / 100

col4, col5, col6 = st.columns(3)
with col4:
    g_opex = st.number_input("Tăng trưởng Chi phí (%/năm)", -100.0, 200.0, 0.0, step=1.0, format="%.2f") / 100
with col5:
    wc0 = st.number_input("Vốn lưu động ban đầu (tỷ VND)", 0.0, 1e6, 3.00, step=0.5, format="%.2f")
    salvage_pct = st.number_input("Tỷ lệ thanh lý TSCĐ cuối kỳ (%)", 0.0, 100.0, 10.0, step=1.0, format="%.2f") / 100
with col6:
    dep_base = st.number_input("Cơ sở Khấu hao (TSCĐ, tỷ VND)", 0.0, 1e6, max(0.0, total_invest - wc0), step=0.5, format="%.2f")

st.markdown("---")

# -----------------------------
# 2. Tinh toan 3 kich ban
# -----------------------------
st.subheader("2. Hiệu quả Tài chính và Khả năng Trả nợ")

scenarios = [
    ("Cơ sở", 0.0, 0.0),
    ("Tốt", sb_best_rev, sb_best_opex),
    ("Xấu", sb_worst_rev, sb_worst_opex),
]

tables = []
summaries = []
cashflow_sets = {}

for name, b_rev, b_op in scenarios:
    df_cf, kq, cfs = build_cashflow_table(
        total_invest=total_invest,
        debt_ratio=debt_ratio,
        collateral_value=collateral_value,
        wacc=wacc,
        tax_rate=tax_rate,
        years=int(years),
        rev_y1=rev_y1,
        opex_y1=opex_y1,
        g_rev=g_rev,
        g_opex=g_opex,
        wc0=wc0,
        dep_base=dep_base,
        salvage_pct=salvage_pct,
        scenario_name=name,
        booster_rev=b_rev,
        booster_opex=b_op,
    )
    tables.append((name, df_cf))
    summaries.append(kq)
    cashflow_sets[name] = cfs

summary_df = pd.DataFrame(summaries)
# Hien thi cac chi so chinh
mcol1, mcol2, mcol3, mcol4 = st.columns(4)
with mcol1:
    st.metric("Vốn Vay dự kiến", fmt_money(summary_df.loc[0, "Vay dự kiến"]))
with mcol2:
    st.metric("LTV (Vay/TSBĐ)", f"{summary_df.loc[0, 'LTV']*100:,.2f}%")
with mcol3:
    st.metric("NPV (Cơ sở)", fmt_money(summary_df.loc[0, "NPV"]))
with mcol4:
    irrv = summary_df.loc[0, "IRR"]
    st.metric("IRR (Cơ sở)", f"{(irrv*100 if not np.isnan(irrv) else float('nan')):,.2f}%")

st.caption("💡 Lưu ý: IRR âm hoặc NaN có thể xuất hiện khi dòng tiền không đổi dấu theo kiểu dự án truyền thống.")

st.dataframe(
    summary_df.assign(
        **{
            "NPV": summary_df["NPV"].apply(lambda x: fmt_money(x)),
            "IRR": summary_df["IRR"].apply(lambda x: f"{x*100:,.2f}%"
                                           if not (pd.isna(x) or np.isnan(x)) else "NaN"),
            "Payback (năm)": summary_df["Payback (năm)"].apply(
                lambda x: f"{x:,.2f}" if not (pd.isna(x) or np.isnan(x)) else "Không hoàn vốn"
            ),
            "Vay dự kiến": summary_df["Vay dự kiến"].apply(lambda x: fmt_money(x)),
            "LTV": summary_df["LTV"].apply(lambda x: f"{x*100:,.2f}%"),
        }
    )
)

# -----------------------------
# 3. Bang dong tien & bieu do
# -----------------------------
st.subheader("3. Dòng tiền hàng năm")
tabs = st.tabs([f"📊 {name}" for name, _ in tables])

for i, (name, df_cf) in enumerate(tables):
    with tabs[i]:
        st.markdown(f"**Kịch bản: {name}**")
        st.dataframe(df_cf.style.format({
            "CAPEX": "{:,.2f}",
            "VLĐ (đầu kỳ)": "{:,.2f}",
            "Doanh thu": "{:,.2f}",
            "Chi phí HĐ": "{:,.2f}",
            "Khấu hao": "{:,.2f}",
            "EBIT": "{:,.2f}",
            "Thuế (tiền mặt)": "{:,.2f}",
            "FCF": "{:,.2f}",
        }))
        st.line_chart(df_cf.set_index("Năm")["FCF"])

# -----------------------------
# 4. Ket luan tu dong
# -----------------------------
def verdict(npv_val, irr_val, wacc_val):
    cond_good = (npv_val > 0) and (not np.isnan(irr_val)) and (irr_val > wacc_val)
    cond_bad = (npv_val <= 0) and (np.isnan(irr_val) or irr_val < wacc_val)
    if cond_good:
        return "Hiệu quả tài chính tốt, **có thể xem xét cấp vốn** (NPV>0, IRR>WACC)."
    elif cond_bad:
        return "Không hiệu quả về tài chính, **không đề xuất cấp vốn** (NPV≤0 hoặc IRR≤WACC)."
    else:
        return "Cần xem xét thêm (chỉ tiêu lẫn lộn hoặc biến động cao)."

st.subheader("4. Kết luận của Hệ thống AI (tự động)")
base_npv = summaries[0]["NPV"]; base_irr = summaries[0]["IRR"]
st.markdown(f"""
**Tóm tắt (Kịch bản Cơ sở):**  
- NPV: `{fmt_money(base_npv)}`  
- IRR: `{(base_irr*100):,.2f}%` so với WACC `{wacc*100:,.2f}%`  
- LTV: `{summaries[0]["LTV"]*100:,.2f}%` trên tài sản đảm bảo  
- Payback: `{('Không hoàn vốn' if pd.isna(summaries[0]['Payback (năm)']) else f"{summaries[0]['Payback (năm)']:,.2f} năm")}`
""")

st.info(verdict(base_npv, base_irr, wacc))

# -----------------------------
# 5. Xuat bao cao (TXT an toan)
# -----------------------------
st.subheader("5. Xuất báo cáo")
def render_report_txt():
    lines = []
    lines.append("KET QUA THAM DINH DU AN - DCF PRO")
    lines.append("="*50)
    lines.append(f"Tong von dau tu: {fmt_money(total_invest)}")
    lines.append(f"Ty le vay: {debt_ratio*100:,.2f}%  |  Gia tri TSBĐ: {fmt_money(collateral_value)}  |  LTV: {summaries[0]['LTV']*100:,.2f}%")
    lines.append(f"WACC: {wacc*100:,.2f}%  |  Thue TNDN: {tax_rate*100:,.2f}%  |  Vong doi: {years} nam")
    lines.append(f"Doanh thu N1: {fmt_money(rev_y1)}  |  Chi phi N1: {fmt_money(opex_y1)}")
    lines.append(f"Tang truong Rev: {g_rev*100:,.2f}%  |  Tang truong Opex: {g_opex*100:,.2f}%")
    lines.append(f"VLĐ ban dau: {fmt_money(wc0)}  |  Co so khau hao: {fmt_money(dep_base)}  |  Thanh ly cuoi ky: {salvage_pct*100:,.2f}%")
    lines.append("")

    for kq in summaries:
        lines.append(f"[Kich ban: {kq['Kịch bản']}]")
        lines.append(f"- NPV: {fmt_money(kq['NPV'])}")
        irr_txt = "NaN" if pd.isna(kq["IRR"]) else f"{kq['IRR']*100:,.2f}%"
        lines.append(f"- IRR: {irr_txt}")
        pb_txt = "Khong hoan von" if pd.isna(kq["Payback (năm)"]) else f"{kq['Payback (năm)']:,.2f} nam"
        lines.append(f"- Payback: {pb_txt}")
        lines.append("")
    lines.append("Ket luan tu dong (Kich ban co so): " + verdict(base_npv, base_irr, wacc))
    return "\n".join(lines)

report_txt = render_report_txt()
st.download_button(
    label="⬇️ Tải Báo cáo (TXT)",
    data=report_txt.encode("utf-8"),
    file_name="Bao_cao_tham_dinh_DCF.txt",
    mime="text/plain",
)

st.caption("🔧 Ghi chú: Nếu môi trường của bạn có thư viện PDF/Word, bạn có thể mở rộng hàm xuất báo cáo để sinh PDF/DOCX.")

# -----------------------------
# (Tuy chon) Goi y mo rong PDF/Word
# -----------------------------
with st.expander("Hướng dẫn mở rộng xuất PDF/Word (tùy chọn)"):
    st.markdown("""
- **PDF**: Cài `reportlab` rồi tạo `SimpleDocTemplate` và `Paragraph` để sinh file PDF từ `report_txt` (hoặc render đẹp bằng bảng).  
- **Word**: Cài `python-docx`, tạo tài liệu, thêm Heading/Paragraph, rồi `doc.save('Bao_cao.docx')`.  
- Sau đó dùng `st.download_button` để cho phép tải file.
""")
