```python
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# =========================
# 1. CẤU HÌNH TRANG WEB
# =========================
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 ỨNG DỤNG DỰ ĐOÁN GIÁ CỔ PHIẾU")
st.markdown(
    """
    ### Phân tích dữ liệu và dự đoán xu hướng giá cổ phiếu
    Nhập mã cổ phiếu để xem dữ liệu lịch sử, biểu đồ giá
    và dự đoán giá tham khảo trong tương lai.
    """
)

st.info(
    "Lưu ý: Kết quả dự đoán chỉ mang tính tham khảo, "
    "không phải lời khuyên đầu tư tài chính."
)

# =========================
# 2. THANH ĐIỀU KHIỂN
# =========================
st.sidebar.header("⚙️ Cài đặt")

ticker = st.sidebar.text_input(
    "Nhập mã cổ phiếu",
    value="AAPL"
).strip().upper()

period = st.sidebar.selectbox(
    "Khoảng thời gian dữ liệu",
    ["1y", "2y", "5y", "10y"],
    index=1
)

forecast_days = st.sidebar.slider(
    "Số ngày muốn dự đoán",
    min_value=1,
    max_value=30,
    value=7
)

analyze_button = st.sidebar.button(
    "🔍 Phân tích cổ phiếu",
    type="primary"
)

# =========================
# 3. TẢI DỮ LIỆU CỔ PHIẾU
# =========================
@st.cache_data(ttl=300)
def load_stock_data(symbol, selected_period):
    stock = yf.Ticker(symbol)
    data = stock.history(
        period=selected_period,
        auto_adjust=True
    )

    if data.empty:
        return pd.DataFrame()

    data = data.dropna(subset=["Close"])
    return data


# =========================
# 4. XỬ LÝ VÀ DỰ ĐOÁN
# =========================
def prepare_prediction(data, days):
    df = data.copy()

    # Chuyển ngày thành số thứ tự
    df["Day"] = np.arange(len(df))

    X = df[["Day"]]
    y = df["Close"]

    # Mô hình hồi quy tuyến tính
    model = LinearRegression()
    model.fit(X, y)

    # Dự đoán giá trong tương lai
    future_days = np.arange(
        len(df),
        len(df) + days
    ).reshape(-1, 1)

    future_prices = model.predict(future_days)

    # Đánh giá tham khảo trên tập dữ liệu huấn luyện
    train_predictions = model.predict(X)
    mae = mean_absolute_error(y, train_predictions)

    return model, future_days.flatten(), future_prices, mae


# =========================
# 5. HIỂN THỊ KẾT QUẢ
# =========================
if analyze_button:
    if not ticker:
        st.error("Vui lòng nhập mã cổ phiếu!")
        st.stop()

    with st.spinner("Đang tải dữ liệu và phân tích..."):
        try:
            data = load_stock_data(ticker, period)

            if data.empty:
                st.error(
                    "Không tìm thấy dữ liệu cổ phiếu. "
                    "Hãy kiểm tra lại mã cổ phiếu."
                )
                st.stop()

            model, future_days, future_prices, mae = (
                prepare_prediction(data, forecast_days)
            )

            # Thông tin tổng quan
            latest_price = float(data["Close"].iloc[-1])
            first_price = float(data["Close"].iloc[0])

            change_percent = (
                (latest_price - first_price) / first_price * 100
                if first_price != 0 else 0
            )

            predicted_price = float(future_prices[-1])

            price_change_percent = (
                (predicted_price - latest_price)
                / latest_price * 100
                if latest_price != 0 else 0
            )

            # =========================
            # 6. CÁC CHỈ SỐ TỔNG QUAN
            # =========================
            st.subheader(f"📊 Tổng quan cổ phiếu: {ticker}")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Giá đóng cửa gần nhất",
                f"{latest_price:,.2f}"
            )

            col2.metric(
                "Biến động trong giai đoạn",
                f"{change_percent:+.2f}%"
            )

            col3.metric(
                f"Dự đoán sau {forecast_days} ngày giao dịch",
                f"{predicted_price:,.2f}",
                f"{price_change_percent:+.2f}%"
            )

            col4.metric(
                "Sai số MAE trên dữ liệu huấn luyện",
                f"{mae:,.2f}"
            )

            st.caption(
                "Giá cổ phiếu được hiển thị theo đơn vị tiền tệ "
                "do nguồn dữ liệu cung cấp. Sai số MAE ở đây "
                "được tính trên dữ liệu huấn luyện, không phản ánh "
                "chính xác khả năng dự đoán trên dữ liệu mới."
            )

            # =========================
            # 7. BIỂU ĐỒ GIÁ LỊCH SỬ
            # =========================
            st.subheader("📈 Biểu đồ giá đóng cửa")

            fig, ax = plt.subplots(figsize=(12, 5))

            ax.plot(
                data.index,
                data["Close"],
                label="Giá đóng cửa lịch sử",
                linewidth=2
            )

            ax.set_title(f"Lịch sử giá cổ phiếu {ticker}")
            ax.set_xlabel("Ngày")
            ax.set_ylabel("Giá")
            ax.legend()
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)
            plt.close(fig)

            # =========================
            # 8. BIỂU ĐỒ DỰ ĐOÁN
            # =========================
            st.subheader("🔮 Biểu đồ dự đoán giá")

            # Chỉ hiển thị 60 phiên gần nhất để biểu đồ dễ nhìn
            recent_data = data.tail(60)

            fig2, ax2 = plt.subplots(figsize=(12, 5))

            ax2.plot(
                recent_data.index,
                recent_data["Close"],
                label="Giá thực tế",
                linewidth=2
            )

            last_date = data.index[-1]

            # Tạo các ngày giao dịch dự kiến
            future_dates = pd.bdate_range(
                start=last_date + pd.Timedelta(days=1),
                periods=forecast_days
            )

            ax2.plot(
                future_dates,
                future_prices,
                label="Giá dự đoán",
                linestyle="--",
                marker="o"
            )

            ax2.set_title(
                f"Dự đoán giá cổ phiếu {ticker}"
            )
            ax2.set_xlabel("Ngày")
            ax2.set_ylabel("Giá")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            st.pyplot(fig2)
            plt.close(fig2)

            # =========================
            # 9. BẢNG GIÁ DỰ ĐOÁN
            # =========================
            st.subheader("📋 Bảng giá dự đoán")

            forecast_table = pd.DataFrame({
                "Ngày dự kiến": future_dates.strftime("%Y-%m-%d"),
                "Giá dự đoán": np.round(future_prices, 2)
            })

            st.dataframe(
                forecast_table,
                use_container_width=True,
                hide_index=True
            )

            # Cho phép tải dữ liệu dự đoán
            csv_data = forecast_table.to_csv(
                index=False
            ).encode("utf-8-sig")

            st.download_button(
                label="📥 Tải bảng dự đoán CSV",
                data=csv_data,
                file_name=f"{ticker}_forecast.csv",
                mime="text/csv"
            )

            # =========================
            # 10. DỮ LIỆU LỊCH SỬ
            # =========================
            with st.expander("Xem dữ liệu lịch sử"):
                st.dataframe(
                    data.tail(30),
                    use_container_width=True
                )

            st.success("Phân tích hoàn tất!")

        except Exception as e:
            st.error(
                "Đã xảy ra lỗi khi phân tích dữ liệu. "
                "Hãy kiểm tra mã cổ phiếu hoặc thử lại sau."
            )
            st.caption(f"Thông tin lỗi: {e}")

else:
    st.markdown("---")
    st.subheader("👋 Chào mừng bạn!")

    st.write(
        """
        Để bắt đầu phân tích:

        1. Nhập mã cổ phiếu ở thanh bên trái.
        2. Chọn khoảng thời gian dữ liệu.
        3. Chọn số ngày muốn dự đoán.
        4. Nhấn **Phân tích cổ phiếu**.
        5. Xem biểu đồ, kết quả dự đoán và tải bảng dữ liệu.
        """
    )

    st.markdown("### Một số mã cổ phiếu có thể thử")

    st.write(
        "- **AAPL**: Apple\n"
        "- **MSFT**: Microsoft\n"
        "- **GOOGL**: Alphabet\n"
        "- **TSLA**: Tesla\n"
        "- **NVDA**: NVIDIA\n"
        "- **FPT.VN**: Có thể thử với FPT trên Yahoo Finance\n"
    )

    st.caption(
        "Dữ liệu phụ thuộc vào khả năng cung cấp của Yahoo Finance. "
        "Mã cổ phiếu Việt Nam có thể cần hậu tố phù hợp, "
        "ví dụ FPT.VN; không phải mã nào cũng có dữ liệu."
    )

st.markdown("---")
st.caption(
    "Stock Price Predictor | Công cụ học tập và nghiên cứu "
    "phân tích dữ liệu tài chính."
)
```
