import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from google import genai
from google.genai.errors import APIError
from pandas.api.types import is_numeric_dtype
from pandas_datareader import wb

# --- CẤU HÌNH BAN ĐẦU ---
st.set_page_config(
    page_title="Phân Tích Dữ Liệu Kinh Tế Vĩ Mô Việt Nam",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- KHAI BÁO CÁC CHỈ SỐ KINH TẾ VĨ MÔ VÀ TÊN HIỂN THỊ ---
INDICATOR_MAP = {
    'NY.GDP.MKTP.KD.ZG': 'Tăng trưởng GDP (năm %)',
    'FP.CPI.TOTL.ZG': 'Lạm phát (giá tiêu dùng, năm %)',
    'SL.UEM.TOTL.ZS': 'Tỷ lệ thất nghiệp (tổng % lực lượng LĐ)',
    'NE.EXP.GNFS.ZS': 'Xuất khẩu Hàng hóa & DV (% GDP)',
    'NE.IMP.GNFS.ZS': 'Nhập khẩu Hàng hóa & DV (% GDP)',
    'GC.DOD.TOTL.GD.ZS': 'Nợ Chính phủ Trung ương (tổng % GDP)',
    'BX.KLT.DINV.CD.WD': 'FDI ròng vào (Triệu USD)',
    'SP.POP.TOTL': 'Dân số (người)',
    'NY.GDP.PCAP.CD': 'GDP bình quân đầu người (USD hiện tại)',
    'NY.GDP.MKTP.CD': 'GDP (USD hiện tại) - Dùng tính tỷ trọng FDI'
}

COUNTRY_CODE = 'VNM'

# --- HÀM TẢI DỮ LIỆU TỪ WORLDBANK (ĐÃ SỬA) ---
@st.cache_data(show_spinner="Đang trích xuất dữ liệu từ World Bank...")
def get_worldbank_data(indicators, country, start_year, end_year):
    """
    Tải dữ liệu từ World Bank Data API sử dụng pandas_datareader.wb.download.
    """
    if not indicators:
        return pd.DataFrame()

    try:
        fdi_code = 'BX.KLT.DINV.CD.WD'
        gdp_code = 'NY.GDP.MKTP.CD'
        
        # Chuẩn bị danh sách indicators
        indicators_to_fetch = list(set(indicators))
        if fdi_code in indicators:
            indicators_to_fetch.append(gdp_code)
            indicators_to_fetch = list(set(indicators_to_fetch))

        # Lấy dữ liệu từ World Bank
        data = wb.download(
            indicator=indicators_to_fetch, 
            country=country, 
            start=start_year,
            end=end_year
        )
        
        # Xử lý dữ liệu
        df = data.reset_index()
        df = df.rename(columns={'year': 'Year', 'country': 'Country'})
        
        # Chuyển đổi năm sang số nguyên
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype(int)
        
        # Tính FDI (% GDP) nếu cần
        if fdi_code in indicators and gdp_code in df.columns:
            df[gdp_code] = pd.to_numeric(df[gdp_code], errors='coerce')
            df[fdi_code] = pd.to_numeric(df[fdi_code], errors='coerce')
            
            gdp_series = df[gdp_code].replace(0, np.nan)
            df['FDI net inflows (% GDP)'] = (df[fdi_code] / gdp_series) * 100
            
            df = df.drop(columns=[fdi_code, gdp_code], errors='ignore')

        # Đổi tên các cột
        final_col_names = ['Year']
        for code in indicators:
            if code == fdi_code:
                final_col_names.append('FDI net inflows (% GDP)')
            elif code in INDICATOR_MAP:
                final_col_names.append(INDICATOR_MAP[code])

        df.columns = [INDICATOR_MAP.get(col, col) for col in df.columns]
        
        # Lọc và sắp xếp
        available_cols = [col for col in final_col_names if col in df.columns]
        df_final = df[available_cols].sort_values(by='Year', ascending=True)

        return df_final
    
    except Exception as e:
        st.error(f"Lỗi khi tải dữ liệu từ World Bank: {e}")
        return pd.DataFrame()

# --- HÀM TÍNH TOÁN THỐNG KÊ MÔ TẢ ---
def calculate_descriptive_stats(df):
    """Tính toán thống kê mô tả chi tiết cho từng chỉ số."""
    stats_list = []
    numeric_cols = [col for col in df.columns if is_numeric_dtype(df[col])]

    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            stats_list.append({
                'Chỉ tiêu': col, 'Trung bình (Mean)': 'N/A', 'Độ lệch chuẩn (Std Dev)': 'N/A', 
                'Giá trị nhỏ nhất (Min)': 'N/A', 'Năm Min': 'N/A',
                'Giá trị lớn nhất (Max)': 'N/A', 'Năm Max': 'N/A',
                'Trung vị (Median)': 'N/A', 'Tứ phân vị Q1': 'N/A', 
                'Tứ phân vị Q3': 'N/A', 'Hệ số biến thiên (CV, %)' : 'N/A'
            })
            continue

        mean_val = series.mean()
        std_val = series.std()
        min_val = series.min()
        max_val = series.max()
        median_val = series.median()
        q1_val = series.quantile(0.25)
        q3_val = series.quantile(0.75)
        cv = (std_val / mean_val) * 100 if mean_val != 0 else np.nan

        try:
            year_min = df.loc[df[col] == min_val, 'Year'].iloc[0]
        except:
            year_min = 'N/A'
            
        try:
            year_max = df.loc[df[col] == max_val, 'Year'].iloc[0]
        except:
            year_max = 'N/A'

        stats_list.append({
            'Chỉ tiêu': col,
            'Trung bình (Mean)': f"{mean_val:,.2f}",
            'Độ lệch chuẩn (Std Dev)': f"{std_val:,.2f}",
            'Giá trị nhỏ nhất (Min)': f"{min_val:,.2f}",
            'Năm Min': year_min,
            'Giá trị lớn nhất (Max)': f"{max_val:,.2f}",
            'Năm Max': year_max,
            'Trung vị (Median)': f"{median_val:,.2f}",
            'Tứ phân vị Q1': f"{q1_val:,.2f}",
            'Tứ phân vị Q3': f"{q3_val:,.2f}",
            'Hệ số biến thiên (CV, %)': f"{cv:,.2f}%" if not np.isnan(cv) else 'N/A'
        })

    return pd.DataFrame(stats_list)

# --- HÀM GỌI API GEMINI ---
def get_ai_analysis(stats_df, country, start_year, end_year, api_key):
    """Gửi bảng thống kê đến Gemini để phân tích."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash'
        
        stats_markdown = stats_df.to_markdown(index=False)

        prompt = f"""
        Bạn là một Chuyên gia Kinh tế Vĩ mô và Phân tích Thị trường Tài chính hàng đầu. 
        Nhiệm vụ của bạn là phân tích tình hình kinh tế của {country} trong giai đoạn từ năm {start_year} đến năm {end_year}.

        Dưới đây là Bảng Thống kê Mô tả chi tiết cho các chỉ số kinh tế vĩ mô quan trọng:
        {stats_markdown}

        Dựa trên bảng thống kê trên và các chỉ số sau (Trung bình, Độ lệch chuẩn, Hệ số biến thiên):
        1.  **Đánh giá Tốc độ Tăng trưởng và Ổn định Kinh tế (dựa trên GDP Growth, Lạm phát, Thất nghiệp)**. 
            Độ lệch chuẩn và Hệ số biến thiên cao cho thấy sự bất ổn.
        2.  **Đánh giá Cán cân Đối ngoại (dựa trên Xuất/Nhập khẩu và FDI)**.
        3.  **Đánh giá Sức khỏe Tài khóa (dựa trên Nợ Chính phủ)**.

        Hãy viết một báo cáo phân tích tổng hợp (khoảng 3-5 đoạn) bằng tiếng Việt, tập trung vào xu hướng, mức độ ổn định và so sánh các chỉ số quan trọng trong giai đoạn này.
        """

        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"

# --- GIAO DIỆN STREAMLIT ---
st.sidebar.header("Tùy Chọn Dữ Liệu")
st.sidebar.markdown(f"**Quốc gia:** Việt Nam ({COUNTRY_CODE})")
st.sidebar.info("Ứng dụng hiện chỉ tập trung vào dữ liệu Việt Nam từ World Bank.")

col_start, col_end = st.sidebar.columns(2)
CURRENT_YEAR = pd.Timestamp('now').year
START_YEAR_DEFAULT = 2000

with col_start:
    start_year = st.number_input("Năm Bắt Đầu", min_value=1960, max_value=CURRENT_YEAR, value=START_YEAR_DEFAULT)
with col_end:
    end_year = st.number_input("Năm Kết Thúc", min_value=1960, max_value=CURRENT_YEAR, value=CURRENT_YEAR)

if start_year > end_year:
    st.sidebar.error("Năm bắt đầu phải nhỏ hơn hoặc bằng năm kết thúc.")

INDICATOR_OPTIONS = {name: code for code, name in INDICATOR_MAP.items() if code != 'NY.GDP.MKTP.CD'}

selected_indicators_names = st.sidebar.multiselect(
    "Chọn các Chỉ số Kinh tế cần trích xuất:",
    options=list(INDICATOR_OPTIONS.keys()),
    default=list(INDICATOR_OPTIONS.keys())[:5]
)

selected_ids = [INDICATOR_OPTIONS[name] for name in selected_indicators_names]

# --- CHỨC NĂNG CHÍNH ---
if selected_ids and start_year <= end_year:
    df_data = get_worldbank_data(selected_ids, COUNTRY_CODE, start_year, end_year)

    if not df_data.empty:
        missing_count = df_data.isnull().sum().sum()
        if missing_count > 0:
            st.warning(f"Cảnh báo: Phát hiện **{missing_count}** giá trị thiếu (Missing Data).")
            df_filled = df_data.ffill().bfill()
            
            df_display = df_filled.copy()
            for col in df_display.columns:
                 if is_numeric_dtype(df_display[col]):
                    df_display[col] = df_display[col].replace([np.inf, -np.inf, np.nan], 'N/A')
            
            st.info("Giá trị thiếu đã được xử lý tự động bằng phương pháp **điền giá trị gần nhất**.")
            
        else:
            df_filled = df_data
            df_display = df_data.replace([np.inf, -np.inf, np.nan], 'N/A')

        tab1, tab2, tab3, tab4 = st.tabs([
            "1. Bảng Dữ liệu & Tải về", 
            "2. Biểu đồ Trực quan", 
            "3. Thống kê Mô tả",
            "4. Phân tích AI Tổng hợp"
        ])
        
        with tab1:
            st.subheader("Bảng Tổng hợp Dữ liệu Kinh tế Vĩ mô")
            st.dataframe(df_display, use_container_width=True, height=500)

            @st.cache_data
            def to_excel(df):
                output = BytesIO()
                df_to_save = df.replace('N/A', np.nan) 
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df_to_save.to_excel(writer, index=False, sheet_name='Du_lieu_WorldBank')
                return output.getvalue()

            excel_data = to_excel(df_filled)
            st.download_button(
                label="📥 Tải Dữ liệu về File Excel (.xlsx)",
                data=excel_data,
                file_name=f'worldbank_data_{COUNTRY_CODE}_{start_year}-{end_year}.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )

        with tab2:
            st.subheader("Trực quan hóa Xu hướng Biến động theo Thời gian")

            chart_type = st.radio(
                "Chọn Loại Biểu Đồ Chính:",
                ('Biểu đồ Đường (Line Chart)', 'Biểu đồ Cột (Bar Chart)', 'Phân tích Tương quan (Scatter/Heatmap)')
            )

            chart_cols = [col for col in df_filled.columns if col != 'Year']
            
            if not chart_cols:
                st.warning("Không có cột dữ liệu hợp lệ để vẽ biểu đồ.")
            else:
                if chart_type in ('Biểu đồ Đường (Line Chart)', 'Biểu đồ Cột (Bar Chart)'):
                    selected_chart_indicators = st.multiselect(
                        "Chọn các chỉ số để hiển thị trên biểu đồ:",
                        options=chart_cols,
                        default=chart_cols[:min(len(chart_cols), 3)]
                    )

                    if selected_chart_indicators:
                        fig, ax = plt.subplots(figsize=(12, 6))
                        color_palette = sns.color_palette("viridis", len(selected_chart_indicators))

                        for i, indicator in enumerate(selected_chart_indicators):
                            if chart_type == 'Biểu đồ Đường (Line Chart)':
                                ax.plot(df_filled['Year'], df_filled[indicator], marker='o', label=indicator)
                            elif chart_type == 'Biểu đồ Cột (Bar Chart)':
                                sns.barplot(x=df_filled['Year'], y=df_filled[indicator], ax=ax, label=indicator, color=color_palette[i])
                                
                        ax.set_title(f"Xu hướng Biến động của các Chỉ số ({start_year}-{end_year})", fontsize=16)
                        ax.set_xlabel("Năm", fontsize=12)
                        ax.set_ylabel("Giá trị", fontsize=12)
                        ax.legend(loc='best')
                        ax.grid(True, linestyle='--', alpha=0.6)
                        plt.xticks(df_filled['Year'].unique(), rotation=45, ha='right')
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                elif chart_type == 'Phân tích Tương quan (Scatter/Heatmap)':
                    corr_method = st.radio("Chọn Phương pháp Tương quan:", ('Biểu đồ Phân tán (Scatter Plot)', 'Biểu đồ Nhiệt Ma trận Tương quan (Heatmap)'))
                    
                    if corr_method == 'Biểu đồ Phân tán (Scatter Plot)':
                        col_x, col_y = st.columns(2)
                        with col_x:
                            indicator_x = st.selectbox("Chọn Chỉ số cho Trục X:", options=chart_cols, index=0)
                        with col_y:
                            indicator_y = st.selectbox("Chọn Chỉ số cho Trục Y:", options=chart_cols, index=min(len(chart_cols)-1, 1))

                        if indicator_x and indicator_y:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.scatter(df_filled[indicator_x], df_filled[indicator_y])
                            
                            for i, row in df_filled.iterrows():
                                ax.annotate(row['Year'], (row[indicator_x], row[indicator_y]), textcoords="offset points", xytext=(0,5), ha='center')
                                
                            ax.set_title(f"Mối tương quan: {indicator_x} vs {indicator_y}", fontsize=16)
                            ax.set_xlabel(indicator_x, fontsize=12)
                            ax.set_ylabel(indicator_y, fontsize=12)
                            ax.grid(True, linestyle='--', alpha=0.6)
                            st.pyplot(fig)

                    elif corr_method == 'Biểu đồ Nhiệt Ma trận Tương quan (Heatmap)':
                        corr_matrix = df_filled[chart_cols].corr(method='pearson')
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, cbar_kws={'label': 'Hệ số tương quan'})
                        ax.set_title("Ma trận Tương quan giữa các Chỉ số", fontsize=16)
                        plt.tight_layout()
                        st.pyplot(fig)

        with tab3:
            st.subheader(f"Thống kê Mô tả Giai đoạn {start_year} - {end_year}")
            stats_df = calculate_descriptive_stats(df_filled)
            st.dataframe(stats_df, use_container_width=True)
            
            st.caption("""
            **Giải thích:** **Độ lệch chuẩn** và **Hệ số biến thiên** (CV) càng cao cho thấy mức độ biến động/bất ổn của chỉ số trong giai đoạn càng lớn.
            """)

        with tab4:
            st.subheader("Phân tích Chuyên sâu từ Gemini AI")
            st.markdown("Chức năng này sử dụng Bảng Thống kê (Tab 3) và các biểu đồ trực quan (Tab 2) làm cơ sở để AI phân tích tình hình kinh tế tổng thể của Việt Nam.")
            
            try:
                api_key = st.secrets["GEMINI_API_KEY"]
            except KeyError:
                api_key = None
                st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")

            if api_key:
                if st.button("🌟 Yêu cầu AI Phân tích Tổng hợp"):
                    with st.spinner('Đang gửi dữ liệu thống kê và chờ Gemini phân tích...'):
                        stats_df_for_ai = calculate_descriptive_stats(df_filled)
                        
                        ai_result = get_ai_analysis(
                            stats_df_for_ai, 
                            "Việt Nam", 
                            start_year, 
                            end_year, 
                            api_key
                        )
                        st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                        st.info(ai_result)

    else:
        st.warning("Không có dữ liệu được tải về cho các chỉ số và khoảng thời gian đã chọn. Vui lòng kiểm tra lại tùy chọn.")

else:
    st.info("Vui lòng chọn ít nhất một Chỉ số và đảm bảo khoảng thời gian hợp lệ.")
