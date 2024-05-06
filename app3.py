import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import plotly.figure_factory as ff

# Cấu hình trang cho layout rộng
st.set_page_config(layout="wide")

# Tải dữ liệu
df_vis = pd.read_csv('Streamlit2.csv')

# Tạo các tab
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Tổng quan", "Biểu đồ", "Gợi ý", "Dự đoán", "Phân tích"])

with tab2:
    # Streamlit App
    st.title('Thị trường xe ô tô hiện nay')
    # Lấy ra giá trị nhỏ nhất và lớn nhất của năm sản xuất từ dữ liệu
    year_min, year_max = df_vis['Năm sản xuất'].min(), df_vis['Năm sản xuất'].max()
    # Lấy ra giá trị nhỏ nhất và lớn nhất của giá xe từ dữ liệu
    price_min, price_max = df_vis['Giá xe'].min(), df_vis['Giá xe'].max()

    # Tạo các container, chia làm 3 cột với tỉ lệ mong muốn
    col1, col2, col3 = st.columns([1, 1, 1])

    with col3:
        # Tạo slider cho năm sản xuất
        year_range = st.slider(
            "Năm sản xuất",
            min_value=int(year_min),
            max_value=int(year_max),
            value=(int(year_min), int(year_max))
        )

        # Tạo slider cho giá xe
        price_range = st.slider(
            "Giá xe (VNĐ)",
            min_value=int(price_min),
            max_value=int(price_max),
            value=(int(price_min), int(price_max)),
            step=1000000  # Bước nhảy 1 triệu VNĐ
        )

    # Lọc dữ liệu dựa trên giá trị của sliders
    filtered_data = df_vis[(df_vis['Năm sản xuất'] >= year_range[0]) & 
                           (df_vis['Năm sản xuất'] <= year_range[1]) & 
                           (df_vis['Giá xe'] >= price_range[0]) & 
                           (df_vis['Giá xe'] <= price_range[1])]

    with col1:
        total_cars = len(filtered_data)
        st.metric("Tổng số xe hiện đang có", total_cars)
        avg_price = filtered_data['Giá xe'].mean()
        st.metric("Giá xe trung bình", f"{avg_price:,.0f} VNĐ")  # Định dạng số theo chuẩn Việt Nam
    with col2:
        colors = ['#9FC490', '#C0DFA1']
        # Tỷ lệ xe theo Xuất xứ
        fig2 = px.pie(filtered_data, names='Xuất xứ', title='Tỷ lệ xe theo Xuất xứ', color_discrete_sequence=colors)
        st.plotly_chart(fig2, use_container_width=True)

    # Sử dụng layout rộng để căn biểu đồ
    col1, col2 = st.columns([2, 2])
    with col1:
        # Số lượng xe theo Kiểu dáng
        fig1 = px.bar(filtered_data['Kiểu dáng'].value_counts().reset_index(),
                      x='index', y='Kiểu dáng',
                      labels={'index':'Kiểu dáng', 'Kiểu dáng':'Số lượng'},
                      title='Số lượng xe theo Kiểu dáng',
                      color_discrete_sequence=colors)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Biểu đồ giá xe trung bình theo năm sản xuất
        fig3 = px.line(filtered_data.groupby('Năm sản xuất')['Giá xe'].mean().reset_index(),
                       x='Năm sản xuất', y='Giá xe', markers=True,
                       title='Giá xe trung bình theo năm sản xuất', 
                       color_discrete_sequence=colors)
        st.plotly_chart(fig3, use_container_width=True)

    # Sắp xếp biểu đồ "Top 10 dòng xe" và "Phân phối giá xe" sử dụng full width
    col3, col4 = st.columns([1, 1])
    with col3:
        # Sắp xếp dữ liệu giảm dần và đảo ngược thứ tự để hiển thị đúng trong biểu đồ bar ngang
        sorted_data = filtered_data['Nhãn hiệu'].value_counts().head(10).reset_index()
        reversed_data = sorted_data.iloc[::-1]

        # Tạo biểu đồ
        fig4 = px.bar(reversed_data,
                    y='index', x='Nhãn hiệu', orientation='h',
                    labels={'index':'Nhãn hiệu', 'Nhãn hiệu':'Số lượng'},
                    title='Top 10 hãng xe được ưa chuộng', 
                    color_discrete_sequence=colors)

        # Hiển thị biểu đồ với Streamlit
        st.plotly_chart(fig4, use_container_width=True)
    with col4:
        # Phân phối giá xe
        fig5 = px.histogram(filtered_data, x='Giá xe', nbins=30, marginal="box",
                            title='Phân phối giá xe', 
                            color_discrete_sequence=colors)
        st.plotly_chart(fig5, use_container_width=True)
    pass

with tab4:
    import streamlit as st
    import numpy as np
    import pandas as pd
    import pickle

    # Tải dữ liệu
    df = pd.read_csv('Streamlit2.csv')
    # # Tiêu đề của webapp
    # st.title('Dự đoán giá xe')

    # Load các mô hình
    with open('LR1.pkl', 'rb') as file:
        model1 = pickle.load(file)
    with open('DT1.pkl', 'rb') as file:
        model2 = pickle.load(file)
    with open('RF1.pkl', 'rb') as file:
        model3 = pickle.load(file)
    with open('XGB1.pkl', 'rb') as file:
        model4 = pickle.load(file)
    with open('GRS2.pkl', 'rb') as file:
        model5 = pickle.load(file)

    # Load label encoder
    with open('label_encoders.pkl', 'rb') as file:
        label_encoders = pickle.load(file)

    # Load scaler price
    with open('scaler_price1.pkl', 'rb') as file:
        scaler_price = pickle.load(file)

    # Load scaler km    
    with open('scaler_km1.pkl', 'rb') as file:
        scaler_km = pickle.load(file)

    # Hàm chuẩn hóa số km đã đi
    def normalize_km(km):
        km_array = np.array(km).reshape(-1, 1)  
        normalized_km = scaler_km.transform(km_array)
        return normalized_km

    # Hàm chuẩn hóa ngược giá xe
    def inverse_normalize_price(scaled_price):
        price_array = np.array(scaled_price).reshape(-1, 1) 
        original_price = scaler_price.inverse_transform(price_array)
        return original_price

    # Function to encode user inputs using the saved mappings
    def encode_input(column, value):
        if column in label_encoders and value in label_encoders[column]:
            return label_encoders[column][value]
        else:
            return -1
    #Tạo form nhập thông tin xe
    with st.form("car_info_form", clear_on_submit=False):
        st.header('Thông tin xe cần Dự đoán')

        car_make = st.selectbox("Tên xe", options=sorted(df['Tên xe'].unique()))
        year = st.slider("Năm sản xuất", int(df['Năm sản xuất'].min()), int(df['Năm sản xuất'].max()))
        origin = st.selectbox("Xuất xứ", options=df['Xuất xứ'].unique())
        body_style = st.selectbox("Kiểu dáng", options=df['Kiểu dáng'].unique())
        transmission = st.selectbox("Hộp số", options=df['Hộp số'].unique())
        ext_color = st.selectbox("Màu ngoại thất", options=df['Màu ngoại thất'].unique())
        int_color = st.selectbox("Màu nội thất", options=df['Màu nội thất'].unique())
        mileage = st.slider("Số km đã đi", min_value=int(df['Số Km đã đi'].min()), max_value=int(df['Số Km đã đi'].max()), step=1000)
        seats = st.selectbox("Số chỗ ngồi", options=df['Số chỗ ngồi'].unique())
        doors = st.selectbox("Số cửa", options=df['Số cửa'].unique())
        drive = st.selectbox("Dẫn động", options=df['Dẫn động'].unique())
        fuel_type = st.selectbox("Nhiên liệu", options=df['Nhiên liệu'].unique())
        engine_size_options = np.arange(int(df['Dung tích'].min()), int(df['Dung tích'].max()))
        engine_size = st.select_slider("Dung tích", options=engine_size_options)

        submit_button = st.form_submit_button("Dự đoán giá")
    
    if submit_button:
        #Encode giá trị đầu vào
        origin_encoded = encode_input("Xuất xứ", origin)
        body_style_encoded = encode_input("Kiểu dáng", body_style)
        transmission_encoded = encode_input("Hộp số", transmission)
        ext_color_encoded = encode_input("Màu ngoại thất", ext_color)
        int_color_encoded = encode_input("Màu nội thất", int_color)
        drive_encoded = encode_input("Dẫn động", drive)
        car_make_encoded = encode_input("Tên xe", car_make)
        fuel_type_encoded = encode_input("Nhiên liệu", fuel_type)
        mileage_normalized = normalize_km(mileage)

        features = {
            "Năm sản xuất": year,
            "Xuất xứ": origin_encoded,
            "Kiểu dáng": body_style_encoded,
            "Hộp số": transmission_encoded,
            "Màu ngoại thất": ext_color_encoded,
            "Màu nội thất": int_color_encoded,
            "Số chỗ ngồi": seats,
            "Số cửa": doors,
            "Dẫn động": drive_encoded,
            "Tên xe": car_make_encoded,
            "Nhiên liệu": fuel_type_encoded,
            "Dung tích": engine_size,
            "Số Km đã đi (scaled)": mileage_normalized,
        }
        

        input_df = pd.DataFrame([features])

        input_df['Số Km đã đi (scaled)'] = input_df['Số Km đã đi (scaled)'].astype(float)
        # Bước 3: Dự đoán giá xe với từng mô hình
        predictions = {
            'Model1 (Linear Regression)': model1.predict(input_df),
            'Model2 (Decision Tree)': model2.predict(input_df),
            'Model3 (Random Forest)': model3.predict(input_df),
            'Model4 (XGBoost)': model4.predict(input_df),
            'Model5 (Grid Search)': model5.predict(input_df),
        }

        # Chuyển đổi kết quả dự đoán từ dạng chuẩn hóa về dạng gốc nếu cần
        for model_name, pred in predictions.items():
            # Đảm bảo rằng `pred` là một số scalar bằng cách sử dụng `item()` hoặc truy cập phần tử đầu tiên
            original_price_pred = inverse_normalize_price(pred).item()  # Hoặc `[0][0]` nếu `pred` là một mảng 2 chiều

            # Bây giờ bạn có thể sử dụng `format` mà không gặp lỗi
            st.write(f"{model_name} - Dự đoán giá: {original_price_pred:.2f} VND")

with tab3:
    import streamlit as st
    import numpy as np
    import pandas as pd
    import pickle

    # Tải dữ liệu
    df = pd.read_csv('Streamlit2.csv')

    # Load các mô hình
    with open('RCM.pkl', 'rb') as file:
        model_rf = pickle.load(file)

    # Load label encoder
    with open('label_encoders_rcm3.pkl', 'rb') as file:
        label_encoders = pickle.load(file)

    # Load scaler
    with open('scale_rcm.pkl', 'rb') as file:
        scaler = pickle.load(file)

    # Hàm chuẩn hóa
    def normalize(values):
        input_df = pd.DataFrame([values])
        normalized_array = scaler.transform(input_df)
        return normalized_array[0]

    # Hàm encoding input
    def encode_input(column, value):
        if column in label_encoders and value in label_encoders[column]:
            return label_encoders[column][value]
        else:
            return -1

    def reverse_encode_label(encoded_labels, feature_name):
        # Kiểm tra xem feature_name có tồn tại trong label_encoders không
        if feature_name in label_encoders:
            # Lấy dictionary ánh xạ ngược (từ mã hóa về nhãn gốc)
            inverse_mapping = {v: k for k, v in label_encoders[feature_name].items()}
            # Áp dụng ánh xạ ngược cho các giá trị đã mã hóa
            original_labels = [inverse_mapping[label] for label in encoded_labels]
            return original_labels
        else:
            raise ValueError(f"No mapping found for feature '{feature_name}'.")


    # Tạo form nhập thông tin xe
    with st.form("car_info_form2", clear_on_submit=False):
        st.header('Yêu cầu khi chọn xe')

        # Nhập các thông tin cần thiết
        price = st.slider("Giá xe", int(df['Giá xe'].min()), int(df['Giá xe'].max()), step=1000000)
        year = st.slider("Năm sản xuất", int(df['Năm sản xuất'].min()), int(df['Năm sản xuất'].max()))
        origin = st.selectbox("Xuất xứ", options=df['Xuất xứ'].unique())
        body_style = st.selectbox("Kiểu dáng", options=df['Kiểu dáng'].unique())
        transmission = st.selectbox("Hộp số", options=df['Hộp số'].unique())
        ext_color = st.selectbox("Màu ngoại thất", options=df['Màu ngoại thất'].unique())
        int_color = st.selectbox("Màu nội thất", options=df['Màu nội thất'].unique())
        mileage = st.slider("Số km đã đi", min_value=int(df['Số Km đã đi'].min()), max_value=int(df['Số Km đã đi'].max()), step=1000)
        seats = st.selectbox("Số chỗ ngồi", options=df['Số chỗ ngồi'].unique())
        doors = st.selectbox("Số cửa", options=df['Số cửa'].unique())
        drive = st.selectbox("Dẫn động", options=df['Dẫn động'].unique())
        fuel_type = st.selectbox("Nhiên liệu", options=df['Nhiên liệu'].unique())
        engine_size_options = np.arange(int(df['Dung tích'].min()), int(df['Dung tích'].max()) + 0.1, 0.1)
        engine_size = st.select_slider("Dung tích", options=engine_size_options)

        submit_button = st.form_submit_button("Gợi ý xe phù hợp với nhu cầu")

    if submit_button:
        # Encode và chuẩn hóa giá trị đầu vào
        values_to_normalize = {
            "Năm sản xuất": year,
            "Xuất xứ": encode_input("Xuất xứ", origin),
            "Kiểu dáng": encode_input("Kiểu dáng", body_style),
            "Hộp số": encode_input("Hộp số", transmission),
            "Màu ngoại thất": encode_input("Màu ngoại thất", ext_color),
            "Màu nội thất": encode_input("Màu nội thất", int_color),
            "Số chỗ ngồi": seats,
            "Số cửa": doors,
            "Dẫn động": encode_input("Dẫn động", drive),
            "Giá xe": price,
            "Nhiên liệu": encode_input("Nhiên liệu", fuel_type),
            "Dung tích": engine_size,
            "Số Km đã đi": mileage
        }

        normalized_values = normalize(values_to_normalize)
        input_df = pd.DataFrame([normalized_values], columns=scaler.feature_names_in_)

        # Dự đoán xác suất các lớp bởi mô hình
        predictions = model_rf.predict_proba(input_df)

        # Tìm chỉ số của 5 xác suất cao nhất
        top_5_indices = np.argsort(predictions[0])[::-1][:5]
        top_5_labels = model_rf.classes_[top_5_indices]

        feature_name = 'Tên xe'  # Tên thuộc tính mà top_5_labels đại diện

        # Reverse encoding
        original_labels = reverse_encode_label(top_5_labels, feature_name)
        for i, score in enumerate(original_labels, start=1):
            st.write(f"Chiếc xe {i} phù hợp với lựa chọn của bạn: {score}")


with tab1:
    # Tiêu đề của webapp
    st.title('Tổng quan về Website')

    st.write("""
    Chào mừng bạn đến với trang web của tôi! 

    Tôi là Phù Trung Thiện, sinh viên năm cuối chuyên ngành Khoa học Dữ liệu và Phân tích Kinh doanh Khóa 46K thuộc rường Đại học Kinh tế - Đại học Đà Nẵng. Trang web này được tạo ra với mục đích chính là phục vụ những người đam mê và tìm hiểu về thị trường xe cũ. Với giao diện thân thiện và các tính năng thông minh, chúng tôi cung cấp cái nhìn sâu sắc về xu hướng thị trường hiện tại, đồng thời đưa ra các gợi ý phù hợp và dự đoán giá xe dựa trên tình trạng của xe. Tất cả dữ liệu được phân tích một cách chính xác và kỹ lưỡng, là kết quả của quá trình nghiên cứu và phát triển trong suốt khóa học của tôi.

    Dự án này không chỉ là một phần quan trọng trong đề án tốt nghiệp của tôi, mà còn thể hiện niềm đam mê và kỹ năng tôi đã tích lũy được trong lĩnh vực khoa học dữ liệu. Mỗi bước phát triển của trang web này đều được thực hiện với sự tỉ mỉ và chính xác cao, từ thu thập dữ liệu, phân tích cho đến triển khai ứng dụng.

    Mọi thắc mắc hay yêu cầu hỗ trợ có thể được gửi đến tôi qua Email tại trungthien0503@gmail.com hoặc qua số điện thoại 0702776903. Để tìm hiểu thêm về các dự án khác của tôi, xin mời bạn ghé thăm trang Github của tôi tại .

    Cảm ơn bạn đã ghé thăm và hy vọng bạn tìm thấy thông tin hữu ích trên trang web của tôi!
    """)

with tab5:
    import streamlit as st
    import pandas as pd
    import plotly.express as px

    st.title('Phân tích thị trường xe ô tô hiện nay')

    st.markdown("""
                Tôi xin bày tỏ lòng biết ơn sâu sắc đến Website bonbanh.com vì đã cung cấp những dữ liệu quý giá, đã hỗ trợ rất lớn trong 
                quá trình nghiên cứu và học tập của tôi. Những thông tin từ bonbanh.com không chỉ giúp tôi tiếp cận được với nguồn tài nguyên 
                phong phú mà còn góp phần mở rộng kiến thức của tôi về ngành ô tô, qua đó nâng cao kỹ năng và sự hiểu biết chuyên môn.

                Bên cạnh đó, tôi cũng xin được gửi lời cảm ơn chân thành nhất đến Thạc sĩ Trần Nguyễn Hoàng Phương - giáo viên hướng dẫn trong
                đề án lần này. Thầy đã không ngừng nỗ lực và tận tâm hướng dẫn, chỉ bảo cho tôi trong suốt quá trình học tập. 
                Sự kiên nhẫn và kiến thức sâu rộng của Thầy đã vô cùng quan trọng, giúp tôi vượt qua những khó khăn và thử thách trong 
                đề án lần này.

                Trong những năm gần đây, thị trường xe ô tô cũ đã trở thành một phần không thể thiếu của ngành công nghiệp ô tô, không chỉ 
                đóng góp vào sự phát triển kinh tế mà còn phản ánh những biến đổi trong thói quen và nhu cầu của người tiêu dùng. Với số lượng 
                xe đang lưu hành trên thị trường là 11.550 chiếc và mức giá trung bình để có thể sở hữu một chiếc xe ô tô cũ là 612.940.433 đồng,
                thị trường xe cũ không ngừng đặt ra những cơ hội và 
                thách thức cho các nhà sản xuất, nhà phân phối, cũng như người mua xe. Bài phân tích này không chỉ nhằm mục đích cung cấp một 
                cái nhìn toàn diện về thị trường xe ô tô cũ thông qua việc phân tích dữ liệu từ quá khứ đến hiện tại, mà còn hướng đến việc đưa 
                ra những dự báo chính xác về xu hướng trong tương lai. Qua đó, chúng ta có thể hiểu sâu hơn về các yếu tố ảnh hưởng đến giá cả, 
                sở thích thương hiệu, lựa chọn kiểu dáng xe, và ảnh hưởng của chính sách đối với quyết định mua bán xe ô tô cũ của người tiêu 
                dùng.
                """)

    col1, col2 = st.columns([1, 1])

    with col1:
        # Tỷ lệ xe theo Xuất xứ
        fig2 = px.pie(df_vis, names='Xuất xứ', title='Tỷ lệ xe theo Xuất xứ')
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown("""
                    Biểu đồ tròn cho thấy phân chia thị phần giữa xe ô tô nhập khẩu và xe ô tô lắp ráp trong nước. 
                    Theo biểu đồ, xe lắp ráp trong nước chiếm tỷ lệ lớn hơn với 64.1%, trong khi xe nhập khẩu chiếm 35.9%. 
                    Điều này cho thấy một xu hướng ưa chuộng xe lắp ráp trong nước trên thị trường xe cũ, có thể do các yếu tố như giá cả 
                    phải chăng hơn, khả năng tiếp cận dịch vụ sau bán hàng dễ dàng hơn, hoặc chính sách thuế ưu đãi từ chính phủ đối với 
                    sản phẩm nội địa. Tuy nhiên, tỷ lệ không nhỏ của xe nhập khẩu cũng cho thấy có một phân khúc không nhỏ người tiêu dùng 
                    quan tâm đến các mẫu xe nước ngoài, có thể là do chất lượng cao, thiết kế đẳng cấp, hoặc công nghệ tiên tiến. 
                    Biểu đồ cung cấp một cái nhìn tổng quan về sự phân chia thị phần giữa hai phân khúc xe này và là điểm khởi đầu quan 
                    trọng cho việc phân tích sâu hơn về hành vi và sở thích của người tiêu dùng trong thị trường xe ô tô cũ
                    """)
    # Số lượng xe theo Kiểu dáng
    fig1 = px.bar(df_vis['Kiểu dáng'].value_counts().reset_index(),
                    x='index', y='Kiểu dáng',
                    labels={'index':'Kiểu dáng', 'Kiểu dáng':'Số lượng'},
                    title='Số lượng xe theo Kiểu dáng')
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("""
                Biểu đồ cột này thể hiện sự phân bố số lượng xe ô tô cũ theo kiểu dáng trên thị trường. Từ biểu đồ, chúng ta có thể thấy rằng:

                - Sedan là kiểu dáng xe có số lượng cao nhất trên thị trường xe cũ với lượng rất lớn, chỉ ra rằng đây có lẽ là kiểu dáng phổ biến nhất, có thể do giá cả hợp lý, tính năng phù hợp với nhu cầu hàng ngày, và chi phí bảo trì tương đối thấp.
                - SUV đứng ở vị trí thứ hai, điều này không quá bất ngờ vì xu hướng toàn cầu hiện nay là ưa chuộng xe SUV do không gian rộng rãi, 
                khả năng vận hành đa dạng và cảm giác an toàn.
                - Hatchback và Crossover lần lượt đứng ở vị trí tiếp theo, cho thấy một sự ưa chuộng nhất định trong phân khúc xe nhỏ gọn và linh hoạt.
                - Bán tải / Pickup, Van/Minivan có số lượng ít hơn, phản ánh nhu cầu thị trường nhắm vào những chiếc xe chủ yếu dành cho công việc hoặc sử dụng gia đình.
                - Coupe và Wagon nằm ở vị trí cuối cùng với số lượng thấp nhất, có thể do chúng hướng đến các phân khúc thị trường hẹp hơn 
                hoặc thị hiếu đặc biệt.

                Nhìn chung, biểu đồ này cung cấp thông tin quý giá về thị hiếu của khách hàng trên thị trường xe cũ, đồng thời cũng gợi ý về những kiểu dáng xe nào nên được đề xuất hoặc quảng cáo nhiều hơn để đáp ứng nhu cầu của thị trường
                """)

    # Biểu đồ giá xe trung bình theo năm sản xuất
    fig3 = px.line(df_vis.groupby('Năm sản xuất')['Giá xe'].mean().reset_index(),
                    x='Năm sản xuất', y='Giá xe', markers=True,
                    title='Giá xe trung bình theo năm sản xuất')
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("""
                Biểu đồ đường này thể hiện mức độ thay đổi của giá xe trung bình theo từng năm sản xuất, từ năm 2004 đến năm 2024. 
                Dưới đây là một số điểm nổi bật từ biểu đồ:

                - Có một xu hướng tăng giá rõ ràng qua từng năm, từ mức trung bình khoảng dưới 200 triệu đồng trong năm 2004 lên đến đỉnh điểm 
                vào khoảng trên 800 triệu đồng vào năm 2023.

                - Vào nằm 2010 giá xe trung bình nhỉnh hơn những năm lân cận là 2009 và 2010 có thể được giải thích rằng trên thị trường những
                dòng xe được sản xuất vào năm 2010 là những dòng xe ở phân khúc cao cấp nên vì đó mà giá trung bình xe cũ ở năm đó cao hơn so 
                với những năm lân cận.
                
                - Từ năm 2012 đến năm 2023, có thể quan sát thấy một sự tăng trưởng ổn định và liên tục, với tốc độ tăng giá tăng dần qua mỗi năm, 
                phản ánh sự phục hồi và tăng trưởng của thị trường xe hơi cũng như nền kinh tế nói chung.
                

                """)

    # Sắp xếp dữ liệu giảm dần và đảo ngược thứ tự để hiển thị đúng trong biểu đồ bar ngang
    sorted_data = df_vis['Nhãn hiệu'].value_counts().head(10).reset_index()
    reversed_data = sorted_data.iloc[::-1]

    # Tạo biểu đồ
    fig4 = px.bar(reversed_data,
                    y='index', x='Nhãn hiệu', orientation='h',
                    labels={'index':'Nhãn hiệu', 'Nhãn hiệu':'Số lượng'},
                    title='Top 10 hãng xe được ưa chuộng')
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown("""
                Biểu đồ cột này trình bày số lượng xe cũ của các hãng xe khác nhau đang được bán trên thị trường xe cũ. 
                Dựa trên biểu đồ, chúng ta có thể phân tích như sau:

                - Toyota dẫn đầu danh sách với số lượng xe cũ được bán ra nhiều nhất. Sự ưa chuộng này có thể phản ánh uy tín của Toyota 
                về độ tin cậy và giá trị tái bán.
                - Hyundai và Kia, hai thương hiệu xe Hàn Quốc, đều có mặt trong top 3, cho thấy sự tăng trưởng về uy tín và chất lượng 
                trong những năm gần đây, cũng như giá trị tốt so với chi phí.
                - Mercedes, một thương hiệu xe sang, cũng góp mặt trong danh sách, điều này chứng tỏ có một thị phần đáng kể người tiêu 
                dùng ưa chuộng xe hơi cao cấp ngay cả khi mua xe cũ.
                - Ford và Mazda đứng ở vị trí tiếp theo, cả hai thương hiệu này đều có lịch sử lâu đời và danh tiếng về xe có chất lượng 
                tốt và chi phí bảo dưỡng hợp lý.
                - Honda và Mitsubishi, thường được biết đến với động cơ bền bỉ và hiệu quả nhiên liệu, cũng có mặt trong danh sách, cho 
                thấy những thương hiệu Nhật Bản vẫn giữ được sức hút trên thị trường xe cũ.
                - VinFast, một thương hiệu xe mới hơn đến từ Việt Nam, có vẻ như đang xây dựng được sự hiện diện trên thị trường xe cũ, 
                điều này phản ánh sự tăng trưởng và mức độ chấp nhận của thương hiệu trong nước.
                - Chevrolet, mặc dù đứng cuối danh sách, vẫn góp mặt trong top 10, cho thấy thương hiệu Mỹ này vẫn có sức ảnh hưởng đáng kể.
                
                Nhìn chung, biểu đồ cho thấy sự đa dạng của thị trường xe cũ, với sự hiện diện của cả các hãng xe truyền thống và các thương hiệu mới nổi. Điều này cung cấp cho người tiêu dùng nhiều lựa chọn và cho thấy sự cạnh tranh sôi động trong ngành công nghiệp này.
                """)

    # Phân phối giá xe
    fig5 = px.histogram(df_vis, x='Giá xe', nbins=30, marginal="box",
                        title='Phân phối giá xe')
    st.plotly_chart(fig5, use_container_width=True)
    st.markdown("""
                Biểu đồ phân phối giá xe ô tô cũ này có dạng hình chuông, phản ánh phân phối chuẩn với phần lớn xe cũ tập trung ở khoảng giá 
                trung bình và số lượng giảm dần khi giá tăng lên hoặc giảm xuống. Dưới đây là phân tích cụ thể:

                - Mức giá phổ biến nhất như trên biểu đồ chúng ta có thể thấy được là khoảng 500 triệu đồng - đây là mức giá có số lượng xe nhiều
                nhất trên thị trường hiện nay.

                - Số lượng xe giảm đáng kể khi giá tăng cao hơn 1 tỷ đồng, chỉ ra rằng có ít xe cũ với mức giá cao trên thị trường. Điều này có thể 
                phản ánh rằng xe cũ cao cấp hoặc xe có tính năng đặc biệt không phổ biến hoặc kém hấp dẫn do giá cao.

                Tương tự, số lượng xe cũ cũng giảm ở phía bên trái của đỉnh biểu đồ dưới khoảng 250 triệu, có thể là do ít xe có giá quá thấp, 
                hoặc xe ở mức giá này có thể nhanh chóng bị mua do giá rẻ.
                """)

