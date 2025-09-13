import streamlit as st
#pip install streamlit-option-menu
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import squarify
from datetime import datetime
import time

st.set_page_config(page_title="RFM Analysis", page_icon="img/logo.png", layout="wide")

df = pd.DataFrame()
# PROCESSING DATA
## load data and merge 2 bảng
df_trans = pd.read_csv('data/Transactions.csv')
df_product = pd.read_csv('data/Products_with_Categories.csv')
df = pd.merge(df_trans, df_product, on='productId', how='left')
df ['total'] = df['price'] * df['items']

## tiền xử lý dữ liệu
### định dạng lại cột Date
string_to_date = lambda x : datetime.strptime(x, "%d-%m-%Y").date()
df['Date'] = df['Date'].apply(string_to_date)
df['Date'] = df['Date'].astype('datetime64[ns]')
### xóa na
df.dropna(inplace=True)
### xóa trùng lặp
if df.duplicated().sum():
  print(f'Trước khi xóa trùng lặp dữ liệu. {df.count()}')
  df = df.drop_duplicates()
  print(f'Sau khi xóa trùng lặp dữ liệu. {df.count()}')
else:
  print('Không có trùng lặp dữ liệu.')
### Chuẩn hóa về mô hình RFM
max_date = df['Date'].max().date()

R_func = lambda x : (max_date - x.max().date()).days
F_func  = lambda x: len(x.unique())
M_func = lambda x : round(sum(x), 2)

df_RFM = df.groupby('Member_number').agg(
    Recency = ('Date',R_func),
    Frequency = ('Date',F_func),
    Monetary = ('total',M_func))
#### xử lý phân phối dữ liệu của RFM
# Vẽ phân phối của 'Recency'
# Create labels for Recency, Frequency, Monetary
r_labels = range(4, 0, -1) # số ngày tính từ lần cuối mua hàng lớn thì gán nhãn nhỏ, ngược lại thì nhãn lớn
f_labels = range(1, 5)
m_labels = range(1, 5)

# Assign these labels to 4 equal percentile groups
r_groups = pd.qcut(df_RFM['Recency'].rank(method='first'), q=4, labels=r_labels)
f_groups = pd.qcut(df_RFM['Frequency'].rank(method='first'), q=4, labels=f_labels)
m_groups = pd.qcut(df_RFM['Monetary'].rank(method='first'), q=4, labels=m_labels)
# Create new columns R, F, M
df_RFM = df_RFM.assign(R = r_groups.values, F = f_groups.values,  M = m_groups.values)
# Create RFM_Segment column by concatenating R, F, M
def join_rfm(x): return str(int(x['R'])) + str(int(x['F'])) + str(int(x['M']))
df_RFM['RFM_Segment'] = df_RFM.apply(join_rfm, axis=1)
# calculate RFM_Segment counts
rfm_count_unique = df_RFM.groupby('RFM_Segment')['RFM_Segment'].nunique()
# Calculate RFM_Score
df_RFM['RFM_Score'] = df_RFM[['R','F','M']].sum(axis=1)

##XÂY DỰNG MÔ HÌNH
def rfm_level(df):
    # Check for special 'STARS' and 'NEW' conditions first
    if df['RFM_Score'] == 12:
        return 'DIAMOND'
    elif (df['R'] >= 3 or df['F'] >= 3) and df['M'] >= 3:
        return 'VIP'
    elif df['R'] == 4 or df['F'] == 1 or df['M'] == 1:
        return 'NEW'
    # Then check for other conditions
    elif df['R'] == 1:
        return 'LOST'
    else:
        return 'REGULARS'

processing_time = []
 # Create a new column RFM_Level
start_time = time.time()
df_RFM['RFM_Level'] = df_RFM.apply(rfm_level, axis=1)
processing_time.append(time.time() - start_time)

# Calculate average values for each RFM_Level, and return a size of each segment
df_rfm_agg = df_RFM.groupby('RFM_Level').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': ['mean', 'count']}).round(0)

df_rfm_agg.columns = df_rfm_agg.columns.droplevel()
df_rfm_agg.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
df_rfm_agg['Percent'] = round((df_rfm_agg['Count']/df_rfm_agg.Count.sum())*100, 2)

# Reset the index
df_rfm_agg = df_rfm_agg.reset_index()

###RFM + KMeans
df_kmeans = df_RFM[['Recency','Frequency','Monetary']]
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sse = {}
silhouette_list = []
k_range = range(2, 20)

for k in k_range: # Silhouette score is not defined for k=1
    kmeans = KMeans(n_clusters=k, random_state=42) # Explicitly set n_init
    model = kmeans.fit(df_kmeans)
    sse[k] = kmeans.inertia_ # SSE to closest cluster centroid
    
# Build model with k=5
start_time = time.time()
model = KMeans(n_clusters=5, random_state=42)
model.fit(df_kmeans)
processing_time.append(time.time() - start_time)

df_kmeans["Cluster"] = model.labels_.copy()

# Calculate average values for each RFM_Level, and return a size of each segment
rfm_agg2 = df_kmeans.groupby('Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': ['mean', 'count']}).round(0)

rfm_agg2.columns = rfm_agg2.columns.droplevel()
rfm_agg2.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
rfm_agg2['Percent'] = round((rfm_agg2['Count']/rfm_agg2.Count.sum())*100, 2)

# Reset the index
rfm_agg2 = rfm_agg2.reset_index()

# Change thr Cluster Columns Datatype into discrete values
rfm_agg2['Cluster'] = 'Cluster '+ rfm_agg2['Cluster'].astype('str')

## Evaluate K-Means performance
from sklearn.metrics import (
    silhouette_score, silhouette_samples, davies_bouldin_score,
    calinski_harabasz_score
)

print("\n📊 Evaluating K-Means Performance:")
# silhoutte
silhouette_avg = silhouette_score(df_kmeans, model.labels_, metric='euclidean')
db_score = davies_bouldin_score(df_kmeans, model.labels_)
ch_score = calinski_harabasz_score(df_kmeans, model.labels_)

# ==== GUI ==== 
def footer():
    #centered footer        
    st.markdown("""<div style="text-align: center;">=_=</div>""", unsafe_allow_html=True)

## sidebar
with st.sidebar:
    st.logo("img/logo.png",icon_image="img/logo.png")
    st.subheader("Đồ án Data Science")
    st.markdown("[Trung tâm tin học](https://csc.edu.vn/) <br/>GVHD: Khuất Thùy Phương <br/>HV: Thái Thị Ngọc Lý", unsafe_allow_html=True)
    
    choice= option_menu(
        menu_title="Nội dung", 
        options=["Giới thiệu đề tài", "Hiểu dữ liệu", "Tiền xử lý dữ liệu", "Xây dựng mô hình", "Dự đoán và đánh giá"], 
        icons=["house", "bar-chart", "stack", "boxes","speedometer",], 
        menu_icon="cast", 
        default_index=0,
        styles={
            "container": {"padding": "5!important", "background-color": "#fafafa"},
            "icon": {"color": "orange", "font-size": "25px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#02ab21", "font-family": "Roboto", "color": "white", "font-weight": "none"},
        }  
    )
     
if choice == "Giới thiệu đề tài":
    st.write("Giới thiệu đề tài |")
    def home_page():
        st.title("Project 1: RFM Analysis")
        st.subheader("Tình huống bài toán")
        st.write("Cửa hàng X chủ yếu bán các sản phẩm thiết yếu cho khách hàng như rau, củ, quả, thịt, cá, trứng, sữa, nước giải khát... Khách hàng của cửa hàng là khách hàng mua lẻ. Chủ cửa hàng X mong muốn có thể bán được nhiều hàng hóa hơn cũng như giới thiệu sản phẩm đến đúng đối tượng khách hàng, chăm sóc và làm hài lòng khách hàng.")
        st.image('img/img1.png', caption='')
        st.markdown("**Vấn đề kinh doanh:** <br/> - tăng doanh thu.<br/>- tăng lượng khách hàng trung thành.", unsafe_allow_html=True)
        st.markdown("**Vấn đề khai phá dữ liệu**<br/> - Phân khúc khách hàng bằng RFM", unsafe_allow_html=True)

    home_page()
elif choice == "Thu thập và hiểu dữ liệu":
    st.write("Thu thập và hiểu dữ liệu |")
    st.write("Dữ liệu giao dịch")
    st.dataframe(df_trans.head())
    st.write("Dữ liệu sản phẩm")
    st.dataframe(df_product.head())
    st.write("Dữ liệu sau khi gộp")
    st.dataframe(df.head())
elif choice == "Tiền xử lý dữ liệu":
    st.write("Tiền xử lý dữ liệu |")    
    st.markdown("**Dữ liệu sau khi tiền xử lý**<br/>- định dạng lại cột Date<br/>- xóa na<br/>- xóa trùng lặp", unsafe_allow_html=True)
    st.write(f'Bảng dữ liệu có: {df.shape}')
    st.dataframe(df.head(3))
    st.subheader("Chuẩn hóa về mô hình RFM")
    st.write(f"Số lượng khách hàng: {len(df.Member_number.unique()):,}")
    st.markdown("**Bảng dữ liệu theo mô hình RFM**", unsafe_allow_html=False)
    df_RFM = df_RFM.sort_values('Monetary', ascending=False)
    st.dataframe(df_RFM.head())
    # phân phối dữ liệu của RFM
    st.markdown("**Phân phối dữ liệu của RFM**")
    col1, col2, col3 = st.columns(3)
    with col1: # Vẽ phân phối của 'Recency'
        fg = plt.figure(figsize=(5,4))
        plt.hist(df_RFM['Recency'], bins=40, edgecolor='black') # Chọn số lượng bins phù hợp
        plt.title('Distribution of Recency')
        plt.xlabel('Recency')
        st.pyplot(fg)

    with col2: # Vẽ phân phối của 'Frequency'
        fg = plt.figure(figsize=(5,4))
        plt.hist(df_RFM['Frequency'], bins=10, edgecolor='black') # Chọn số lượng bins phù hợp
        plt.title('Distribution of Frequency')
        plt.xlabel('Frequency')
        st.pyplot(fg)

    with col3: # Vẽ phân phối của 'Monetary'
        plt.hist(df_RFM['Monetary'], bins=40, edgecolor='black') # Chọn số lượng bins phù hợp
        plt.title('Distribution of Monetary')
        plt.xlabel('Monetary')
        st.pyplot(fg)
    st.text("Bảng dữ liệu sau khi phân nhóm RFM")
    st.dataframe(df_RFM.head())
    
elif choice == "Xây dựng mô hình":
    st.write("Xây dựng mô hình |")
    tab1, tab2, tab3 = st.tabs(["Manual RFM", "RFM + KMeans", "RFM + Hierachical clustering"])
    with tab1:
        st.subheader("Phân khúc khách hàng theo RFM")
        st.dataframe(df_RFM[::600])
        df_RFM['RFM_Level'] = df_RFM.apply(rfm_level, axis=1)
        st.markdown("""
        **Ý nghĩa các phân khúc khách hàng**<br/>
        - DIAMOND: Khách hàng kim cương, mua hàng rất thường xuyên, chi tiêu nhiều tiền và gần đây.<br/>
        - VIP: Khách hàng VIP, mua hàng thường xuyên, chi tiêu nhiều tiền và gần đây.<br/>
        - REGULARS: Khách hàng thường xuyên, mua hàng không thường xuyên, chi tiêu trung bình và không gần đây.<br/>
        - NEW: Khách hàng mới, mua hàng không thường xuyên, chi tiêu ít tiền và gần đây.<br/>
        - LOST: Khách hàng đã mất, mua hàng không thường xuyên, chi tiêu ít tiền và không gần đây.
        """, unsafe_allow_html=True)
        
        #Create our plot and resize it.
        fig = plt.gcf()
        ax = fig.add_subplot()
        fig.set_size_inches(10, 8)

        colors_dict = {'DIAMOND':'grey','LOST':'red', 'REGULARS':'pink', 'NEW':'green', 'VIP':'yellow'}

        squarify.plot(sizes=df_rfm_agg['Count'],
                    text_kwargs={'fontsize':8, 'fontname':"Roboto"},
                    color=colors_dict.values(),
                    label=['{} \n{:.0f} days \n{:.0f} transactions \n{:.0f} $ \n{:.0f} customers ({}%)'.format(*df_rfm_agg.iloc[i])
                            for i in range(0, len(df_rfm_agg))], alpha=0.5 )


        plt.title("Phân khúc khách hàng theo mô hình RFM", fontsize=12)
        plt.axis('off')
        st.pyplot(fig)

        # fig1 = px.scatter(df_rfm_agg, x="RecencyMean", y="FrequencyMean", size="MonetaryMean", color="RFM_Level",
        #    hover_name="RFM_Level", size_max=60)
        # st.pyplot(fig1)
    with tab2:
        #st.subheader("Phân khúc khách hàng theo mô hình RFM + KMeans")
        st.write("Dữ liệu đầu vào")
        st.dataframe(df_kmeans.tail())
        st.write("Tìm số cụm")
        fg2 = plt.figure(figsize=(10, 6))
        plt.subplot(1,2,1)
        plt.title('The Elbow Method')
        plt.xlabel('k')
        plt.ylabel('SSE')

        plt.plot(list(sse.keys()), list(sse.values()), marker='o')
        plt.grid(True, alpha=0.3)

        plt.subplot(1,2,2)
        plt.title('Silhouette Score')
        plt.xlabel('k')
        plt.ylabel('Silhouette')
        plt.plot(k_range, silhouette_list, 'bo-', linewidth=2, markersize=8, color='red')
        plt.grid(True, alpha=0.3)
        st.pyplot(fg2)
        
        st.write("Kết quả phân cụm với K=5")
        st.dataframe(df_kmeans.head())
        
        fig4 = plt.gcf()
        ax = fig4.add_subplot()
        fig4.set_size_inches(14, 10)

        colors_dict2 = {'Cluster0':'yellow','Cluster1':'royalblue', 'Cluster2':'cyan','Cluster3':'red', 'Cluster4':'purple', }

        squarify.plot(sizes=rfm_agg2['Count'],
                    text_kwargs={'fontsize':8, 'fontname':"sans serif"},
                    color=colors_dict2.values(),
                    label=['{} \n{:.0f} days \n{:.0f} transactions \n{:.0f} $ \n{:.0f} customers ({}%)'.format(*rfm_agg2.iloc[i])
                            for i in range(0, len(rfm_agg2))], alpha=0.5 )

        plt.title("KMeans-RFM Segments",fontsize=13)
        plt.axis('off')
        st.pyplot(fig4)
        
    with tab3:
        st.subheader("Phân khúc khách hàng theo mô hình RFM + Hierachical clustering")

    st.write(f"Silhouette Score: {silhouette_avg:.4f}")
    st.write(f"Davies-Bouldin Index: {db_score:.4f} (lower is better)")
    st.write(f"Calinski-Harabasz Index: {ch_score:.2f} (higher is better)")
elif choice == "Dự đoán và đánh giá":
    st.write("Dự đoán và đánh giá |")

st.sidebar.markdown("""<div style="text-align: center;">09.2025</div>""", unsafe_allow_html=True)
# main
footer()