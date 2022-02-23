#!/usr/bin/env python
# coding: utf-8

# # Part A. 재구매 유형 분리 로직 개발

# # 0. Imort library

# In[1]:


import numpy as np
import pandas as pd 

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = 'notebook_connected'

from datetime import date, timedelta

import warnings
warnings.filterwarnings("ignore")

from IPython.display import Image


# # 1. Load Data

# In[2]:


data_client = pd.read_csv("'21년 해카톤_고객정보.csv", encoding='cp949')
data_contact = pd.read_csv("'21년 해카톤_접촉정보.csv", encoding='cp949')
data_car = pd.read_csv("'21년 해카톤_차량정보.csv", encoding='cp949')

data_client.rename(columns={'CUS_ID':'고객ID', 'PSN_BIZR_YN':'개인사업자여부', 'SEX_SCN_NM':'성별', 'TYMD':'생년월', 'CUS_ADM_TRY_NM':'주소_행정시도명', 'CUS_N_ADMZ_NM':'주소_시군구명', 'CUS_ADMB_NM':'주소_행정동명', 'CLB_HOUS_PYG_NM':'주택 평형', 'REAI_BZTC_AVG_PCE':'주택 평균가격'}, inplace=True)
data_contact.rename(columns={'CNTC_SN':'접촉일련번호', 'CUS_ID':'고객ID', 'CNTC_DT':'접촉일자', 'CNTC_CHAN_NM':'접촉채널명', 'CNTC_AFFR_SCN_NM':'접촉업무명'}, inplace=True)
data_car.rename(columns={'CAR_ID':'차량ID', 'CUS_ID':'고객ID', 'WHOT_DT':'출고일자', 'CAR_HLDG_STRT_DT':'보유시작일자', 'CAR_HLDG_FNH_DT':'보유종료일자', 'CAR_NM':'차명', 'CAR_CGRD_NM_1':'차량등급명1', 'CAR_CGRD_NM_2':'차량등급명2', 'CAR_ENG_NM':'엔진타입명', 'CAR_TRIM_NM':'트림명'}, inplace=True)


# 현대자동차 제공데이터인 **고객정보, 접촉정보, 차량정보** 데이터를 불러왔습니다. 직관적인 해석을 위해 column명을 모두 한글로 변경했습니다.

#  

# # 2. Data Pre-Processing

# ### 1) 고객정보: data_client

# In[3]:


# NA값 치환
data_client['개인사업자여부'] = data_client['개인사업자여부'].fillna('N')
data_client['주소_행정시도명'] = data_client['주소_행정시도명'].fillna("")
data_client['주소_시군구명'] = data_client['주소_시군구명'].fillna("")
data_client['주소_행정동명'] = data_client['주소_행정동명'].fillna("")

# 연령변수 생성
data_client['생년월'] = data_client['생년월'].astype(str)
data_client['출생연도'] = data_client['생년월'].str[:4].astype(int)
data_client['연령'] = 2021 - data_client['출생연도']

# 주소컬럼 병합
data_client['주소'] = data_client['주소_행정시도명'] + " " + data_client['주소_시군구명'] + " " + data_client['주소_행정동명']
data_client.drop(columns = ['생년월', '출생연도', '주소_행정시도명', '주소_시군구명', '주소_행정동명'], inplace = True)


# 먼저 **고객정보** 데이터입니다. **개인사업자여부, 주소** column의 NA값을 분석이 용이하도록 각각 fillna 해주었고, **생년월 및 출생연도** column을 활용하여 **고객의 연령**을 알 수 있는 column을 생성했습니다. 또한 3단계로 나뉘어진 주소를 하나의 column으로 병합했습니다. 마지막으로 분석에 활용할 column만 남기고 나머지 column은 모두 drop했습니다.

# ### 2) 접촉일자: data_contact

# In[4]:


# 접촉일자 type 변경
data_contact['접촉일자'] = pd.to_datetime(data_contact['접촉일자'], format='%Y%m%d')


# 두번째로 **접촉정보** 데이터입니다. object type으로 되어있는 **접촉일자** column을 datetime으로 변경했습니다.

# ### 3) 차량정보: data_car

# In[5]:


# 보유종료일자 NaN 처리
data_car['보유종료일자'] = data_car['보유종료일자'].fillna(20210614.0)

# 출고일자, 보유시작일자, 보유종료일자 type 변경
data_car['출고일자'] = pd.to_datetime(data_car['출고일자'], format='%Y%m%d')
data_car['보유시작일자'] = pd.to_datetime(data_car['보유시작일자'], format='%Y%m%d')
data_car['보유종료일자'] = pd.to_datetime(data_car['보유종료일자'], format='%Y%m%d')

# 출고~보유, 보유일 파생변수 생성
data_car['출고_보유시작'] = data_car['보유시작일자'] - data_car['출고일자']
data_car['보유시작_보유종료'] = data_car['보유종료일자'] - data_car['보유시작일자']
data_car['출고_보유종료'] = data_car['보유종료일자'] - data_car['출고일자']

# 한 고객이 한 차량만 거래한 경우 제외
# 고객ID별 차량ID 개수 count
data_car_count = pd.DataFrame(data_car.groupby('고객ID').count())
data_car_count.reset_index(inplace=True)
data_car_count = data_car_count[['고객ID', '차량ID']]

# 기존 데이터와 고객ID 기준 병합
data_car_not1 = pd.merge(data_car, data_car_count, on='고객ID', how='inner')

# 차량ID count가 1이 아닌 데이터만 추출 
data_car_not1 = data_car_not1[data_car_not1['차량ID_y'] != 1]
data_car_not1.rename(columns={'차량ID_y':'거래차량대수', '차량ID_x':'차량ID'}, inplace=True)


# 마지막으로 **차량정보** 데이터입니다. 파생변수를 생성하기 위해 보유종료일자 column의 NA값에 현재날짜로 fillna 했습니다. 그리고 **출고일자, 보유시작일자, 보유종료일자** column의 type을 datetime으로 변경했습니다. 이를 통해 파생변수를 생성했습니다. <br><br>
# >**출고_보유시작**: 보유시작일자 - 출고일자<br>
# >**보유시작_보유종료**: 보유종료일자 - 보유시작일자<br>
# >**출고_보유종료**: 보유종료일자 - 출고일자<br>
# 
# 또한 대차/추가구매 고객을 분리하는 로직을 개발하는데 차량구매건수가 1건인 고객은 분석에 포함하지 않는 것이 바람직하다고 판단하여 **고객별 거래차량대수** column을 생성하고 건수가 1건인 고객 데이터를 제외하였습니다.

#  

# # 3. 재구매 유형 분리/추정 로직 개발

# ### 1) Small EDA

# In[6]:


data_car_mean = data_car_not1.groupby(['고객ID', '출고일자', '보유시작일자', '보유종료일자']).mean().reset_index()
data_car_mean['거래차량대수'].value_counts()


# **차량정보** 데이터에서 **고객ID, 출고일자, 보유시작일자, 보유종료일자** 기준으로 groupby하여 데이터를 정렬하고, **거래차량대수** column을 value count하여 분포를 확인했습니다. 그 결과 거래차량대수가 2대인 고객이 100만건으로 가장 많았습니다.<br><br> 주고객의 재구매 경향을 분석하여 최적합 차량을 제공한다는 분석목표와 더불어 제한적인 개발환경을 고려하여, **고객별 거래차량대수가 2대**인 데이터만을 분석에 활용하였습니다.

# ### 2) 분석 DataFrame 및 '대차/추가구매 소요기간' column 생성

# In[7]:


# 거래차량대수가 2대인 데이터만 추출
data_car_2 = data_car_mean[data_car_mean['거래차량대수'] == 2]

# 홀수행, 짝수행 기준으로 기존차량/다음차량 분리
data_다음차량 = data_car_2[1::2]
data_기존차량 = data_car_2[::2]

# 다음차량의 출고일자에서 기존차량의 출고일자를 빼서 대차/추가구매 소요기간 계산
data_diff = data_다음차량['출고일자'].reset_index() - data_기존차량['출고일자'].reset_index()

# column merge를 위해 index 맞추기
data_다음차량 = data_다음차량.set_index('고객ID')
data_다음차량.reset_index(inplace=True)

# column merge
data_다음차량['대차/추가구매 소요기간'] = data_diff['출고일자']


# 고객ID, 출고일자, 보유시작일자, 보유종료일자별로 정렬한 **data_car_mean**데이터에서 거래차량대수가 2대인 데이터만 추출하고,<br> 홀수행과 짝수행 기준으로 분리하여 **기존차량거래 데이터프레임**과 **다음차량거래 데이터프레임**을 생성했습니다.<br><br>
# 다음차량거래 출고일자에서 기존차량거래 출고일자를 빼서 **대차/추가구매 소요기간** column을 생성했습니다. <br>그리고 다음차량거래 데이터프레임(data_다음차량)에 해당 column을 추가했습니다.

# ### 3) 평균보유일수 column 생성

# In[8]:


from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

# 거래차량대수가 2대인 데이터만 추출
data_car_only2 = data_car_not1[data_car_not1['거래차량대수'] == 2]
data_car_days = data_car_only2[['고객ID', '차량ID', '출고_보유시작', '보유시작_보유종료', '출고_보유종료']]

# datetime 변수를 연산가능한 int로 변경
data_car_days['출고_보유시작'] = data_car_days['출고_보유시작'].astype(str).str[:-5]
data_car_days['출고_보유시작'] = data_car_days['출고_보유시작'].astype(int)
data_car_days['보유시작_보유종료'] = data_car_days['보유시작_보유종료'].astype(str).str[:-5]
data_car_days['보유시작_보유종료'] = data_car_days['보유시작_보유종료'].astype(int)
data_car_days['출고_보유종료'] = data_car_days['출고_보유종료'].astype(str).str[:-5]
data_car_days['출고_보유종료'] = data_car_days['출고_보유종료'].astype(int)

# 고객ID로 groupby하여 평균보유일수 계산
data_car_days = pd.DataFrame(data_car_days.groupby(['고객ID']).mean())
data_car_days.reset_index(inplace=True)


# 다음으로 datetime type으로 되어있는 **출고_보유시작, 보유시작_보유종료, 출고_보유종료** column을 int type으로 변환하였습니다.<br> 이후 **고객ID**를 기준으로 groupby하여 카테고리별 평균보유일수를 계산했습니다. 이를 통해 고객별로 거래차량을 얼마나 오래 보유하고 있는지 파악할 수 있습니다.

# ### 4) 고객별 평균보유일수와 대차/추가구매 소요기간을 나타내는 DataFrame 생성

# In[9]:


# 계산된 columns 고객ID 기준으로 병합
data_car_2_고객ID = data_car_2.groupby('고객ID').mean().reset_index()
data_car_2_고객ID = pd.merge(data_다음차량, data_car_2_고객ID, on='고객ID', how='inner')
data_car_2_고객ID = pd.merge(data_car_2_고객ID, data_car_days, on='고객ID', how='inner')

# datetime 변수를 연산가능한 int로 변경
data_car_2_고객ID['대차/추가구매 소요기간'] = data_car_2_고객ID['대차/추가구매 소요기간'].astype(str).str[:-5]
data_car_2_고객ID['대차/추가구매 소요기간'] = data_car_2_고객ID['대차/추가구매 소요기간'].astype(int)

# 대차/추가구매 구분을 위한 차이 column 생성
data_car_2_고객ID['대차/추가구매'] = data_car_2_고객ID['대차/추가구매 소요기간'] - data_car_2_고객ID['출고_보유종료']
data_car_2_고객ID['|대차/추가구매|'] = abs(data_car_2_고객ID['대차/추가구매 소요기간'] - data_car_2_고객ID['출고_보유종료'])

# 최종으로 분리에 활용할 데이터프레임 추출
data_car_all_time = data_car_2_고객ID[['고객ID', '대차/추가구매 소요기간', '출고_보유종료', '대차/추가구매', '|대차/추가구매|']]


# '해카톤 과제설명서'의 대차/추가구매 기간 정의를 참고하여, 계산한 고객별 평균보유일수 columns(출고_보유종료, 보유시작_보유종료, 출고_보유시작) 중 **출고_보유종료**를 분리 로직에 활용하는 것이 적합하다고 판단하였습니다. 마찬가지로 과제설명서의 기간 정의를 참고하여, **대차/추가구매 소요기간**과 **출고_보유종료기간**의 차이를 계산하여 분리에 활용하는 것이 유의미할 것이라고 판단했습니다. 그리고 정교한 분리를 위해 차이의 실제값과 절대값을 모두 분석에 활용할 수 있도록 절댓값을 계산한 **|대차/추가구매|** column을 생성했습니다.<br><br>
# 따라서 고객별 **대차/추가구매 소요기간, 출고_보유종료, 대차/추가구매, |대차/추가구매|** 기간을 column으로 한 DataFrame을 분리에 활용할 최종 데이터프레임으로 선정하였습니다.

# ### 5) Small EDA

# In[10]:


fig = go.Figure()
fig.add_trace(go.Box(y=data_car_all_time['대차/추가구매 소요기간'], name='대차/추가구매 소요기간',
                marker_color = 'indianred'))
fig.add_trace(go.Box(y=data_car_all_time['대차/추가구매'], name = '대차/추가구매',
                marker_color = 'lightseagreen'))
fig.add_trace(go.Box(y=data_car_all_time['|대차/추가구매|'], name = '|대차/추가구매|',
                marker_color = 'yellow'))
fig.add_trace(go.Box(y=data_car_all_time['출고_보유종료'], name = '출고_보유종료',
                marker_color = 'purple'))

fig.show()


# In[11]:


대차추가구매_소요기간 = pd.DataFrame(data_car_all_time['대차/추가구매 소요기간'].agg(['count', 'mean', 'std', 'min', 'max']))
대차추가구매 = pd.DataFrame(data_car_all_time['대차/추가구매'].agg(['count', 'mean', 'std', 'min', 'max']))
대차추가구매_절댓값 = pd.DataFrame(data_car_all_time['|대차/추가구매|'].agg(['count', 'mean', 'std', 'min', 'max']))
출고_보유종료 = pd.DataFrame(data_car_all_time['출고_보유종료'].agg(['count', 'mean', 'std', 'min', 'max']))

pd.concat([대차추가구매_소요기간, 대차추가구매, 대차추가구매_절댓값, 출고_보유종료], axis=1)


# 위의 4가지 column의 특성을 파악한 결과, 값의 분포 범위가 넓고 **|대차/추가구매|, 출고_보유종료** column의 경우 이상치가 상당히 존재하고 있음을 알 수 있습니다.<br> 따라서 분리 알고리즘을 선택하고 정규화 방식을 선정할 때 이러한 특성을 반영해야 한다고 판단했습니다.

# ### 6) 군집 알고리즘 수행

# >**ㄱ. 정규화**<br><br>
# 정교한 군집화를 위해 **RobustScaler**를 활용하여 정규화를 진행했습니다. RobustScaler는 중앙값과 IQR을 사용하여 스케일링을 하는 것으로, 아웃라이어의 영향을 최소화 해줍니다. 이는 StandardScaler와 비교했을 때 동일한 값을 더 넓게 분포시킵니다. 위의 Small EDA 결과에서 알 수 있듯이 고객데이터의 특성상 극단적인 값들이 존재하므로 본 분석에서는 RobustScaler가 더 적합하다고 생각하여 정규화를 진행하였습니다.

# > **ㄴ. K-means Clustering**<br><br>
# 분리에 활용할 DataFrame의 column은 모두 **int type**입니다. 또한 정답 데이터셋(label)이 없는 데이터이기 때문에 군집을 위해서는 **비지도학습**을 수행해야 합니다. <br><br>
# K-means Clustering은 모집단 또는 범주에 대한 사전 정보가 없을 때 주어진 관측값 사이의 거리를 측정함으로써 유사성을 이용하여 분석하는 기법입니다. 전체 데이터를 집단으로 그룹화하는데, 이를 통해 각 집단의 성격을 파악하여 데이터의 구조를 이해할 수 있습니다. K-means Clustering의 핵심은 새로운 데이터와 기존 데이터 간 유클리디언 거리가 최소가 되도록 분류하는 것입니다. 기존 데이터를 기준점으로 유클리디언 거리를 측정하고 거리가 최소화 되도록 K개의 군집으로 clustering 합니다. K-means Clustering은 간단한 알고리즘으로 짧은 시간에 계산이 가능하며 탐색적인 방법이므로 대용량 데이터에 적합합니다. 또한 거리를 기준으로 분류하기 때문에 수치형 데이터일때 더 높은 성능을 가지는 알고리즘 입니다.

# ![군집](군집.png)

# *(이미지 출처 : https://opentutorials.org/course/4548/28942)*<br><br>
# 고객별 평균보유기간과 대차/추가구매기간을 활용하여 비슷한 특성을 가진 고객을 분류하기 위해서는 **군집화 알고리즘**을 활용해야 한다고 판단했으며, 정답 데이터셋이 없기 때문에 **비지도학습 군집 알고리즘**을 활용해야 한다고 판단했습니다.<br><br>
# 그 중에서도 현재 DataFrame의 특성(수치형)을 잘 반영하여 군집할 수 있는 **K-means Clustering**을 선정하여 **대차/추가구매** 데이터를 분리했습니다.

# In[12]:


# 군집화 데이터프레임 생성
col = ['대차/추가구매 소요기간', '출고_보유종료', '|대차/추가구매|']

group_data = data_car_all_time[col]

# 로버스트 스케일 정규화
from sklearn import preprocessing
from scipy.stats import boxcox
group_data = group_data.values
group_data = preprocessing.robust_scale(group_data)

# 2개로 군집화
from sklearn.cluster import KMeans
group = KMeans(n_clusters = 2)
group.fit(group_data)

centroids = group.cluster_centers_ 
labels = group.labels_
data_car_all_time['label'] = labels


# 먼저 군집화에 사용할 column을 선택했습니다. 모델이 **기간** 관점으로 학습하도록 하기 위하여 절댓값이 아닌 **대차/추가구매** column은 제외하였습니다.<br><br>
# 그리고 Robust Scaler로 **정규화**를 한 뒤 K를 2로 설정하여 (대차 / 추가구매) **K-means Clustering**을 수행했습니다.
# 수행결과를 전체 데이터프레임(data_car_all_time)에 **label** column으로 추가하였습니다. 

# ### 7. 군집 결과 시각화 (2차원, 3차원)

# In[13]:


# 군집 시각화 (2차원)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data_2D = data_car_all_time[['대차/추가구매 소요기간', '출고_보유종료', '|대차/추가구매|']]
pca = PCA(n_components = 2)
fit = pca.fit_transform(data_2D)
pca_data = pd.DataFrame(data = fit, columns = ['PC1', 'PC2'])

plt.figure(figsize = (20,10))
plt.scatter('PC1', 'PC2', data = pca_data, c = data_car_all_time['label'])

display()


# In[14]:


# 군집 시각화 (3차원)
pca3 = PCA(n_components=3)
data_pca3 = pca3.fit_transform(data_2D)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_pca3[:,0], data_pca3[:,1], data_pca3[:,2], c=data_car_all_time['label'], s=50, edgecolors='white')
ax.set_title('3D of Target distribution by diagnosis')
ax.set_xlabel('pcomp 1')
ax.set_ylabel('pcomp 2')
ax.set_zlabel('pcomp 3')

plt.show()


# (*군집 결과 시각화는 다차원 데이터를 PCA(차원축소)하여 시각화한 결과이기 때문에 실제 데이터의 분포와는 차이가 있을 수 있습니다.)<br>
# 
# 이상으로, 과제1의 **재구매 유형분리 로직개발** part를 완료했습니다. 이제 이를 바탕으로 EDA하여 대차 / 추가구매 데이터에 어떠한 유의미한 인사이트가 있을 지 분석하도록 하겠습니다.

# # 4. EDA

# ### 1) 대차/추가구매 DataFrame 분리

# In[15]:


# data_car에서 활용할 columns 추출
data_car_info = data_car[['고객ID', '차명', '차량등급명1', '차량등급명2', '엔진타입명', '트림명']]

# 고객ID 기준으로 병합
data_car_merge = pd.merge(data_car_info, data_car_all_time, on='고객ID', how='inner')
data_client_merge = pd.merge(data_client, data_car_all_time, on='고객ID', how='inner')
data_contact_merge = pd.merge(data_contact, data_car_all_time, on='고객ID', how='inner')

# 대차/추가구매 데이터 분리
data_car_merge_0 = data_car_merge[data_car_merge['label'] == 0]
data_car_merge_1 = data_car_merge[data_car_merge['label'] == 1]

data_client_merge_0 = data_client_merge[data_client_merge['label'] == 0]
data_client_merge_1 = data_client_merge[data_client_merge['label'] == 1]

data_contact_merge_0 = data_contact_merge[data_contact_merge['label'] == 0]
data_contact_merge_1 = data_contact_merge[data_contact_merge['label'] == 1]


# 위의 분석결과로 나뉜 **고객별 label(0, 1)** 을 기준으로 데이터를 분리했습니다.

# ### 2) Small EDA

# In[16]:


fig = go.Figure()
fig.add_trace(go.Box(y=data_car_merge_0['|대차/추가구매|'], name='label 0',
                marker_color = 'indianred'))
fig.add_trace(go.Box(y=data_car_merge_1['|대차/추가구매|'], name = 'label 1',
                marker_color = 'lightseagreen'))

fig.show()


# 해카톤 과제정의서의 대차/추가구매 정의에 따르면 **대차**의 경우는 기존 보유하던 차량을 처분하고, 신차를 출고하는 경우로 대차 소요 기간과 보유 기간이 유사하며, 보유 기간이 대차 소요 기간 보다 길 수도 있습니다. **추가구매**의 경우는 기존 보유하던 차량을 처분하지 않고, 추가로 신차를 출고하는 경우로 보유 기간과 추가구매 소요 기간이 일정 수준의 차이를 보입니다.<br>
# 
# 따라서, 대차/추가구매 기간이 더 긴 label 0의 데이터를 **추가구매**로, 대차/추가구매 기간이 더 짧은 label 1의 데이터를 **대차**로 선정했습니다.

# In[19]:


# DataFrame 이름 변경
추가구매_고객정보 = data_client_merge_0
대차_고객정보 = data_client_merge_1

추가구매_접촉정보 = data_contact_merge_0
대차_접촉정보 = data_contact_merge_1

추가구매_차량 = data_car_merge_0
대차_차량 = data_car_merge_1

