import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import os, urllib
import datetime
from pandas_datareader import data
from utils import *
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib as mat
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb
import warnings
import threading
from sklearn.cluster import KMeans
import numpy as np
from kneed import DataGenerator, KneeLocator
from neuralprophet import NeuralProphet
from pycaret.anomaly import *
import torch
mat.style.use('ggplot')

def data_init(lookback_min,lookback_max):
    global target, daypara, df, df2, df_4pycaret, df_temp
    target, portfolio, daypara = sidebar_func(lookback_min,lookback_max)
    target = st.selectbox('Target', portfolio)
    with st.spinner('Fetching Fata...'):
        df,df2,df_4pycaret,df_temp = fetch_data(target,daypara)
    df = original_df_parse(df)

def data_browse():
    global target, daypara, df, df2, df_4pycaret, df_temp
    st.dataframe(df)
    st.line_chart(df["price"])
    st.bar_chart(df["vol"])

def feature_engineer():
    global target, daypara, df, df2, df_4pycaret, df_temp
    lookback = st.sidebar.slider("Lookback for Feature",30,240)
    df_parsed = feature_df_parse(df,lookback)
    st.dataframe(df_parsed)
    (X_train_FI, y_train_FI), (X_test_FI, y_test_FI) = get_feature_importance_data(df_parsed)
    regressor = xgb.XGBRegressor(gamma=0.0,n_estimators=150,base_score=0.7,colsample_bytree=1,learning_rate=0.05)
    xgbModel = regressor.fit(X_train_FI,y_train_FI, \
                      eval_set = [(X_train_FI, y_train_FI), (X_test_FI, y_test_FI)], \
                     verbose=False)
    eval_result = regressor.evals_result()
    training_rounds = range(len(eval_result['validation_0']['rmse']))
    #plot
    plt.figure()
    plt.xticks(rotation='vertical')
    plt.bar([i for i in range(len(xgbModel.feature_importances_))], xgbModel.feature_importances_.tolist(), tick_label=X_test_FI.columns)
    plt.title('Feature importance of the technical indicators.')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

def clustering():
    global target, daypara, df, df2, df_4pycaret, df_temp
    st.write(df)
    lookback = len(df.index) * (-1)
    X= np.array(df["price"][lookback:])
    sum_of_squared_distances = []
    K = range(1,15)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(X.reshape(-1,1))
        sum_of_squared_distances.append(km.inertia_)
    kn = KneeLocator(K, sum_of_squared_distances,S=1.0, curve="convex", direction="decreasing")
    kn.plot_knee(figsize=(7,3))
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.subheader("Search Number of Regime")
    st.pyplot()
    st.subheader("Plotting in Reallity")
    with st.spinner("Loading Chart..."):
        kmeans = KMeans(n_clusters= kn.knee).fit(X.reshape(-1,1))
        c = kmeans.predict(X.reshape(-1,1))
        minmax = []
        for i in range(kn.knee):
            minmax.append([-np.inf,np.inf])
        for i in range(len(X)):
            cluster = c[i]
            if X[i] > minmax[cluster][0]:
                minmax[cluster][0] = X[i]
            if X[i] < minmax[cluster][1]:
                minmax[cluster][1] = X[i]
        plt.figure(figsize=(11,5),dpi=30)
        plt.title("Clustering Pressure/Support of {}".format(target),fontsize=20)
        plt.ylabel("price")
        index_p=[]
        index_s=[]
        a= np.transpose(minmax)
        a= np.sort(a)
        for i in range(len(X)):
            colors = ['b','g','r','c','m','y','k','w']
            c = kmeans.predict(X[i].reshape(-1,1))[0]
            color = colors[c]
            if X[i] in a[1]:
                index_s.append(i)
            if X[i] in a[0]:
                index_p.append(i)
            plt.scatter(i,X[i],c = color,s = 20, marker= "o")
        for i in range(len(minmax)):
            plt.hlines(a[0][i], xmin= index_p[i]-10, xmax= index_p[i]+10, colors="red", linestyle="--")
            plt.text(index_p[i]-15, a[0][i], "Pressure= {:.2f}".format(a[0][i]), fontsize=13)
            plt.hlines(a[1][i], xmin= index_s[i]-10, xmax= index_s[i]+10, colors="b", linestyle="--")
            plt.text(index_s[i]-15, a[1][i], "Support= {:.2f}".format(a[1][i]), fontsize=13)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

def time_pattern():
    global target, daypara, df, df2, df_4pycaret, df_temp
    EPOCH = st.sidebar.slider("Epochs",100,1000)
    model = NeuralProphet(
        growth="linear",
        changepoints=None,
        n_changepoints=30,
        changepoints_range=0.95,
        trend_reg=0,
        trend_reg_threshold=False,
        yearly_seasonality="auto",
        weekly_seasonality=True,
        daily_seasonality="auto",
        seasonality_mode="additive",
        seasonality_reg=0,
        n_forecasts=30,
        n_lags=60,  ##determines autoregression 
        num_hidden_layers=0,
        d_hidden=None,
        ar_sparsity=None,
        learning_rate=None,
        epochs=EPOCH,
        loss_func="Huber",
        normalize="auto",
        impute_missing=True,
    )
    metrics = model.fit(df2, validate_each_epoch=True, freq="D")
    future = model.make_future_dataframe(df2, periods=252, n_historic_predictions=len(df2))
    with st.spinner("Training..."):
        forecast = model.predict(future)
        fig, ax = plt.subplots(1,2,figsize=(17,7))
        ax[0].plot(metrics["MAE"], 'ob', linewidth=6, label="Training Loss")  
        ax[0].plot(metrics["MAE_val"], '-r', linewidth=2, label="Validation Loss")
        ax[0].legend(loc='center right')
        ax[0].tick_params(axis='both', which='major')
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Loss")
        ax[0].set_title("Model Loss (MAE)")
        ax[1].plot(metrics["SmoothL1Loss"], 'ob', linewidth=6, label="Training Loss")  
        ax[1].plot(metrics["SmoothL1Loss_val"], '-r', linewidth=2, label="Validation Loss")
        ax[1].legend(loc='center right')
        ax[1].tick_params(axis='both', which='major')
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Loss")
        ax[1].set_title("Model Loss (SmoothL1Loss)")
        st.subheader("Loss Check")
        st.pyplot()
        with st.spinner("Recognizing Time Pattern"):
            st.subheader("Time Pattern")
            model.plot_parameters()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()

def highlight_out(s,col):
    is_out = pd.Series(data=False, index=s.index)
    is_out[col] = s.loc[col] == 1
    return ['background-color: yellow' if is_out.any() else '' for v in is_out]

def ad_pycaret():
    global target, daypara, df, df2, df_4pycaret, df_temp
    df_4pycaret = original_df_parse(df_4pycaret)
    df_ad = get_outliers(data = df_4pycaret)
    outliers = []
    for i in range(len(df_ad)):
        if(df_ad['Anomaly'][i]==1):
            outliers.append(df_ad['Date'][i])
    out_date = df_ad['Date'][df_ad['Anomaly']==1]
    out_price = df_ad['price'][df_ad['Anomaly']==1]
    plt.figure(figsize=(11, 5))
    plt.plot(df_ad["price"].index, df_ad['price'],label=target)
    plt.plot(out_price,marker = '^' ,markersize = 10, label = 'outliers')
    plt.ylabel('price')
    plt.title('Figure 2: {} price'.format(target))
    plt.legend()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.write(df_ad.style.apply(highlight_out, col = ["Anomaly"], axis=1))
    st.pyplot()

 

def main():
    global target, daypara, df, df2, df_4pycaret, df_temp
    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Data Browse","Feature Engineering","Clustering","Time Pattern","Anomaly Detection(Algo)","Anomaly Detection(DL)"])
    if app_mode == "Data Browse":
        st.title("Welcome")
        data_init(30,504)
        data_browse()
    elif app_mode == "Feature Engineering":
        data_init(30,504)
        feature_engineer()
    elif app_mode == "Clustering":
        st.title("Cluster to Find Support/Pressure")
        data_init(60,504)
        clustering()
    elif app_mode == "Time Pattern":
        data_init(150,504)
        time_pattern()
    elif app_mode == "Anomaly Detection(Algo)":
        data_init(150,504)
        ad_pycaret()
    elif app_mode == "Anomaly Detection(DL)":
        data_init(252,756)
        ad_dl()


def sidebar_func(min,max):
    portfolio = ["^TWII"]   #need sql
    target = st.sidebar.text_input('XXXX.tw or ^DJI')
    portfolio.append(target)
    daypara = st.sidebar.slider("Lookback",min,max)
    return target, portfolio, daypara

def original_df_parse(df):
    get_kd(df)
    get_technical_indicators(df)
    df = df.fillna(0)
    return df

def feature_df_parse(df,lookback):
    lookback = lookback*(-1)
    df_feature = pd.DataFrame(data = df)
    df_feature.pop('high')
    df_feature.pop('low')
    df_feature.pop('Date')
    #df_feature.pop('ma240')
    df_feature_parsed = df_feature[lookback:]
    return df_feature_parsed

def get_feature_importance_data(data_income):
    data = data_income.copy()
    y = data['price']
    X = data.iloc[:, 1:]
    train_samples = int(X.shape[0] * 0.65)
    X_train = X.iloc[:train_samples]
    X_test = X.iloc[train_samples:]
    y_train = y.iloc[:train_samples]
    y_test = y.iloc[train_samples:]
    return (X_train, y_train), (X_test, y_test)

def fetch_data(target,daypara):
    from pandas_datareader import data
    now = datetime.date.today()
    start = now - datetime.timedelta(days = int(daypara))
    data = data.DataReader(target, "yahoo", str(start), str(now))
    c = data["Close"]
    h = data['High']
    l = data["Low"]
    v = data["Volume"]
    a = c.index
    dit = {
        "Date" : a,
        "price": c, 
        "high" : h,
        "low" : l,
        'vol': v
        }
    df = pd.DataFrame(dit); df_4pycaret = pd.DataFrame(dit); df_temp = pd.DataFrame(dit)
    dit2 = {
        "ds": a,
        "y": c
        }
    df2 = pd.DataFrame(dit2)
    aa=a.astype("int64")
    dit3= {
        "timestamp": aa, 
        "price": c, 
        "high": h, 
        "low": l, 
        "vol": v
        }
    df3= pd.DataFrame(dit3); df3.index= range(len(df.index))
    return df, df2, df_4pycaret, df_temp

def get_technical_indicators(dataset):
    # Create 5 and 20 days Moving Average
    dataset['ma5'] = dataset['price'].rolling(window=5).mean()
    dataset['ma10'] = dataset['price'].rolling(window=10).mean()
    dataset['ma20'] = dataset['price'].rolling(window=20).mean()
    dataset['ma60'] = dataset['price'].rolling(window=60).mean()
    dataset['ma120'] = dataset['price'].rolling(window=120).mean()
    dataset['ma240'] = dataset['price'].rolling(window=240).mean()
    # Create MACD
    dataset['ema26'] = dataset["price"].ewm(span=26).mean()
    dataset['ema12'] = dataset["price"].ewm(span=12).mean()
    dataset['MACD'] = (dataset['ema12']-dataset['ema26'])
    # Create Bollinger Bands
    dataset['sd22'] = dataset['price'].rolling(22).std()
    dataset['upper_band'] = dataset['ma20'] + (dataset['sd22']*2)
    dataset['lower_band'] = dataset['ma20'] - (dataset['sd22']*2)

k = 0
def k_val(rsv):
  global k
  k = (2/3) * k + (1/3) * rsv
  return k

d = 0
def d_val(k):
  global d
  d = (2/3) * d + (1/3) * k

def get_kd(dataset):
    temp = pd.DataFrame(index = dataset.index)
    temp['window_max'] = dataset['high'].rolling("9D").max()
    temp['window_min'] = dataset['low'].rolling("9D").min()
    rsv = np.zeros(len(dataset.index))
    temp['rsv'] = 100 * (dataset['price'] - temp["window_min"]) / (temp["window_max"]-temp["window_min"])
    dataset['k_val'] = 0
    dataset['k_val'] = temp['rsv'].apply(k_val)
    dataset['d_val'] = 0
    dataset['d_val'] = dataset['k_val'].apply(k_val)

def login_process():
    password = "admin"
    block = st.sidebar.empty()
    entered = block.text_input('Password',type="password")
    global check
    check = False
    if entered == password:
        check = True
    elif entered:
        st.sidebar.info("Please enter a valid password")

###DEEPANT AD
MODEL_SELECTED = "deepant" # Possible Values ['deepant', 'lstmae']
#LOOKBACK_SIZE = 30
#THRESHOLD = 0.55
#EPOCH = 1000

def ad_dl():
    global target, daypara, df, df2, df_4pycaret, df_temp
    LOOKBACK_SIZE = st.sidebar.slider("Rolling Window Size",30,60)
    THRESHOLD = st.sidebar.slider("Threshold",0.55,0.8)
    EPOCH = st.sidebar.slider("Epoch",200,1000)
    data, _data = read_modulate_data(df_temp)
    X,Y,T = data_pre_processing(df_temp,LOOKBACK_SIZE)
    loss = compute(X,Y,LOOKBACK_SIZE,EPOCH)
    outlier_cnn, loss_df= threshold_ctrl(THRESHOLD, loss, T)
    #st.write(loss_df)
    st.write(loss_df.style.apply(highlight_out, col = ["anomaly"], axis=1))
    """
    Visualization 
    """
    plt.figure(figsize=(11,5))
    sns.set_style("darkgrid")

    ax = sns.distplot(loss_df["loss"],bins=100, label="Frequency")
    ax.set_title("Frequency Distribution | Kernel Density Estimation")
    ax.set(xlabel='Anomaly Confidence Score', ylabel='Frequency (sample)')
    plt.axvline(THRESHOLD, color="k", linestyle="--")
    plt.legend()
    st.pyplot()

    plt.figure(figsize=(11,5))
    ax = sns.lineplot(x="Date", y="loss", data=loss_df, color='g', label="Anomaly Score")
    ax.set_title("Anomaly Confidence Score ")
    ax.set(ylabel="Anomaly Confidence Score", xlabel="Date")
    plt.axhline(THRESHOLD, color="r", linestyle="--")
    plt.legend()
    st.pyplot()

    plt.figure(figsize=(11,5))
    plt.plot(df.index,df["price"])
    plt.plot(outlier_cnn, marker="^", markersize=10, label="Anomaly", color="g")
    plt.title("Anomaly in Price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot()

def data_pre_processing(df,LOOKBACK_SIZE):
    """
        Data pre-processing : Function to create data for Model
    """
    try:
        scaled_data = MinMaxScaler(feature_range = (0, 1))
        data_scaled_ = scaled_data.fit_transform(df)
        df.loc[:,:] = data_scaled_
        _data_ = df.to_numpy(copy=True)
        X = np.zeros(shape=(df.shape[0]-LOOKBACK_SIZE,LOOKBACK_SIZE,df.shape[1]))
        Y = np.zeros(shape=(df.shape[0]-LOOKBACK_SIZE,df.shape[1]))
        timesteps = []
        for i in range(LOOKBACK_SIZE-1, df.shape[0]-1):
            timesteps.append(df.index[i])
            Y[i-LOOKBACK_SIZE+1] = _data_[i+1]
            for j in range(i-LOOKBACK_SIZE+1, i+1):
                X[i-LOOKBACK_SIZE+1][LOOKBACK_SIZE-1-i+j] = _data_[j]
        return X,Y,timesteps
    except Exception as e:
        print("Error while performing data pre-processing : {0}".format(e))
        return None, None, None

class LSTMAE(torch.nn.Module):
    """
        Model : Class for LSTMAE model
    """
    def __init__(self, LOOKBACK_SIZE, DIMENSION):
        super(LSTMAE, self).__init__()
        self.lstm_1_layer = torch.nn.LSTM(DIMENSION, 128, 1)
        self.dropout_1_layer = torch.nn.Dropout(p=0.2)
        self.lstm_2_layer = torch.nn.LSTM(128, 64, 1)
        self.dropout_2_layer = torch.nn.Dropout(p=0.2)
        self.lstm_3_layer = torch.nn.LSTM(64, 64, 1)
        self.dropout_3_layer = torch.nn.Dropout(p=0.2)
        self.lstm_4_layer = torch.nn.LSTM(64, 128, 1)
        self.dropout_4_layer = torch.nn.Dropout(p=0.2)
        self.linear_layer = torch.nn.Linear(128, DIMENSION)
        
    def forward(self, x):
        x, (_,_) = self.lstm_1_layer(x)
        x = self.dropout_1_layer(x)
        x, (_,_) = self.lstm_2_layer(x)
        x = self.dropout_2_layer(x)
        x, (_,_) = self.lstm_3_layer(x)
        x = self.dropout_3_layer(x)
        x, (_,_) = self.lstm_4_layer(x)
        x = self.dropout_4_layer(x)
        return self.linear_layer(x)

class DeepAnT(torch.nn.Module):
    """
        Model : Class for DeepAnT model
    """
    def __init__(self, LOOKBACK_SIZE, DIMENSION):
        super(DeepAnT, self).__init__()
        self.conv1d_1_layer = torch.nn.Conv1d(in_channels=LOOKBACK_SIZE, out_channels=16, kernel_size=1)
        self.relu_1_layer = torch.nn.ReLU()
        self.maxpooling_1_layer = torch.nn.MaxPool1d(kernel_size=2)
        self.conv1d_2_layer = torch.nn.Conv1d(in_channels=16, out_channels=16, kernel_size=1)
        self.relu_2_layer = torch.nn.ReLU()
        self.maxpooling_2_layer = torch.nn.MaxPool1d(kernel_size=2)
        self.flatten_layer = torch.nn.Flatten()
        self.dense_1_layer = torch.nn.Linear(16, 40)
        self.relu_3_layer = torch.nn.ReLU()
        self.dropout_layer = torch.nn.Dropout(p=0.25)
        self.dense_2_layer = torch.nn.Linear(40, DIMENSION)
        
    def forward(self, x):
        x = self.conv1d_1_layer(x)
        x = self.relu_1_layer(x)
        x = self.maxpooling_1_layer(x)
        x = self.conv1d_2_layer(x)
        x = self.relu_2_layer(x)
        x = self.maxpooling_2_layer(x)
        x = self.flatten_layer(x)
        x = self.dense_1_layer(x)
        x = self.relu_3_layer(x)
        x = self.dropout_layer(x)
        return self.dense_2_layer(x)

def make_train_step(model, loss_fn, optimizer):
    """
        Computation : Function to make batch size data iterator
    """
    def train_step(x, y):
        model.train()
        yhat = model(x)
        loss = loss_fn(y, yhat)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()
    return train_step

def compute(X,Y,LOOKBACK_SIZE,EPOCH):
    """
        Computation : Find Anomaly using model based computation 
    """
    if str(MODEL_SELECTED) == "lstmae":
        model = LSTMAE(LOOKBACK_SIZE,26)
        criterion = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        train_data = torch.utils.data.TensorDataset(torch.tensor(X.astype(np.float32)), torch.tensor(X.astype(np.float32)))
        train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=32, shuffle=False)
        train_step = make_train_step(model, criterion, optimizer)
        for epoch in range(EPOCH):
            loss_sum = 0.0
            ctr = 0
            for x_batch, y_batch in train_loader:
                loss_train = train_step(x_batch, y_batch)
                loss_sum += loss_train
                ctr += 1
            print("Training Loss: {0} - Epoch: {1}".format(float(loss_sum/ctr), epoch+1))
        hypothesis = model(torch.tensor(X.astype(np.float32))).detach().numpy()
        loss = np.linalg.norm(hypothesis - X, axis=(1,2))
        return loss.reshape(len(loss),1)
    elif str(MODEL_SELECTED) == "deepant":
        model = DeepAnT(LOOKBACK_SIZE,4)
        criterion = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(list(model.parameters()), lr=1e-5)
        train_data = torch.utils.data.TensorDataset(torch.tensor(X.astype(np.float32)), torch.tensor(Y.astype(np.float32)))
        train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=32, shuffle=False)
        train_step = make_train_step(model, criterion, optimizer)
        for epoch in range(EPOCH):
            loss_sum = 0.0
            ctr = 0
            for x_batch, y_batch in train_loader:
                loss_train = train_step(x_batch, y_batch)
                loss_sum += loss_train
                ctr += 1
            print("Training Loss: {0} - Epoch: {1}".format(float(loss_sum/ctr), epoch+1))
        hypothesis = model(torch.tensor(X.astype(np.float32))).detach().numpy()
        loss = np.linalg.norm(hypothesis - Y, axis=1)
        return loss.reshape(len(loss),1)
    else:
        print("Selection of Model is not in the set")
        return None

def read_modulate_data(data):
    """
        Data ingestion : Function to read and formulate the data
    """
    #data = pd.read_csv(data_file)
    data.fillna(data.mean(), inplace=True)
    df = data.copy()
    data.set_index("Date", inplace=True)
    data.index = pd.to_datetime(data.index)
    return data, df

def threshold_ctrl(threshold, loss, T):
  """
      sensitivity of detection
  """
  loss_init=pd.DataFrame(loss,columns=["loss"])
  anomaly = [0]*len(loss)
  for i in range(len(loss)):
    if loss[i]>=threshold:
      anomaly[i]=1  

  dic_loss = {"Date":T, "loss":loss_init["loss"], "anomaly":anomaly}
  loss_df = pd.DataFrame(dic_loss)
  loss_df["loss"] = loss_df["loss"].astype(float)
  loss_df.index = loss_df["Date"]
  out_price_date = loss_df["Date"][loss_df["anomaly"]==1]
  anomaly_cnn = df["price"][out_price_date.index]
  return anomaly_cnn, loss_df


###

#if __name__ == "__main__":
login_process()
if check:
    main()