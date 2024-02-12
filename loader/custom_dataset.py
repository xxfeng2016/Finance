import sys
import os

# sys.path.append(os.path.dirname(os.path.abspath(os.path.join(os.getcwd(), "AutoTrading"))))

from torch.utils.data import Dataset
import torch

from glob import glob
import pandas as pd
import numpy as np
######################
from plotly.subplots import make_subplots
import plotly.graph_objects as go
######################
import scipy.stats as stats
from sklearn.preprocessing import RobustScaler, power_transform
# from sklearn.decomposition import PCA
from datetime import datetime, time

np.set_printoptions(floatmode="fixed", precision=8, suppress=True)
torch.set_printoptions(sci_mode=False)

class DS(Dataset):

    def __init__(self, data_paths = None, window_size=(1, 0), sliding_size=(0, 30), target_size=(0, 10)):
        if data_paths == [] or data_paths is None:
            raise ValueError("No data paths given")
        
        self.data_paths = data_paths
        
        self.window_size = pd.Timedelta(minutes=window_size[0], seconds=window_size[1])
        self.sliding_size = pd.Timedelta(minutes=sliding_size[0], seconds=sliding_size[1])
        self.target_size = pd.Timedelta(minutes=target_size[0], seconds=target_size[1])
        
        self.df = pd.read_parquet(self.data_paths.pop(0))
        self.start_idx = 0
        self.start_time = self.df['STCK_CNTG_HOUR'].iloc[0]
        self.date_list = self.df['STCK_CNTG_HOUR'].dt.date.unique()
        self.idx = 0
        self.total_idx = 0
    
    def __len__(self):
        return len(self.df) + sum([len(pd.read_parquet(path)) for path in self.data_paths])

    def calc_idx(self, df_idx):
        return self.total_idx + df_idx
    
    def reset(self):
        try:
            self.start_idx = 0
            self.total_idx += self.idx
            self.df = pd.read_parquet(self.data_paths.pop(0))
            self.start_time = self.df['STCK_CNTG_HOUR'].iloc[0]
            self.date_list = self.df['STCK_CNTG_HOUR'].dt.date.unique()
        except IndexError:
            raise StopIteration

    def __getitem__(self, idx):
        if self.start_time.time() < time(9, 0, 0):
            end_time = datetime.combine(self.start_time.date(), time(9, 0, 0))
            
            x = self.df[(self.df['STCK_CNTG_HOUR'] >= self.start_time) & (self.df['STCK_CNTG_HOUR'] < end_time)]
            y = self.df[(self.df['STCK_CNTG_HOUR'] >= end_time) & (self.df["STCK_CNTG_HOUR"] <= end_time + self.target_size)]
            
            # 다음 슬라이딩 시작 시간으로 업데이트
            self.start_time = end_time
            # 다음 슬라이딩 윈도우의 시작 인덱스 업데이트
            self.start_idx = self.df.index[self.df['STCK_CNTG_HOUR'] >= self.start_time].min()
        
        elif self.start_time >= datetime.combine(self.start_time.date(), time(15, 20, 0)) - self.target_size: # 
            try:
                end_time = datetime.combine(self.date_list[np.where(self.date_list > self.start_time.date())][0], time(9, 0, 0))
            except IndexError:
                self.idx += len(self.df[(self.df['STCK_CNTG_HOUR'] >= self.start_time)])
                
                self.reset()
                return self.__getitem__(idx)
            
            except StopIteration:
                raise StopIteration
            
            x = self.df[(self.df['STCK_CNTG_HOUR'] >= self.start_time) & (self.df['STCK_CNTG_HOUR'] < end_time)]
            y = self.df[(self.df['STCK_CNTG_HOUR'] >= end_time) & (self.df["STCK_CNTG_HOUR"] <= end_time + self.target_size)]
            
            # 다음 슬라이딩 시작 시간으로 업데이트
            self.start_time = end_time
            # 다음 슬라이딩 윈도우의 시작 인덱스 업데이트
            self.start_idx = self.df.index[self.df['STCK_CNTG_HOUR'] >= self.start_time].min()

        else:
            end_time = self.start_time + self.window_size
            x = self.df[(self.df['STCK_CNTG_HOUR'] >= self.start_time) & (self.df['STCK_CNTG_HOUR'] < end_time)]
            y = self.df[(self.df['STCK_CNTG_HOUR'] >= end_time) & (self.df['STCK_CNTG_HOUR'] <= end_time + self.target_size)]
            
            # 다음 슬라이딩 시작 시간으로 업데이트
            self.start_time += self.sliding_size
            # 다음 슬라이딩 윈도우의 시작 인덱스 업데이트
            self.start_idx = self.df.index[self.df['STCK_CNTG_HOUR'] >= self.start_time].min()
            
        if x.empty:
            return self.__getitem__(idx)
        self.idx = self.calc_idx(x.index[-1])
        return self.idx, x, y

class collate_func():
    def __init__(self, is_train):
        self.is_train = is_train
        self.scaler = RobustScaler()

        ##################
        self.fig = make_subplots(rows=3, cols=4)
        ##################
        
    def __call__(self, batch):
        batch_x, batch_y = [], []
        max_length = max(len(x) for idx, x, y in batch)
        
        for idx, x, y in batch:
            # 위치 인코딩을 위한 시간 차이 배열
            fy = self.featuring_y(x, y)
            fx = self.featuring_x(x)
            
            padded_x = np.pad(fx, ((0, max_length - fx.shape[0]),(0, 0)), mode="constant", constant_values=99.)
            
            batch_x.append(padded_x)
            batch_y.append(fy)
            
        return idx, torch.tensor(np.array(batch_x), dtype=torch.float16, requires_grad=False), torch.tensor(np.array(batch_y), dtype=torch.float16, requires_grad=False)
    
    def featuring_x(self, x):
        if len(x) < 2:
            return np.column_stack([[0], [0], [0], x["CCLD_DVSN"].values]).astype(np.float16)
        volume = x["CNTG_VOL"].values
        amount = x["STCK_PRPR"].values * x["CNTG_VOL"].values
        t_diffs = self.calc_tdiff(x["STCK_CNTG_HOUR"].values[0], x["STCK_CNTG_HOUR"].values)
        trade_type = x["CCLD_DVSN"].values
        
        # rob_x = self.scaler.fit_transform(np.column_stack([volume, amount, t_diffs]))
        boxcox_x = np.column_stack([stats.boxcox(volume + 1e-9)[0], stats.boxcox(amount + 1e-9)[0], stats.boxcox(t_diffs)[0]])
        # pt = power_transform(np.column_stack([volume, amount, t_diffs]), method="")
        
        # 시각화
        # self.fig.add_trace(go.Scatter(x=x.index, y=volume, mode="lines", name="거래량"),row=1, col=1)
        # self.fig.add_trace(go.Scatter(x=x.index, y=amount, mode="lines", name="거래대금"),row=2, col=1)
        # self.fig.add_trace(go.Scatter(x=x.index, y=t_diffs, mode="lines", name="시간 차이"),row=3, col=1)
        # self.fig.add_trace(go.Scatter(x=x.index, y=rob_x[:, 0], mode="lines", name="Robust 거래량"),row=1, col=2)
        # self.fig.add_trace(go.Scatter(x=x.index, y=rob_x[:, 1], mode="lines", name="Robust 거래대금"),row=2, col=2)
        # self.fig.add_trace(go.Scatter(x=x.index, y=rob_x[:, 2], mode="lines", name="Robust 시간 차이"),row=3, col=2)
        # self.fig.add_trace(go.Scatter(x=x.index, y=boxcox_x[:, 0], mode="lines", name="BoxCox 거래량"),row=1, col=3)
        # self.fig.add_trace(go.Scatter(x=x.index, y=boxcox_x[:, 1], mode="lines", name="BoxCox 거래대금"),row=2, col=3)
        # self.fig.add_trace(go.Scatter(x=x.index, y=boxcox_x[:, 2], mode="lines", name="BoxCox 시간 차이"),row=3, col=3)
        # self.fig.add_trace(go.Scatter(x=x.index, y=pt[:, 0], mode="lines", name="Yeo Johnson 거래량"),row=1, col=3)
        # self.fig.add_trace(go.Scatter(x=x.index, y=pt[:, 1], mode="lines", name="Yeo Johnson 거래대금"),row=2, col=3)
        # self.fig.add_trace(go.Scatter(x=x.index, y=pt[:, 2], mode="lines", name="Yeo Johnson 시간 차이"),row=3, col=3)
        # self.fig.show()

        return np.column_stack([boxcox_x, trade_type]) # np.column_stack([t_diffs, volume, amount, trade_type])
    
    def featuring_y(self, x, y):
        try:    
            y_max = y["STCK_PRPR"].idxmax(axis=0, skipna=True) # target_size 내 최고가 index
            y_vol = (y.loc[y_max]["STCK_PRPR"] - x.iloc[-1]["STCK_PRPR"]) / x.iloc[-1]["STCK_PRPR"] * 100 # 변동률 계산
            y_t = self.calc_tdiff(x.iloc[-1]["STCK_CNTG_HOUR"], y.loc[y_max]["STCK_CNTG_HOUR"]) # input의 마지막 시간과 최고점까지의 시간 차이 계산
        except ValueError:
            return np.array([0, 0], dtype=np.float16)
        
        return np.array([y_vol, y_t], dtype=np.float16)
    
    def calc_tdiff(self, base, target):
        e = 10**-9 # unix 시간은 나노초 단위이므로 1초 단위로 바꿔주기위해 곱해준다.
        try:
            result = ((target - base).total_seconds())
        except AttributeError:
            result = ((target - base) * e).astype(np.float32)
        
        if type(result) == np.ndarray:
            # 순서대로 n ms 초 가중합
            result += np.arange(start=1, stop=len(result)+1, step=1, dtype=np.float32) * (10 ** -3)
                
        return result
    
    # def get_sinusoid_encoding_table(self, t_diffs):
    #     """ 
    #         https://paul-hyun.github.io/transformer-01/#:~:text=Position%20encoding
    #     """
    #     d_model = 1
    #     def cal_angle(position, i_hidn):
    #         return position / np.power(10000, 2 * (i_hidn // 2) / d_model) # 10000이라는 값은 트랜스포머 모델을 만든 사람들이 실험적으로 가장 성능이 좋았다고 보고한 값. 수학적인 근거는 없음
    #     def get_pos_angle_vec(position):
    #         return [cal_angle(position, i_hidn) for i_hidn in range(d_model)]
        
    #     sinusoid_table = np.array([get_pos_angle_vec(int(t_diff)) for t_diff in t_diffs])
    #     sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # even index sin 
    #     sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # odd index cos
        
    #     # plt.pcolormesh(sinusoid_table, cmap="RdBu")
    #     # plt.xlabel("Depth")
    #     # plt.xlim((0, d_model))
    #     # plt.ylabel("Position")
    #     # plt.colorbar()
    #     # plt.show()

    #     return sinusoid_table.flatten().astype(np.float16)
    

    
    # def calc_time_diff(self, t_array):
    #     t_diffs = (t_array - t_array[0]) * 10 ** -9
    #     t_diffs = t_diffs.astype(np.float32)
        
    #     even = t_diffs % 2 == 0
        
    #     pos = np.where(even, np.sin(t_diffs), np.cos(t_diffs))
    #     return pos