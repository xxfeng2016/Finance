import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.join(os.getcwd(), "AutoTrading"))))

from torch.utils.data import Dataset
import torch

from glob import glob
import pandas as pd
from tqdm import tqdm
import numpy as np
import scipy.stats as stats
from sklearn.decomposition import PCA

from train_conf import encoding_map, pad_idx, device, map_size
from datetime import datetime, time
import matplotlib.pyplot as plt
np.set_printoptions(floatmode="fixed", precision=8, suppress=True)

class DS(Dataset):

    def __init__(self, window_size=(1, 0), sliding_size=(0, 30), target_window=(0, 10), data_paths = None):
        self.data_paths = data_paths
        self.current_file_idx = 0
        
        self.window_size = pd.Timedelta(minutes=window_size[0], seconds=window_size[1])
        self.sliding_size = pd.Timedelta(minutes=sliding_size[0], seconds=sliding_size[1])
        self.target_window = pd.Timedelta(minutes=target_window[0], seconds=target_window[1])
        
        self.df = pd.read_parquet(data_paths[self.current_file_idx])
        self.start_idx = 0
        self.start_time = None
        self.current_batch = 0
    
    def __len__(self):
        return len(self.df)

    def reset(self):
        if self.current_file_idx < len(self.data_paths) - 3: # 마지막 3개는 validation을 위함
            self.df = pd.read_parquet(self.data_paths[self.current_file_idx])
            self.current_file_idx += 1
            self.start_time = self.df['STCK_CNTG_HOUR'].iloc[0]
        else:
            return StopIteration

    def __getitem__(self, idx):
        self.start_time = self.df['STCK_CNTG_HOUR'].iloc[self.start_idx]
        

        # if end_time.date() not in self.df['STCK_CNTG_HOUR'].dt.date.unique():
        #     self.reset()
        #     self.start_time = self.df['STCK_CNTG_HOUR'].iloc[self.start_idx]
        #     end_time = self.start_time + self.window_size

        if self.start_time.time() < time(9, 0, 0):
            end_time = datetime.combine(self.start_time.date(), time(9, 0, 0))
            
            x = self.df[(self.df['STCK_CNTG_HOUR'] >= self.start_time) & (self.df['STCK_CNTG_HOUR'] < end_time)]
            y = self.df[(self.df['STCK_CNTG_HOUR'] > end_time) & (self.df["STCK_CNTG_HOUR"] <= end_time + self.sliding_size)]
            
            # 다음 슬라이딩 시작 시간으로 업데이트
            self.start_time = end_time
            # 다음 슬라이딩 윈도우의 시작 인덱스 업데이트
            self.start_idx = self.df.index[self.df['STCK_CNTG_HOUR'] >= self.start_time].min()
        
        elif self.start_time >= datetime.combine(self.start_time.date(), time(15, 20, 0)) - self.target_window: # 
            end_time = datetime.combine(self.start_time.date() + pd.Timedelta(days=1), time(9, 0, 0))
            
            x = self.df[(self.df['STCK_CNTG_HOUR'] >= self.start_time) & (self.df['STCK_CNTG_HOUR'] < end_time)]
            y = self.df[(self.df['STCK_CNTG_HOUR'] > end_time) & (self.df["STCK_CNTG_HOUR"] <= end_time + self.sliding_size)]
            
            # 다음 슬라이딩 시작 시간으로 업데이트
            self.start_time = end_time
            # 다음 슬라이딩 윈도우의 시작 인덱스 업데이트
            self.start_idx = self.df.index[self.df['STCK_CNTG_HOUR'] >= self.start_time].min()

        else:
            end_time = self.start_time + self.window_size
            x = self.df[(self.df['STCK_CNTG_HOUR'] >= self.start_time) & (self.df['STCK_CNTG_HOUR'] < end_time)]
            y = self.df[(self.df['STCK_CNTG_HOUR'] >= x['STCK_CNTG_HOUR'].max() + self.target_window) & (self.df['STCK_CNTG_HOUR'] <= x['STCK_CNTG_HOUR'].max() + self.target_window)]
            
            # 다음 슬라이딩 시작 시간으로 업데이트
            self.start_time += self.sliding_size
            # 다음 슬라이딩 윈도우의 시작 인덱스 업데이트
            self.start_idx = self.df.index[self.df['STCK_CNTG_HOUR'] >= self.start_time].min()
                        
        
        if x.empty:
            return self.__getitem__()
            
        # 데이터셋 한 개가 끝난 경우
        return x, y

class collate_func():
    def __init__(self, is_train):
        self.is_train = is_train
        
    def __call__(self, batch):
        batch_x, batch_y = [], []
        max_length = max(len(x) for x, y in batch)
        for x, y in batch:
            # 데이터 추출
            vol = []
            amount = []
            
            # 위치 인코딩을 위한 시간 차이 배열
            t_diffs = self.get_sinusoid_encoding_table(x["STCK_CNTG_HOUR"].values)

            # 정규화 및 스케일링
            vol = self.std_boxcox(x["CNTG_VOL"].values)
            amount = self.std_boxcox(x["STCK_PRPR"].values * x["CNTG_VOL"].values)
            trade_type = x["CCLD_DVSN"].values
            
            pca = PCA(n_components=1).fit_transform(np.column_stack([vol, amount, trade_type])).flatten().astype(np.float16)
            pos_enc_pca = pca + t_diffs
            
            padded_x = np.pad(pos_enc_pca, (0, max_length - len(pos_enc_pca)), mode="constant", constant_values=999.)
            
            # batch_x.append(padded_x.reshape(-1,1))
            batch_x.append(padded_x)
            batch_y.append(self.calc_sharp(y["STCK_PRPR"].values))
            
        return torch.tensor(batch_x, dtype=torch.float16, requires_grad=False), torch.tensor(batch_y, dtype=torch.float16, requires_grad=False)
    
    def get_sinusoid_encoding_table(self, t_diffs):
        """ 
            https://paul-hyun.github.io/transformer-01/#:~:text=Position%20encoding
        """
        d_model = 1
        def cal_angle(position, i_hidn):
            return position / np.power(10000, 2 * (i_hidn // 2) / d_model) # 10000이라는 값은 트랜스포머 모델을 만든 사람들이 실험적으로 가장 성능이 좋았다고 보고한 값. 수학적인 근거는 없음
        def get_posi_angle_vec(position):
            return [cal_angle(position, i_hidn) for i_hidn in range(d_model)]
        
        sinusoid_table = np.array([get_posi_angle_vec(int(t_diff)) for t_diff in t_diffs])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # even index sin 
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # odd index cos
        
        # plt.pcolormesh(sinusoid_table, cmap="RdBu")
        # plt.xlabel("Depth")
        # plt.xlim((0, d_model))
        # plt.ylabel("Position")
        # plt.colorbar()
        # plt.show()

        return sinusoid_table.flatten().astype(np.float16)
    
    def calc_time_diff(self, t_array):
        t_diffs = (t_array - t_array[0]) * 10 ** -9
        t_diffs = t_diffs.astype(np.float32)
        
        even = t_diffs % 2 == 0
        
        pos = np.where(even, np.sin(t_diffs), np.cos(t_diffs))
        return pos
    
    def std_boxcox(self, x):
        return stats.boxcox(x)[0]
        
    def calc_sharp(self, target_window):
        """
            :param returns: 
            :return: 샤프 비율
        """
        # 첫 번째 최소값과 최대값의 인덱스 찾기
        if target_window.size < 1: return 0.
        
        min_index = np.where(target_window == np.min(target_window))[0][0]  # 최소값의 첫 번째 인덱스
        max_index = np.where(target_window == np.max(target_window))[0][0]  # 최대값의 첫 번째 인덱스
        
        # 최소값과 최대값
        min_value = target_window[min_index]
        max_value = target_window[max_index]
        
        # 가능한 수익률
        rate = (max_value - min_value) / min_value
        
        # 최소값이 최대값보다 뒤에 있으면 변화율을 음수로 만듦
        if min_index > max_index:
            rate *= -1
        
        # 주가 변동성 계산
        volatility = ((target_window[1:] / target_window[:-1]) - 1)
        
        if volatility.size == 0:
            return 0.
        
        # 변동률의 표준편차 계산
        std_dev = np.std(volatility)
        
        if std_dev == 0:
            return 0.
        
        # 샤프 비율 계산
        sharpe_ratio = rate / std_dev
        if np.isnan(sharpe_ratio) or np.isinf(sharpe_ratio):
            sharpe_ratio = 0.
        return sharpe_ratio 
    #0.01