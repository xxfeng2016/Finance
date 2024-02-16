import torch
from torch.utils.data import IterableDataset, DataLoader
import pandas as pd
import numpy as np
from datetime import datetime, time

class ParquetFileDataset(IterableDataset):
    def __init__(self, file_path):
        super(ParquetFileDataset, self).__init__()
        
        self.file_path = file_path
        self.df = pd.read_parquet(self.file_path)
        self.current_time = self.df['STCK_CNTG_HOUR'].iloc[0]
        self.current_idx = 0
        self.date_list = self.df['STCK_CNTG_HOUR'].dt.date.unique()
        
    def __iter__(self):
        while True:
            if self.current_time < datetime.combine(self.current_time.date(), time(9, 0, 0)):
                end_time = datetime.combine(self.current_time.date(), time(9, 0, 0))

                x = self.df[(self.df['STCK_CNTG_HOUR'] >= self.current_time) & (self.df['STCK_CNTG_HOUR'] < end_time)]
                y = self.df[(self.df['STCK_CNTG_HOUR'] >= end_time) & (self.df["STCK_CNTG_HOUR"] <= end_time + self.target_size)]

                # 다음 슬라이딩 시작 시간으로 업데이트
                self.current_time = end_time
                # 다음 슬라이딩 윈도우의 시작 인덱스 업데이트
                self.current_idx = self.df.index[self.df['STCK_CNTG_HOUR'] >= self.current_time].min()

            elif self.current_time >= datetime.combine(self.current_time.date(), time(15, 20, 0)) - self.target_size: # 
                try:
                    end_time = datetime.combine(self.date_list[np.where(self.date_list > self.current_time.date())][0], time(9, 0, 0))
                except IndexError:
                    self.idx += len(self.df[(self.df['STCK_CNTG_HOUR'] >= self.current_time)])

                except StopIteration:
                    raise StopIteration

                x = self.df[(self.df['STCK_CNTG_HOUR'] >= self.current_time) & (self.df['STCK_CNTG_HOUR'] < end_time)]
                y = self.df[(self.df['STCK_CNTG_HOUR'] >= end_time) & (self.df["STCK_CNTG_HOUR"] <= end_time + self.target_size)]

                # 다음 슬라이딩 시작 시간으로 업데이트
                self.current_time = end_time
                # 다음 슬라이딩 윈도우의 시작 인덱스 업데이트
                self.current_idx = self.df.index[self.df['STCK_CNTG_HOUR'] >= self.current_time].min()

            else:
                end_time = self.current_time + self.window_size
                x = self.df[(self.df['STCK_CNTG_HOUR'] >= self.current_time) & (self.df['STCK_CNTG_HOUR'] < end_time)]
                y = self.df[(self.df['STCK_CNTG_HOUR'] >= end_time) & (self.df['STCK_CNTG_HOUR'] <= end_time + self.target_size)]

                # 다음 슬라이딩 시작 시간으로 업데이트
                self.current_time += self.sliding_size
                # 다음 슬라이딩 윈도우의 시작 인덱스 업데이트
                self.current_idx = self.df.index[self.df['STCK_CNTG_HOUR'] >= self.current_time].min()

            if x.empty:
                return self.__iter__()
            
            yield x, y

       
class ParquetIterableDataset(IterableDataset):
    def __init__(self, file_paths):
        super(ParquetIterableDataset, self).__init__()
        self.file_paths = file_paths

    def __iter__(self):
        # 파일 경로 리스트를 순회하며 각 파일에 대한 서브 데이터셋을 생성
        for file_path in self.file_paths:
            dataset = ParquetFileDataset(file_path)
            
            for data in dataset:
                yield data
from glob import glob
import os

file_paths = glob(os.path.join("data", "raw", "*.parquet")) # 파일 경로 리스트

# 메인 IterableDataset 인스턴스 생성
dataset = ParquetIterableDataset(file_paths=file_paths)

# DataLoader 인스턴스 생성
loader = DataLoader(dataset, batch_size=1)  # 적절한 배치 사이즈 설정

# 데이터 로딩 및 처리
for data in loader:
    print(data)