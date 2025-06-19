import os
import pandas as pd
from pandas import DataFrame
import talib

from lightgbm import LGBMRanker

def cal_code_feature(df: DataFrame):
    # 计算RSI指标
    df['RSI2'] = talib.RSI(df['close'], timeperiod=2) / 100
    df['RSI5'] = talib.RSI(df['close'], timeperiod=5) / 100
    df['RSI10'] = talib.RSI(df['close'], timeperiod=10) / 100

def cal_group_feature(df: DataFrame):
    # 将code列的数据类型转换为int
    df['code'] = df['code'].astype(int)

    # 基于date分组，计算ret5的rank百分比
    df['rev5_rank'] = df.groupby('date')['rev5'].rank(pct=True)
    # 基于date分组，计算RSI2的rank百分比
    df['RSI2_rank'] = df.groupby('date')['RSI2'].rank(pct=True)
    # 基于date分组，计算RSI5的rank百分比
    df['RSI5_rank'] = df.groupby('date')['RSI5'].rank(pct=True)
    # 基于date分组，计算RSI10的rank百分比
    df['RSI10_rank'] = df.groupby('date')['RSI10'].rank(pct=True)


def cal_label(df: DataFrame):

    # 基于date分组，计算ret5的rank，从0开始
    df['ret5'] = (df['close'].shift(-5) - df['close']) / df['close']
    df = df.dropna(subset=['ret5'])
    df['label'] = df.groupby('date')['ret5'].rank(pct=True)
    df['label'] = (df['label'] * 16).astype(int)

    # 基于date列对DataFrame进行排序
    df = df.sort_values('date')
    return df


def train_model(df: DataFrame):
    print("start to train model")
    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",        
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5
    )
    features = ['RSI2_rank', 'RSI5_rank', 'RSI10_rank', 'rev5_rank', 'RSI2', 'RSI5', 'RSI10']
    model.fit(df[features], df['label'], group=df.groupby('date').size().values)
    return model

def read_data(csv_dir: str):
    dataframes = {}
    for filename in os.listdir(csv_dir):
        if 'sh' in filename or 'FXP' in filename:
            continue
        if filename.endswith('.csv'):
            file_path = os.path.join(csv_dir, filename)
            try:
                df = pd.read_csv(file_path)
                df['code'] = filename[:-4]
                df['code'] = df['code'].astype(str)
                df['rev5'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
                
                
                df = df[df["date"] < "2024-12-20"]
                df = df[df["date"] > "2016-02-01"]

                cal_code_feature(df)
                df = df.drop("Unnamed: 0", axis=1)

                dataframes[filename] = df
                print(f"成功读取文件: {filename}, 形状: {df.shape}")
            except Exception as e:
                print(f"读取文件 {filename} 时出错: {e}")
    # 将所有DataFrame进行union操作
    all_data = pd.concat(dataframes.values(), ignore_index=True)
    return all_data

def predict_model(model, df: DataFrame):
    predict_df = df
    features = ['RSI2_rank', 'RSI5_rank', 'RSI10_rank', 'rev5_rank', 'RSI2', 'RSI5', 'RSI10']
    predict_df['prediction'] = model.predict(predict_df[features])
    return predict_df

if __name__ == '__main__':
    csv_dir = '/Users/rui.chengcr/PycharmProjects/qstrader/qs_data/price/'

    all_data = read_data(csv_dir)

    cal_group_feature(all_data)

    all_data = cal_label(all_data)

    # 对所有浮点数列保留4位小数
    float_columns = all_data.select_dtypes(include=['float64', 'float32']).columns
    all_data[float_columns] = all_data[float_columns].round(4)

    # 取2021年以前的数据作为训练集
    train_data = all_data[all_data['date'] < '2021-01-01'].copy()
    print(f"训练集日期范围: {train_data['date'].min()} 到 {train_data['date'].max()}")
    
    # 使用训练集训练模型
    model = train_model(train_data)
    
    # 取2021年以后的数据作为测试集
    test_data = all_data[all_data['date'] >= '2021-01-01'].copy()
    print(f"测试集日期范围: {test_data['date'].min()} 到 {test_data['date'].max()}")

    # 对测试集进行预测
    predict_data = predict_model(model, test_data)
    print(f"预测集日期范围: {predict_data['date'].min()} 到 {predict_data['date'].max()}")

    # 按日期分组，每组取prediction最大的30行
    top_predictions = predict_data.groupby('date').apply(
        lambda x: x.nlargest(20, 'prediction')
    ).reset_index(drop=True)[['date', 'code', 'prediction']]
    
    print(f"筛选后的数据形状: {top_predictions.shape}")
    print(f"筛选后的日期范围: {top_predictions['date'].min()} 到 {top_predictions['date'].max()}")
    print(f"每日股票数量: {top_predictions.groupby('date').size().describe()}")
    
    # 保存筛选后的数据
    
    top_predictions.to_csv('top_predictions.csv', index=False)
    print("已保存筛选后的数据到 top_predictions.csv")


