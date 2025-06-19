import os
import pandas as pd
from pandas import DataFrame
import talib
import akshare as ak
from lightgbm import LGBMRanker


features = set([])


def kbars(df: DataFrame, features: list):
    df["KMID"] = (df["close"] - df["open"]) / df["open"]
    df["KLEN"] = (df["high"] - df["low"]) / df["open"]
    df["KMID2"] = (df["close"] - df["open"]) / (df["high"] - df["low"]+0.0001)
    df["KUP"] = (df['high'] - df[["close", "open"]].max(axis=1)) / df["open"]
    df["KUP2"] = (df['high'] - df[["close", "open"]].max(axis=1)) / (df["high"] - df["low"]+0.0001)
    df["KLOW"] = (df[["close", "open"]].min(axis=1) - df["low"]) / df["open"]
    df["KLOW2"] = (df[["close", "open"]].min(axis=1) - df["low"]) / (df["high"] - df["low"]+0.0001)
    df["KSFT"] = (2 * df["close"] - df["high"] - df["low"]) / df["open"]
    df["KSFT2"] = (2 * df["close"] - df["high"] - df["low"]) / (df["high"] - df["low"]+0.0001)
    features.add("KMID")
    features.add("KLEN")
    features.add("KMID2")
    features.add("KUP")
    features.add("KUP2")
    features.add("KLOW")
    features.add("KLOW2")
    features.add("KSFT")
    features.add("KSFT2")
    return df

def cal_code_feature(df: DataFrame, features: list):
    df = kbars(df, features)
    df = cal_windows_feature(df, [3, 5, 10, 20, 40], features)
    return df

def cal_windows_feature(df: DataFrame, windows: list, features: list):
    
    for w in windows:
        df["ROC_" + str(w)] = talib.ROC(df['close'], timeperiod=w)
        df["SMA_" + str(w)] = talib.SMA(df['close'], timeperiod=w)        
        df["RSI_" + str(w)] = talib.RSI(df['close'], timeperiod=w) / 100        
        df["STD_" + str(w)] = talib.STDDEV(df['close'], timeperiod=w)        
        df["SLOPE_" + str(w)] = talib.LINEARREG_SLOPE(df['close'], timeperiod=w)
        
        features.add("ROC_" + str(w))
        features.add("SMA_" + str(w))
        features.add("RSI_" + str(w))
        features.add("STD_" + str(w))
        features.add("SLOPE_" + str(w))
    
    return df


def cal_group_feature(df: DataFrame, features: list):
    # 将code列的数据类型转换为int
    df['code'] = df['code'].astype(int)

    # 基于date分组，计算RSI2的rank百分比
    df['RSI_3_rank'] = df.groupby('date')['RSI_3'].rank(pct=True)
    # 基于date分组，计算RSI5的rank百分比
    df['RSI_5_rank'] = df.groupby('date')['RSI_5'].rank(pct=True)
    # 基于date分组，计算RSI10的rank百分比
    df['RSI_10_rank'] = df.groupby('date')['RSI_10'].rank(pct=True)
    features.add("RSI_3_rank")
    features.add("RSI_5_rank")
    features.add("RSI_10_rank")


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
    print(f"features: {features}")
    
    train_features = list(features)
    model.fit(df[train_features], df['label'], group=df.groupby('date').size().values)
    print("finish to train model")
    return model

def read_data(csv_dir: str, features: list, symbols: list):
    dataframes = {}
    for symbol in symbols:
        filename = symbol + ".csv"
        if filename.endswith('.csv'):
            file_path = os.path.join(csv_dir, filename)
            df = pd.read_csv(file_path)
            df['code'] = filename[:-4]
            df['code'] = df['code'].astype(str)
            base_features = ['open', 'high', 'low', 'close', 'volume']
            window = [3, 5, 10]
            for f in base_features:
                for w in window:
                    df[f"{f}_{w}"] = df[f].shift(w)
            
            df = df[df["date"] < "2024-12-20"]
            df = df[df["date"] > "2016-02-01"]

            cal_code_feature(df, features)
            df = df.drop("Unnamed: 0", axis=1)

            dataframes[filename] = df
            print(f"成功读取文件: {filename}, 形状: {df.shape}")
    # 将所有DataFrame进行union操作
    all_data = pd.concat(dataframes.values(), ignore_index=True)
    return all_data

def predict_model(model, df: DataFrame):
    predict_df = df
    predict_df['prediction'] = model.predict(predict_df[list(features)])
    return predict_df

def get_index_stock(index_code="000300", year='20218'):
    index_stock_cons_csindex_df = ak.index_stock_cons(symbol=index_code)
    filter_df = index_stock_cons_csindex_df[index_stock_cons_csindex_df["纳入日期"] < str(year)]
    out = []
    for code in filter_df["品种代码"].to_list():
        out.append(code)
    return out

def get_symbols():
    stock_list = get_index_stock()
    out = []
    csv_dir = '/Users/rui.chengcr/PycharmProjects/qstrader/qs_data/price/'
    for root, dirs, files in os.walk(csv_dir):
        for file_name in files:
            out.append(file_name.split(".")[0])

    out = list(filter(lambda x: x in stock_list, out))
    return out

if __name__ == '__main__':
    csv_dir = '/Users/rui.chengcr/PycharmProjects/qstrader/qs_data/price/'

    symbols = get_symbols()
    all_data = read_data(csv_dir, features, symbols)

    cal_group_feature(all_data, features)


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
    
    # 保存筛选后的数据
    
    top_predictions.to_csv('top_predictions.csv', index=False)
    print("已保存筛选后的数据到 top_predictions.csv")


