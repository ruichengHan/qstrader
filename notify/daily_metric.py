import datetime
import json
import os
import sys

import akshare as ak
import lark_oapi as lark
import numpy as np
import talib
from lark_oapi.api.im.v1 import *


def get_index_stock(code):
    # 拿历史
    df = ak.stock_zh_index_daily(symbol=code)
    df["date"] = df["date"].apply(lambda x: x.strftime("%Y-%m-%d"))
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    df = df[df["date"] >= "2020-01-01"]
    df = df[df["date"] < today]
    close = df["close"].to_numpy()

    # 拿最新的
    current = ak.stock_zh_index_spot_sina()
    current = current[current["代码"] == code]
    current_price = current["最新价"].to_numpy()

    return np.concatenate((close, current_price))


def run(code, mode):
    close = get_index_stock(code)
    rsi = talib.RSI(close, timeperiod=2)
    today_rsi = rsi[-1]
    cumulate_rsi = rsi[-1] + rsi[-2] + rsi[-3]
    diff = (close[-1] / close[-2] - 1) * 100
    output = "\n".join([code, "当天价格: <b>%.1f</b> (%s%.2f%s)" % (close[-1], "+" if diff > 0 else "-", abs(diff), "%"), "RSI2: <b>%.1lf</b>" % today_rsi,
                        "3d累积RSI2: <b>%.1lf</b>" % cumulate_rsi])
    # 如果是daily的，那就默认发一次
    if mode == "day":
        return output
    # 小时级的默认只有需要再发
    if mode == "hour" and (today_rsi < 5 or cumulate_rsi < 65):
        return output
    if mode == "sell" and today_rsi > 75:
        return output
    return ""


def send_msg(msg):
    # 创建client
    app_id = os.environ["lark_app_id"]
    secret = os.environ["lark_app_secret"]
    client = lark.Client.builder().app_id(app_id).app_secret(secret).log_level(lark.LogLevel.DEBUG).build()

    # 构造请求对象
    content = json.dumps({"text": msg})
    print(content)

    request: CreateMessageRequest = CreateMessageRequest.builder() \
        .receive_id_type("open_id") \
        .request_body(CreateMessageRequestBody.builder()
                      .receive_id("ou_75e27e5d4a680e3e2c27d732c8a0e9de")
                      .msg_type("text")
                      .content(content)
                      .build()).build()

    client.im.v1.message.create(request)


if __name__ == '__main__':
    mode = sys.argv[1]
    codes = ["sh000300"]
    out = [run(code, mode) for code in codes]
    msg = "\n".join(out)
    if len(msg) > 10:
        send_msg(msg)
