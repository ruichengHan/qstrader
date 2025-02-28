import json
import os

import akshare as ak
import lark_oapi as lark
import talib
from lark_oapi.api.im.v1 import *


def get_index_stock(code):
    df = ak.stock_zh_index_daily(symbol=code)
    df["date"] = df["date"].apply(lambda x: x.strftime("%Y-%m-%d"))
    df = df[df["date"] >= "2016-01-01"]
    return df


def run(code):
    df = get_index_stock(code)
    date = df["date"].to_numpy()[-1]
    close = df["close"].to_numpy()
    rsi = talib.RSI(close, timeperiod=2)
    today_rsi = rsi[-1]
    cumulate_rsi = rsi[-1] + rsi[-2] + rsi[-3]
    return "\t".join([date, code, "当天RSI2: %.1lf" % today_rsi, "3d累积RSI2: %.1lf" % cumulate_rsi])


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

    response: CreateMessageResponse = client.im.v1.message.create(request)
    if not response.success():
        lark.logger.error(
            f"client.im.v1.message.create failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}, resp: \n{json.dumps(json.loads(response.raw.content), indent=4, ensure_ascii=False)}")

if __name__ == '__main__':
    codes = ["sh000300", "sz399905"]
    out = [run(code) for code in codes]
    msg = "\n".join(out)
    send_msg(msg)
