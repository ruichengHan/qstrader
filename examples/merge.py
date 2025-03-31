import akshare as ak
import pandas as pd
import talib
from qstrader.statistics.tearsheet import TearsheetStatistics
from qstrader.trading.backtest import BacktestTradingSession


class Broker(object):
    def __init__(self):
        self.account = 1000000
        self.cur_code = ""
        self.amount = 0
        self.buy_strategy = ""
        self.prev_account = 1000000
        self.profits = []
        self.fee = 0.0005
        self.equity_curve = []
        self.benmark_code = "sh000300"
        self.benchmark_equity_curve = []

    def buy(self, code, price, buy_strategy):
        if self.cur_code != "":
            return
        self.cur_code = code
        self.amount = self.account / price
        self.prev_account = self.account
        self.account = 0
        self.buy_strategy = buy_strategy
        print(date, "buy ", code, buy_strategy)

    def can_buy(self):
        return self.cur_code == ""

    def can_sell(self, code):
        return self.cur_code == code

    def sell(self, code, price):
        self.account = self.amount * price * (1 - self.fee)
        self.cur_code = ""
        self.amount = 0
        profit = self.account - self.prev_account
        print(date, "sell ", code, profit)
        self.profits.append(profit)

    def update_equity(self, index_data, date, index):
        date = pd.Timestamp(date)
        if self.cur_code == "":
            self.equity_curve.append((date, self.account))
        else:
            self.equity_curve.append((date, self.amount * index_data[self.cur_code]["close"][index]))

        self.benchmark_equity_curve.append((date, index_data[self.benmark_code]["close"][index]))

    def get_equity_curve(self):
        equity_df = pd.DataFrame(
            self.equity_curve, columns=['Date', 'Equity']
        ).set_index('Date')
        equity_df.index = equity_df.index.date
        return equity_df

    def get_benchmark_equity_curve(self):
        equity_df = pd.DataFrame(
            self.benchmark_equity_curve, columns=['Date', 'Equity']
        ).set_index('Date')
        equity_df.index = equity_df.index.date
        return equity_df


def print_profit_metrics(broker):
    print('total account = ', broker.account)
    print('total len = ', len(broker.profits))
    winning_len = len(list(filter(lambda x: x > 0, broker.profits)))
    print('winning len = ', winning_len)
    print("winning rate = ", winning_len * 1.0 / len(broker.profits))
    print("avg profit = ", sum(broker.profits) / len(broker.profits))
    print("max loss = ", min(broker.profits))


if __name__ == '__main__':
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", None)

    code_list = ["sz399905", 'sh000300']

    index_data = {}
    for code in code_list:
        df = ak.stock_zh_index_daily(symbol=code)
        df["date"] = df["date"].apply(lambda x: x.strftime("%Y-%m-%d"))
        df = df[df["date"] >= "2016-01-01"]
        index_data[code] = {"code": code, "data": df}

    for code in code_list:
        j = index_data[code]
        df = j["data"]
        close = df["close"].to_numpy()
        j["close"] = close
        j["open"] = df["open"].to_numpy()
        j["rsi2"] = talib.RSI(close, timeperiod=2)
        j["high"] = df["high"].to_numpy()
        j["low"] = df["low"].to_numpy()

    dates = index_data["sh000300"]["data"]["date"].to_numpy()

    broker = Broker()

    r_stop = 65
    default_hold_days = 1
    hold_days = 1
    for (index, date) in enumerate(dates):
        for code in code_list:
            # 昨天收盘RSI2满足条件，第二天开盘开干
            rsi2 = index_data[code]["rsi2"][index]
            rsi2_2d = rsi2 + index_data[code]["rsi2"][index - 1]
            rsi2_3d = rsi2 + index_data[code]["rsi2"][index - 1] + index_data[code]["rsi2"][index - 2]
            close = index_data[code]["close"][index]
            today_open = index_data[code]["open"][index]
            today_close = index_data[code]["close"][index]
            prev_low = index_data[code]["low"][index - 1]
            today_high = index_data[code]["high"][index]

            # 看能不能买
            if broker.can_buy():
                # if rsi2 < 5:
                #     broker.buy(code, close, "rsi")
                # if rsi2_2d < 35:
                #     broker.buy(code, close, "cu_rsi")
                if rsi2_3d < 65:
                    broker.buy(code, close, "cu_rsi")
                if prev_low * 0.995 < today_open < prev_low < today_high:
                    broker.buy(code, prev_low, "oops")
                    hold_days = default_hold_days

            if broker.can_sell(code):
                if broker.buy_strategy == "cu_rsi":
                    if rsi2 >= r_stop:
                        broker.sell(code, close)
                if broker.buy_strategy == "rsi":
                    if rsi2 >= r_stop:
                        broker.sell(code, close)
                if broker.buy_strategy == "oops":
                    hold_days -= 1
                    if hold_days < 0:
                        broker.sell(code, close)

        broker.update_equity(index_data, date, index)

    if broker.cur_code != "":
        price = index_data[broker.cur_code]["close"][-1]
        broker.sell(broker.cur_code, price)

    tearsheet = TearsheetStatistics(
        strategy_equity=broker.get_equity_curve(),
        benchmark_equity=broker.get_benchmark_equity_curve(),
        title='rsi2'
    )
    tearsheet.plot_results()
