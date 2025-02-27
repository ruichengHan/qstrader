import akshare as ak
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)


def get_index_stock(index_code="000300", year='20218'):
    """
    获取一个指数的成分股
    Parameters
    ----------
    index_code
    year

    Returns
    -------

    """
    index_stock_cons_csindex_df = ak.index_stock_cons(symbol=index_code)
    filter_df = index_stock_cons_csindex_df[index_stock_cons_csindex_df["纳入日期"] < str(year)]
    out = []
    for code in filter_df["品种代码"].to_list():
        out.append(code)
    return out


def get_stock_his(code, start, end):
    non_adjust = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust="")
    hfq_adjust = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust="hfq")
    output = pd.merge(non_adjust, hfq_adjust, how='inner', on=['日期', '股票代码'])
    output = output[
        ["日期", "股票代码", "开盘_x", "收盘_x", "最高_x", "最低_x", '开盘_y', '收盘_y', '最高_y', '最低_y',
         '成交量_x', '换手率_x']]
    output.columns = ["date", "code", "open", 'close', 'high', 'low', 'adjust_open', 'adjust_close', 'adjust_high',
                      'adjust_low', 'volume', 'turnover']
    return output


def get_index_his(code):
    df = ak.stock_zh_index_daily(symbol=code)
    df["adjust_close"] = df["close"]
    df.to_csv(f"../qs_data/price/{code}.csv")


def download_stock_price():
    stock_300 = get_index_stock('000300')
    stock_500 = get_index_stock('399905')
    all_stock = sorted(set(stock_300 + stock_500))

    print("all stock size = ", len(all_stock))
    cnt = 0
    for code in all_stock:
        print(code)
        cnt += 1
        if cnt % 10 == 0:
            print('cnt = ', cnt)
        output = get_stock_his(code, "20160101", "20250101")
        output.to_csv(f"../qs_data/price/{code}.csv")

get_index_his("sh000300")