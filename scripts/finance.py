import json

import akshare as ak
import requests
import time

from ak_share_data import get_index_stock


def download_stock_finance(code):
    stock_financial_abstract_ths_df = ak.stock_financial_abstract_ths(symbol=code, indicator="按年度")
    stock_financial_abstract_ths_df.to_csv(f"../qs_data/finance/{code}_year.csv")

    quarter_df = ak.stock_financial_abstract_ths(symbol=code, indicator="按单季度")
    quarter_df.to_csv(f"../qs_data/finance/{code}_quarter.csv")


def download_finance_report_date(code, fr):
    if code[0] == '0':
        code = "sz" + code
    elif code[0] == '6':
        code = 'sh' + code
    else:
        code = "sz" + code

    ms = int(time.time() * 1000)
    url = f"https://proxy.finance.qq.com/ifzqgtimg/appstock/news/noticeList/search?page=1&symbol={code}&n=101&_var=finance_notice&noticeType=0103&from=web&_={ms}"
    text = requests.get(url).text
    text = text[text.find("{"):]
    j = json.loads(text)
    out = j
    total_page = j["data"]["total_page"]
    if total_page > 1:
        for i in range(2, total_page + 1):
            url = f"https://proxy.finance.qq.com/ifzqgtimg/appstock/news/noticeList/search?page={i}&symbol={code}&n=101&_var=finance_notice&noticeType=0103&from=web&_={ms}"
            text = requests.get(url).text
            text = text[text.find("{"):]
            j = json.loads(text)
            out["data"]["data"] += j["data"]["data"]

    text = json.dumps(out, ensure_ascii=False)
    fr.write(text)
    fr.write("\n")


if __name__ == '__main__':
    stock_300 = get_index_stock('000300')
    stock_500 = get_index_stock('399905')
    all_stock = sorted(set(stock_300 + stock_500))

    fr = open("../qs_data/finance/date.txt", 'w')
    for index, stock in enumerate(all_stock):
        print(index, stock)
        download_finance_report_date(stock, fr)
    fr.close()
