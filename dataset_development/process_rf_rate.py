import pandas as pd
import math

def process_tree_month_t_bill_rates(tb_rate):
    tb_rate = tb_rate.rename(columns={"DATE": "date", "DTB3": "rate"})
    tb_rate = tb_rate.set_index("date")
    tb_rate.index.name = "date"
    tb_rate.loc[tb_rate.rate == "."] = math.nan
    tb_rate["rate"] = pd.to_numeric(tb_rate["rate"])
    
    date_index = pd.date_range(tb_rate.index.min(), tb_rate.index.max())
    tb_rate = tb_rate.reindex(date_index)
    tb_rate["rate"] = tb_rate["rate"].fillna(method="ffill")
    tb_rate["rate"] = tb_rate["rate"] / 100
    tb_rate = tb_rate.round(4)

    return tb_rate


if __name__ == "__main__":
    tb_rate = pd.read_csv("./datasets/excel/three_month_treasury_bill_rate.csv", parse_dates=["DATE"], low_memory=False)

    tb_rate = process_tree_month_t_bill_rates(tb_rate)

    tb_rate.to_csv("./datasets/macro/t_bill_rate_3m.csv", index=True)