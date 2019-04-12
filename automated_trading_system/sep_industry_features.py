def add_indmom(sep):
    # sep contains tickers in one industry
    dates = list(sep.index.unique())

    for date in dates:
        sep_for_date = sep.loc[sep.index == date]
        avg_weekly_industry_momentum = sep_for_date["mom12m_actual"].mean()
        sep.loc[sep.index == date, "indmom"] = avg_weekly_industry_momentum

    return sep
