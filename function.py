
def dt(df):
    list_ = []
    for i in range(len(df)-1):
        if df.iloc[i+1]["caseid"] == df.iloc[i]["caseid"]:
            list_ = list_ + [df.iloc[i+1]["ts"]-df.iloc[i]["ts"] ]
        else:
            list_ = list_ + [list_[-1]]
    list_ = list_ + [list_[-1]]

    return list_


