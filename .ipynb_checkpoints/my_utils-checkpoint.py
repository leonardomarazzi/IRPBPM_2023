def dt(df):
    list_ = []
    for i in range(len(df)-1):
        if df.iloc[i+1]["caseid"] == df.iloc[i]["caseid"]:
            list_ = list_ + [df.iloc[i+1]["ts"]-df.iloc[i]["ts"] ]
        else:
            list_ = list_ + [list_[-1]]
    list_ = list_ + [list_[-1]]
    return list_


# def first_L_events(df,L):
#     df_t = df.groupby(["caseid"]).agg(list).reset_index()[["caseid","event"]]

#     list_len = df_t["event"].map(lambda x: len(x))
#     df_t["len"] = list_len
#     df_L = df_t.loc[df_t['len'] >= L]

#     list_case = df_L["event"].map(lambda x: x[:L])

#     out = pd.DataFrame(columns = ["caseid","list_case"])    

    
#     output = df_L["caseid"]
#     output["list_case"] = [i for i in list_case]

    
#     return output


def first_L_events(df,L):
    df_t = df.groupby(["caseid"]).agg(list).reset_index()[["caseid","event"]]

    list_len = df_t["event"].map(lambda x: len(x))
    df_t["len"] = list_len
    df_L = df_t.loc[df_t['len'] >= L] 
    df_L["event_prefix"] = df_L["event"].map(lambda x: x[0:L])
    return df_L[["caseid","event_prefix"]]

def aggregation_encoding(df):
    def avg(x):
        avg = x[0]
        for i in x[1:]:
            avg = avg + i
        return avg/len(x)
    
    
    df_grouped = df.groupby(["caseid"]).agg(list)
    for activity in df["activity"].unique():
        df_grouped[activity] = df_grouped["activity"].map(lambda x : x.count(activity)/len(x))

    for activity in df["resource"].unique():
        df_grouped[activity] = df_grouped["resource"].map(lambda x : x.count(activity)/len(x))
    df_grouped["avg_t"] = df_grouped["t"].map(lambda x : avg(x))


    return df_grouped[df_grouped.columns[6:]].copy()

def index_encoding(df):
    df_grouped= df.groupby(["caseid"]).agg(list)
    max_lenght = df_grouped["activity"].map(lambda x: len(x)).max()
    
    for i in range(0,max_lenght):
        for act in df["activity"].unique():
            df_grouped[f"{act}_{i+1}"] = df_grouped["activity"].map(lambda x : 1 if act in x else 0).copy()
        for resource in df["resource"].unique():
            df_grouped[f"{resource}_{i+1}"] = df_grouped["resource"].map(lambda x : 1 if resource in x else 0).copy() 
    
        t_i = []
    
        for index in range(0,len(df_grouped)):
            try:
                t_i.append(df_grouped.iloc[index]["ts"][i])
            except:
                t_i.append(0)
        
        df_grouped[f"t_{i+1}"] = t_i

    df_grouped["out"] = df_grouped["y"].map(lambda x : x[0])
    return df_grouped[df_grouped.columns[6:]].copy()
    