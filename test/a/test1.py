import datetime
import json
import os
import re
import shutil
import traceback
import warnings
# from memory_profiler import profile
from collections import Counter

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import web
from sklearn import metrics
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_samples, silhouette_score

warnings.filterwarnings("ignore")

urls = (
    "/ai/v1/load_time_elasticity", "load_time_elasticity"
)

app = web.application(urls, globals())


class Calculate(object):

    # @profile
    def Transfer(self, df_hour, df_day, future_max_t, future_min_t):
        # df_hour['date']=pd.to_datetime(df_hour['collect_time_hour'].astype('str').str[:8])
        # df_hour['h']=df_hour['collect_time_hour'].astype('str').str[8:]

        df_hour['date'] = pd.to_datetime(df_hour['collect_time_hour'].str.split(" ").str[0]).astype("str")
        df_hour['h'] = df_hour['collect_time_hour'].str.split(" ").str[1].str.split(":").str[0]

        print(df_hour.head(2))

        count = 0
        feature = 'load'
        df_hour.loc[df_hour[feature] >= 0, 'tmp_flag'] = 0
        df_hour["tmp_load"] = df_hour[feature].copy()
        df_hour['min_flag'] = 0
        df_hour['max_flag'] = 0
        for i in range(0, len(df_hour), 120 * 24):
            if len(df_hour) - i < 120 * 24 + 1:
                Q3 = df_hour["tmp_load"][:120 * 24].quantile(0.75)
                Q1 = df_hour["tmp_load"][:120 * 24].quantile(0.25)

                max1 = 4 * Q3 - 3 * Q1
                min1 = 4 * Q1 - 3 * Q3

                print("最大阈值：", max1, "最小阈值：", min1)

                if min1 < 0:
                    min1 = 0
                # 查找
                max_index = df_hour["tmp_load"][:120 * 24][df_hour["tmp_load"] > max1].index
                min_index = df_hour["tmp_load"][:120 * 24][df_hour["tmp_load"] < min1].index

                # 替换
                for j in min_index:
                    count += 1
                    df_hour.loc[j, 'min_flag'] = 1
                    df_hour.loc[j, "tmp_load"] = (df_hour.loc[j - 1, "tmp_load"] + df_hour.loc[j + 1, "tmp_load"]) / 2
                for j in max_index:
                    count += 1
                    df_hour.loc[j, "tmp_load"] = (df_hour.loc[j - 1, "tmp_load"] + df_hour.loc[j + 1, "tmp_load"]) / 2
                    df_hour.loc[j, 'max_flag'] = 1
            else:
                Q3 = df_hour["tmp_load"][-(i + 120 * 24):-i].quantile(0.75)
                Q1 = df_hour["tmp_load"][-(i + 120 * 24):-i].quantile(0.25)

                max1 = 4 * Q3 - 3 * Q1
                min1 = 4 * Q1 - 3 * Q3

                print("最大阈值：", max1, "最小阈值：", min1)

                if min1 < 0:
                    min1 = 0
                # 查找
                max_index = df_hour["tmp_load"][-(i + 120 * 24):-i][df_hour["tmp_load"] > max1].index
                min_index = df_hour["tmp_load"][-(i + 120 * 24):-i][df_hour["tmp_load"] < min1].index

                # 替换
                for j in min_index:
                    count += 1
                    df_hour.loc[j, 'min_flag'] = 1
                    df_hour.loc[j, "tmp_load"] = (df_hour.loc[j - 1, "tmp_load"] + df_hour.loc[j + 1, "tmp_load"]) / 2

                for j in max_index:
                    count += 1
                    df_hour.loc[j, 'max_flag'] = 1
                    df_hour.loc[j, "tmp_load"] = (df_hour.loc[j - 1, "tmp_load"] + df_hour.loc[j + 1, "tmp_load"]) / 2
            print('查找明显异常日的个数：', count)

        ##################################################################################
        dict_tmp = {}
        list_flag = []
        for code, group in df_hour.groupby('date'):
            if len(group[feature].T.tolist()) == 24:
                dict_tmp[code] = group[feature].T.tolist()

                if (1 in group['min_flag'].values) or (1 in group['max_flag'].values) or (
                        1 in group['tmp_flag'].values):
                    list_flag.append(1)
                else:
                    list_flag.append(0)

        df_tmp0 = pd.DataFrame(dict_tmp).T

        df_tmp0['total'] = df_tmp0.sum(axis=1)
        df_tmp0['flag'] = list_flag
        df_tmp0.reset_index(inplace=True)
        df_tmp0.rename(columns={'index': 'date'}, inplace=True)

        #################################################################################

        df_day.rename(columns={'collect_time_day': 'date', 'max_temperature': 'max_t', 'min_temperature': 'min_t',
                               }, inplace=True)
        df_tmp0['date'] = df_tmp0['date'].astype('str')
        df_day['date'] = df_day['date'].astype('str')

        # print(df_day.head(),df_tmp0.head(),"*********临时*************")
        df_tmp0 = pd.merge(df_day[['date', 'max_t', 'min_t']], df_tmp0, on=['date'])

        df_tmp0['season_min'] = df_tmp0["min_t"].rolling(4).mean().fillna(method='bfill').apply(lambda x: int(x))
        df_tmp0['season_max'] = df_tmp0["max_t"].rolling(4).mean().fillna(method='bfill').apply(lambda x: int(x))

        final_df = df_tmp0[df_tmp0['flag'] == 0]

        tmp1 = final_df[list(range(1, 12))]
        tmp2 = tmp1.T.diff(1).T.iloc[:, 1:]
        tmp3 = tmp2.copy()
        for i in tmp2.columns:
            tmp3[i] = tmp2[i] / (tmp1[i] + 1e-8)
        number = []
        for i in tmp3.index:
            j = np.argmax(tmp3.loc[i, :])
            number.append(j + 2)

        x = Counter(number)

        h_split = max(zip(x.values(), x.keys()))[1]

        # 可视化上午同比增长率最大所对应时段，频率最大的时段为每日负载集中加载时段
        # self.plot_x(x)

        final_df = self.is_workday(final_df, h_split)

        flag, result = self.v_var(final_df, future_max_t, future_min_t)

        return flag, result

    def plot_x(self, x):
        X = x.keys()
        Y = x.values()

        plt.rcParams['font.sans-serif'] = 'SimHei'
        plt.rcParams['axes.unicode_minus'] = False
        fig = plt.figure()
        plt.bar(list(X), list(Y), 0.4, color="green")
        plt.xlabel("小时")
        plt.ylabel("频数")
        plt.title("负载集中加载时刻统计")
        plt.show()

    def is_workday(self, df, h_split):
        df["cluster"] = 0
        df_part1 = df[df['season_min'] <= 9]
        df_part2 = df[df['season_min'] >= 24]
        df_part3 = df[(df['season_min'] > 9) & (df['season_min'] < 24)]

        for df_tmp0 in [df_part1, df_part2, df_part3]:

            col = ['total', h_split - 1, h_split, h_split + 1]

            zscore_scaler = preprocessing.StandardScaler()
            x = data_scaler_1 = zscore_scaler.fit_transform(df_tmp0[col])

            n_clusters = 2  #

            clusterer = KMeans(n_clusters=n_clusters, random_state=10).fit(x)
            cluster_labels = clusterer.labels_
            df_tmp0['cluster'] = cluster_labels

            tmp_dict = {}
            for code, group in df_tmp0.groupby('cluster'):
                tmp_dict[code] = group[[col[0]]].mean().values[0]
            tmp_dict = dict(sorted(tmp_dict.items(), key=lambda x: x[1], reverse=False))

            df.loc[df_tmp0[df_tmp0['cluster'] == list(tmp_dict.keys())[0]].index, "cluster"] = '0'
            df.loc[df_tmp0[df_tmp0['cluster'] == list(tmp_dict.keys())[1]].index, "cluster"] = '1'
        print(df["cluster"].value_counts())

        ############################################
        df["weekday"] = pd.to_datetime(df["date"]).dt.weekday

        dict_score = {}
        dict_df = {}
        for v in ["1", "0"]:
            x = np.array(df[df['cluster'] == v][['weekday']].value_counts().values).reshape(-1, 1)
            clusterer = KMeans(n_clusters=2, random_state=10).fit(x)
            dict_score[v] = silhouette_score(x, clusterer.labels_)

            tmp_df = pd.DataFrame(df[df['cluster'] == v][['weekday']].value_counts())
            tmp_df.reset_index(drop=False, inplace=True)
            tmp_df['cluster'] = clusterer.labels_
            dict_df[v] = tmp_df

        ###########################################
        tmp_dict = {}
        if dict_score["1"] > dict_score["0"]:
            val = "1"
        else:
            val = "0"

        tmp_df = dict_df[val].copy()
        if len(tmp_df[tmp_df['cluster'] == 0]) > len(tmp_df[tmp_df['cluster'] == 1]):
            w_day = tmp_df[tmp_df['cluster'] == 0].weekday.tolist()
            h_day = list(set(range(7)) - set(w_day))
        else:
            w_day = tmp_df[tmp_df['cluster'] == 1].weekday.tolist()
            h_day = list(set(range(7)) - set(w_day))
        print(h_day, w_day)

        df['cluster_adjust'] = df['cluster']
        df.loc[df["weekday"].isin(h_day), 'cluster_adjust'] = "0"
        df.loc[df["weekday"].isin(w_day), 'cluster_adjust'] = "1"
        print(df['cluster_adjust'].value_counts())

        ######################################
        df_tmp0 = \
            df[['date', 'max_t', 'min_t', 'total', 'season_min', 'season_max', "weekday", "cluster", "cluster_adjust"]][
                df["cluster_adjust"] == "1"]
        X_train = df_tmp0[['max_t', 'min_t', 'total', 'season_min', 'season_max', "weekday"]]
        clf = IsolationForest(max_samples=100, contamination=0.07)
        clf.fit(X_train)

        df_tmp0["isolate_cluster"] = clf.predict(X_train)

        df.loc[df_tmp0[df_tmp0["isolate_cluster"] == -1].index, "cluster_adjust"] = '0'
        df[df["cluster_adjust"] == '1']

        print(df['cluster_adjust'].value_counts())

        return df

    def v_var(self, final_df, future_max_t, future_min_t):
        df_tmp0 = final_df[final_df["cluster_adjust"] == '1']

        list_ix = []

        ix1 = df_tmp0[
            (np.abs(df_tmp0['min_t'] - future_min_t) <= 2) & (np.abs(df_tmp0['max_t'] - future_max_t) <= 1)].index
        ix2 = df_tmp0[
            (np.abs(df_tmp0['min_t'] - future_min_t) <= 1) & (np.abs(df_tmp0['max_t'] - future_max_t) <= 2)].index
        list_ix.extend(set(ix1))
        list_ix.extend(set(ix2))

        df_tmp1 = final_df.loc[set(list_ix), range(24)]
        tp = pd.DataFrame([], index=range(24))

        if len(set(list_ix)) > 20:
            flag = 0

            df_tmp1 = final_df.loc[set(list_ix), range(24)]
            tp = pd.DataFrame([], index=range(24))

            # 典型负荷曲线获取

            tp['type'] = 0
            for h in range(24):
                X = np.array(df_tmp1[h]).reshape(len(df_tmp1), -1)

                tp.loc[h, 'type'] = np.median(X)

            df_tmp02 = df_tmp1 - tp["type"]

            dict_ix = {}
            df_final = df_tmp1.copy()
            for ix_f in df_tmp1.index:
                raw_list = []
                dt_tmp0 = pd.DataFrame(df_tmp02.loc[ix_f])
                m = np.abs(dt_tmp0).mean().values[0]
                thr = m / 3
                dt_tmp1 = dt_tmp0[np.abs(dt_tmp0)[ix_f] > m]
                for ix in dt_tmp1.index:
                    r = ix + 1
                    while r < 24:

                        val0 = np.abs(dt_tmp0.loc[ix:r].sum()).values[0]
                        val1 = np.abs(dt_tmp0.loc[ix:r]).max().values[0]

                        if (val0 < thr) and (val1 > m):

                            val_tmp = r
                            while (val_tmp not in dt_tmp1.index) and (ix < val_tmp):
                                val_tmp -= 1
                            if ix < val_tmp:
                                raw_list.append([ix, val_tmp])

                            break
                        r += 1

                if len(raw_list) > 0:
                    i = 0
                    update_list = raw_list.copy()
                    while i < len(update_list):
                        v1 = raw_list[i].copy()
                        for l in range(i + 1, len(raw_list)):

                            if (v1[0] <= raw_list[l][0] <= v1[1]) or (v1[0] <= raw_list[l][1] <= v1[1]):
                                v1[0] = np.min([v1[0], raw_list[l][0]])
                                v1[1] = np.max([v1[1], raw_list[l][1]])
                                update_list.remove(raw_list[l])

                        #                     print(i,l,raw_list,update_list,v1,raw_list[i])

                        update_list.insert(i, v1)
                        update_list.remove(raw_list[i])

                        raw_list = update_list.copy()

                        i += 1

                    print("日期索引为{},具体日期：{}，负荷的转移的时间跨度{}".format(ix_f, df_tmp0.loc[ix_f, 'date'], update_list))
                    dict_ix[ix_f] = update_list
                    df_final.loc[ix_f, update_list[0][0]:update_list[0][1]] = tp.loc[
                                                                              update_list[0][0]:update_list[0][1],
                                                                              'type'].values

            # 纵向弹性系数计算

            tp['var'] = 0

            for h in range(24):
                span = list(range(2, 31))
                span.extend(list(np.linspace(0, 1, 11)))

                X = np.array(df_final[h]).reshape(len(df_final), -1)

                df_tmp02 = pd.DataFrame(X)
                q1 = df_tmp02[0].quantile(0.25)
                q3 = df_tmp02[0].quantile(0.75)
                thr_min = q1 - 2 * (q3 - q1)
                thr_max = q3 + 2 * (q3 - q1)
                tp.loc[h, 'var'] = df_tmp02[(df_tmp02[0] > thr_min) & (df_tmp02[0] < thr_max)][0].std()
        else:
            flag = 1

        return flag, tp


class load_time_elasticity:

    # @profile
    def POST(self):
        try:
            # 通过post接收传入的Json数据
            data = web.data()
            json_data = json.loads(data)

            # Json格式解析为python格式
            device_id = json_data["device_id"]

            collect_time_hour = json_data["collect_time_hour"]
            load = json_data["load"]

            collect_time_day = json_data["collect_time_day"]
            min_temperature = json_data["min_temperature"]
            max_temperature = json_data["max_temperature"]

            future_max_t = json_data["future_max_t"]
            future_min_t = json_data["future_min_t"]

            status = 200
            message_type = "normal df_hour transmission"
        except Exception as err:
            print('error_message', err.__class__.__name__, err)
            status = 400
            message_type = 'Parameter Error'

            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
            print(traceback.format_exc())
            """
            with open("../log/log.txt","a") as f:
                f.write("\n"+">>>" * 10+"train"+">>>" * 10+"\n")
                f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
                f.write(traceback.format_exc())"""

        if status == 200:
            try:
                df_day = pd.DataFrame({"collect_time_day": collect_time_day, "min_temperature": min_temperature,
                                       "max_temperature": max_temperature})

                df_hour = pd.DataFrame({"collect_time_hour": collect_time_hour, "load": load})

                flag, tp = Calculate().Transfer(df_hour, df_day, future_max_t, future_min_t)

                message_type = "Model training was successful!"
            except Exception as err:
                print('err_message', err.__class__.__name__, err)
                status = 500
                message_type = r'Internal Error'

                print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
                print(traceback.format_exc())
                with open("../log/log.txt", "a") as f:
                    f.write("\n" + ">>>" * 10 + "train" + ">>>" * 10 + "\n")
                    f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
                    f.write(traceback.format_exc())

        # python格式转换为json格式
        dict_result = {}

        dict_result["device_id"] = device_id
        dict_result["status"] = status
        dict_result["message"] = message_type

        dict_result["flag"] = flag

        if flag == 0:
            dict_result["result"] = tp["var"].apply(lambda x: int(x)).tolist()
        else:
            dict_result["result"] = []

        print(dict_result)

        result = json.dumps(dict_result)

        return result


if __name__ == "__main__":
    app.run()
#_author:"zqs"
#date:2023/11/6