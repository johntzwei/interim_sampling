import os
import pickle
import itertools
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import pandas as pd

PUBLIC_RELEASE_PATH = "C:/Users/t-johnnywei/Documents/GitHub/ToShipOrNotToShip\public_release"

def load_data(use_cache=True):
    cache_filename = "./data.pickle"
    data = defaultdict(dict)
    if use_cache and os.path.isfile(cache_filename):
        with open(cache_filename, 'rb') as handle:
            data = pickle.load(handle)
    else:
        _, campaigns_list, _ = next(os.walk(PUBLIC_RELEASE_PATH))
        counter = 1
        for campaign in campaigns_list:
            if campaign not in data:
                data[campaign] = defaultdict(dict)
            for _, _, systems_list in os.walk(f"{PUBLIC_RELEASE_PATH}/{campaign}"):
                for system in systems_list:
                    if system not in data[campaign]:
                        data[campaign][system] = defaultdict(dict)
                    print(f"Loading {counter}/{len(campaigns_list)} campaign")
                    xls = pd.ExcelFile(f"{PUBLIC_RELEASE_PATH}/{campaign}/{system}")
                    for datatype in xls.sheet_names:
                        if datatype in ["hum_annotations",
                                        "full_test"]:
                            data[campaign][system][datatype] = pd.read_excel(
                                xls, datatype)
                        else:
                            df = pd.read_excel(xls, datatype)
                            # transform to dictionary
                            df_dict = df.set_index("Unnamed: 0").transpose()
                            df_dict = df_dict.iloc[0].to_dict()
                            data[campaign][system][datatype] = df_dict
                counter += 1

        # save the cache data
        if use_cache:
            with open(cache_filename, 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Annotated data loaded")
    return data


def pairs(data):
    for (k, v) in data.items():
        for i, j in itertools.combinations(v, 2):
            yield (v[i], v[j])
            
def filter_pairs(lp_source=None, lp_target=None, size=None):
    lp_pairs = []
    data = load_data()
    
    if lp_source == 'X' and lp_target == 'X':
        # randomly sample from all the data
        arr = np.concatenate([np.ones(size), np.zeros(len(data) - size)])
        np.random.seed(11997)
        np.random.shuffle(arr)

        for (i, j), mask in zip(pairs(data), arr):
            if mask > 0:
                lp_pairs.append((i, j))
    else:
        # subset only the specific language pair
        for i, j in pairs(data):    
            assert(i['hum_annotations']['Target'].unique() == j['hum_annotations']['Target'].unique())
            target = i['hum_annotations']['Target'].unique()
            assert(len(target) == 1)

            assert(i['hum_annotations']['Source'].unique() == j['hum_annotations']['Source'].unique())
            source = i['hum_annotations']['Source'].unique()
            assert(len(source) == 1)

            if source[0] == lp_source and target[0] == lp_target:
                lp_pairs.append((i, j))

    return lp_pairs
            
def test_all(pairs, power_func):
    results = []
    
    for df1, df2 in tqdm(pairs):
        row = []
        
        diff = df1['hum_annotations']['Score'].mean() - df2['hum_annotations']['Score'].mean()
        row.append(diff)

        power, avg_len = power_func(df1['hum_annotations']['Score'], df2['hum_annotations']['Score'])
        row.extend((power, avg_len))
            
        results.append(row)
            
    return np.array(results)