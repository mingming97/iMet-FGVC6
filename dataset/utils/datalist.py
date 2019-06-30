import pandas as pd

def datalist_from_file(label_file):
    data = pd.read_csv(label_file)
    datalist = []
    for index, row in data.iterrows():
        img_path = row['id'] + '.png'
        ids = row['attribute_ids'].strip().split(' ')
        datalist.append((img_path, [int(id) for id in ids]))
    return datalist