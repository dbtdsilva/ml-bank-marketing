import csv

def get_csv_data(filename, fieldnames):
    data = []
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=fieldnames)
        for entry in reader:
            data += [ entry ]
    return data

def get_indexed_data(data, enums):
    data_indexed = []
    for idx, entry in enumerate(data):
        new_entry = {}
        for (key, val) in entry.items():
            new_entry[key] = enums[key].index(val)
        data_indexed += [ new_entry ]
    return data_indexed

def get_raw_array(keys, data):
    arr = []
    for idx, value in enumerate(data):
        sub = []
        for idx, key in enumerate(keys):
            sub += [value[key]]
        arr += [sub]
    return arr

def export_new_csv(filename, keys, data):
    f = open(filename, 'w')
    for idx, key in enumerate(keys):
        f.write(key + ('\n' if idx == len(keys) - 1 else ','))
    for idx, value in enumerate(data):
        for idx, key in enumerate(keys):
            f.write(str(value[key]) + ('\n' if idx == len(keys) - 1 else ','))
    f.close()

def main():
    keys = ('buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class')
    enums = {
        'buying' : ['vhigh', 'high', 'med', 'low'],
        'maint' : ['vhigh', 'high', 'med', 'low'],
        'class': ['unacc', 'acc', 'good', 'vgood'],
        'doors' : ['2', '3', '4', '5more'],
        'persons' : ['2', '4', 'more'],
        'lug_boot' : ['small', 'med', 'big'],
        'safety' : ['low', 'med', 'high']
    }

    data = get_csv_data('car.data', keys)
    data_indexed = get_indexed_data(data, enums)
    export_new_csv('car_indexed.data', keys, data_indexed)
    # Load using Octave
    # data = csvread('car_indexed.data')
    #print(get_raw_array(keys, data_indexed))

if __name__ == '__main__':
    main()