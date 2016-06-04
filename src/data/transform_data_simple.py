import csv
import sys

def get_csv_data(filename, fieldnames, has_header=True):
    data = []
    with open(filename) as csvfile:
        if has_header:
            csvfile.readline()
        reader = csv.DictReader(csvfile, delimiter=';', fieldnames=fieldnames)
        for entry in reader:
            data += [ entry ]
    return data

def get_indexed_data(data, enums):
    data_indexed = []
    for idx, entry in enumerate(data):
        new_entry = {}
        for (key, val) in entry.items():
            new_entry[key] = enums[key].index(val) if key in enums else val
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
    for idx, value in enumerate(data):
        for idx, key in enumerate(keys):
            f.write(str(value[key]) + ('\n' if idx == len(keys) - 1 else ','))
    f.close()

def main():
    keys = ("age","job","marital","education","default","balance","housing","loan",
        "contact","day","month","duration","campaign","pdays",
        "previous","poutcome","y")

    enums = {
        'job' : ['admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown'],
        'marital' : ['divorced','married','single','unknown'],
        'education' : ["unknown","secondary","primary","tertiary"],
        'default' : ['no','yes'],
        'housing' : ['no','yes'],
        'loan' : ['no','yes'],
        'contact' : ['unknown','cellular','telephone'],
        'month' : ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'],
        'poutcome' : ['failure','unknown','success','other'],
        'y' : ['yes','no']
    }

    data = get_csv_data(sys.argv[1], keys)
    data_indexed = get_indexed_data(data, enums)
    export_new_csv(sys.argv[2], keys, data_indexed)
    # Load using Octave
    # data = csvread('car_indexed.data')
    #print(get_raw_array(keys, data_indexed))

if __name__ == '__main__':
    main()
