folder = "/home/brenta/scratch/jason/logs/voc/vanilla/exp_88"

from os import listdir
from os.path import isfile, join

def get_log_file(folder):
    onlyfiles = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]
    log_files = [x for x in onlyfiles if 'training_order' not in x]
    assert len(log_files) == 1
    return log_files[0]

def find_best_val_acc(log_file):
    print(log_file)
    best_val_acc = 0
    lines = open(log_file, 'r').readlines()
    for line in lines[1:]:
        parts = line[:-1].split(',')
        val_acc = float(parts[5])
        best_val_acc = max(best_val_acc, val_acc)
    
    return best_val_acc

if __name__ == "__main__":
    # log_file = get_log_file(folder)
    log_file = "/home/brenta/scratch/jason/logs/voc/vanilla/exp_88/log_382020_232755.csv"
    print(find_best_val_acc(log_file))