import os

def getctd(ctdpath):
    ctd = {}
    with open(ctdpath, 'r') as ctdf:
        for i in xrange(28):
            ctdf.readline()
        for line in ctdf:
            content = line.split('\t')
            name = content[0].strip()
            mesh = content[1].strip()
            syns = content[7].strip()
            ctd[mesh] = (name, syns)
    return ctd



def getnames(ncbi_dir):
    dir_names = ['train', 'dev', 'test', 'ctd']
    sub_dirs = [os.path.join(ncbi_dir, name) for name in dir_names]

    data = {}
    for sub_dir, dir_name in zip(sub_dirs, dir_names):
        namepath = os.path.join(sub_dir, 'name.txt')
        names = []
        with open(namepath, 'r') as namef:
            for line in namef:
                names.append(line.strip())
        data[dir_name] = names
    
    return data

def getwrongsamples(outputpath):
    

if __name__ == '__main__':
