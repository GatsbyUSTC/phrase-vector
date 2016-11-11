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
    data = []
    with open(outputpath, 'r') as ws:
        for line in ws:
            content = line.strip().split('\t')
            oid = int(content[0])
            rmesh = content[1]
            wid = int(content[2])
            data.append((oid, rmesh, wid))
    return data        

if __name__ == '__main__':
    ncbi_dir = '../data/ncbi'
    output_dir = '../outputs'

    ctd = getctd(os.path.join(ncbi_dir, 'CTD_diseases-2015-06-04.tsv'))
    names = getnames(ncbi_dir)
    ws = getwrongsamples(os.path.join(output_dir, 'wrongsamples.txt'))

    aopath = os.path.join(output_dir, 'ao.txt')
    with open(aopath, 'w') as aof:
        for w in ws:
            oid, rmesh, wid = w
            oname = names['train'][oid]
            rname = ctd[rmesh]
            wname = names['ctd'][wid]
            aof.write(oname + '\t' + str(rname) + '\t' + wname + '\n')
