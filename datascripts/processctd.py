



if __name__ == '__main__':
    name_path = './name.txt'
    mesh_path = './mesh.txt'
    ctd_path = '../data/ncbi/CTD_diseases-2015-06-04.tsv'
    output_path = './output.txt'
    ctdnames, ctdmeshes = [], []
    ctd = {}
    with open(ctd_path, 'r') as ctdf:
        for i in xrange(28):
            ctdf.readline()
        
        for line in ctdf:
            content = line.split('\t')
            name = content[0].strip()
            mesh = content[1].strip()
            ctdnames.append(name)
            ctdmeshes.append(mesh)
            ctd[mesh] = name
            syns = content[7].strip().split('|')
            syns = [syn.lower() for syn in syns]
            if len(syns) == 1 and syns[0] == '':
                continue
            ctdnames.extend(syns)
            ctdmeshes.extend([mesh]*len(syns))
    right, total = 0, 0
    for name in ctdnames:
        if ctdnames.count(name) > 1:
            print name
    # with open(name_path, 'r') as namef, open(mesh_path, 'r') as meshf:
    #     for name, mesh in zip(namef, meshf):
    #         if name.strip() in ctdnames:
    #             right += 1
    #             if mesh.strip() != ctdmeshes[ctdnames.index(name.strip())]:
    #                 print name, mesh, ctdmeshes[ctdnames.index(name.strip())]
    #         # elif ctd.has_key(mesh.strip()):
    #         #     print name,':', ctd[mesh.strip()]
    #         total += 1
    #     print right, total
    #     # for mesh in ctd:
        #     name, pmeshes, syns = ctd[mesh]
        #     pnames = [ctd[pmesh][0] for pmesh in pmeshes if pmesh in ctd]
        #     for syn in syns:
        #         if syn not in pnames:
        #             continue
        #         output.write(str(syn) + '\t')
        #         output.write(str(pnames) + '\n')
            # output.write(str(name) + '\t')
            # output.write(str(syns) + '\t')
            # output.write(str(pnames) + '\n')
            # ctd[mesh].append(pnames)

