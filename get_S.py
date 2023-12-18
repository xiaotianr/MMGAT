import argparse
import os
import re

########Get S to construct the heterogeous graph###################
parser = argparse.ArgumentParser(description="Process ATAC-seq data.")
parser.add_argument('--hash', default='gse',type=str,help='The The hash of the task')
parser.add_argument('--path', default='./save/',type=str,help='Path to the bed file') #contain hg38.fa GSE.bed GSE.bam  path/
parser.add_argument('--genome', default='hg38',type=str,help='The hash of the task')
parser.add_argument('--peak_flank',type=int, default=50, help='peaks length')
args = parser.parse_args()
######################################################
def makeseq(input_name, out_name):
    file = open(input_name, 'r')
    data1 = open(out_name, 'w')
    data = file.readlines()
    file.close()
    times = 0
    for t1 in data:
        if t1.startswith('>'):
            tt = 0
        else:
            ttt = 'A' + '\t' + 'peaks' + '\t' + t1.strip().upper() + '\t' + '1' + '\n'
            data1.writelines(ttt)
            times += 1
    data1.close()
def getchrom_index(args):
    data = []
    name = args.hash  
    path = args.path+name+'/' #./save/hash/
    genome = args.genome
    
    hint_tobias_name = path +  name + "_final_hint_tobias.bed"
    file = open(hint_tobias_name, 'r')
    data1 = file.readlines()
    file.close()
    ll = len(data1)
    test_val = ['train.bed', 'test.bed']
    test_val_fa = [path+name + 'train1.fa', path+name + 'test1.fa']
    pattern = re.compile(r'chr')
    file0 = open(test_val[0], 'w')
    file1 = open(test_val[1], 'w')
    for j in range(ll):
        t = data1[j]
        match_res = pattern.match(t)
        t1 = t.split('\t')
        if match_res:
            t11 = round((int(t1[1]) + int(t1[2])) / 2) - args.peak_flank - 1
            t12 = round((int(t1[1]) + int(t1[2])) / 2) + args.peak_flank 
            if t11 > 0 and t12 > 0:
                tt = str(t1[0]) + '\t' + str(int(t11)) + '\t' + str(int(t12)) + '\n'  
                if j <= 1000:
                    if j % 2 == 0:
                        file0.writelines(tt)
                    else:
                        file1.writelines(tt)
                else:
                    file0.writelines(tt)

    file0.close()
    file1.close()
    for k in range(2):
        cmd = 'bedtools getfasta -fi /home/ubuntu/server/genome/'+genome+'.fa -bed ' + test_val[k] + ' -fo ' + test_val_fa[k]
        os.system(cmd)
        os.remove(test_val[k])

def get_data(args):
    name = args.hash  
    path = args.path+name+'/' #./save/hash/
    innames = [path+name + 'train1.fa', path+name + 'test1.fa']
    outnames = [path+name + '_encode_AC.seq', path+name + '_encode_B.seq', path+name + '_encode.seq']
    for i in range(2):
        makeseq(innames[i], outnames[i])
        cmd = 'cat ' + outnames[i] + ' >> ' + outnames[2]
        os.system(cmd)
        path_encode = path + 'data'
        if not os.path.exists(path_encode):
            os.mkdir(path_encode)
        path_encode = path + 'data/encode_'  + str(args.peak_flank * 2 + 1)
        if not os.path.exists(path_encode):
            os.mkdir(path_encode)

        cmd = 'mv ' + outnames[i] + ' '+ path_encode

        os.system(cmd)

    path_encode = path + 'data/encode_'  + str(args.peak_flank * 2 + 1)
    if not os.path.exists(path_encode):
        os.mkdir(path_encode)
    cmd = 'mv ' + outnames[2] + ' ' + path_encode
    os.system(cmd)
    path_txt = path + 'data/encode_tfbs.txt'
    file = open(path_txt, 'a+')
    txxt = name + '_encode' + '\t' + name + '_encode' + '\n'
    file.writelines(txxt)
    file.close()
    os.remove(innames[0])
    os.remove(innames[1])

if __name__ == '__main__':
    getchrom_index(args)
    get_data(args)