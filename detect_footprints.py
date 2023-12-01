import argparse
from genericpath import exists
import pyBigWig
import os
import shutil
# from multiprocessing import  Process
import threading

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument('--hash', default='fbd7c1da229c4007bf2d7b8a1ba1cf03',type=str,help='The hash of the task')
parser.add_argument('--path', default='./upload/',type=str,help='Path to the bed file and bam file') #contain hg38.fa GSE.bed GSE.bam  path/
parser.add_argument('--genome', default='hg38',type=str,help='The hash of the task')
parser.add_argument('--savepath', default='./save/',type=str,help='Path to save the result file')
parser.add_argument('--peak_flank',type=int, default=50, help='peaks length')
args = parser.parse_args()

def get_hint(name,savepath,bedname,bamname,genome):
    # step1.Obtaining hint-atac footprint bed files
    os.system("rgt-hint footprinting --atac-seq --paired-end --organism="+genome+" --output-location="+savepath+" --output-prefix=hint_pre"+name+" "+bamname+" "+bedname)
    
    # step2.Obtaining standard hint-atac footprint bed files
    hint_pre_name=savepath+"/hint_pre"+name+".bed" #hsah/hint_prehash.bed
    hint_pre_file=open(hint_pre_name)
    hint_data=hint_pre_file.readlines()
    hint_pre_file.close()
    # hint_path=savepath+"hint" #path/hint
    # if not os.path.exists(hint_path):
    #     os.mkdir(hint_path)
    hint_name=savepath+"/hint_"+name+".bed" #path/hint/hint_GSE.bed
    hint_file=open(hint_name,'w')
    for t in hint_data:
        t=t.strip().split('\t')
        line = t[0]+'\t'+t[1]+'\t'+t[2]+'\n'
        hint_file.writelines(line)
    hint_file.close()

def get_tobias(name,savepath,bedname,bamname,genome):
    #step3 Obtaining TOBIAS footprint bw files
    tobias_atac_path = savepath + "/tobias_atac" # path/tobias_atac
    if not os.path.exists(tobias_atac_path):
        os.mkdir(tobias_atac_path)
    os.system("TOBIAS ATACorrect --bam "+bamname+" --genome ./genome/"+genome+".fa --peaks "+bedname+" --outdir "+tobias_atac_path)
    #get _corrected.bw  path/tobias_atac/GSE_corrected.bw
    tobias_footprint_bw_path=savepath+"/tobias_footprint_bw" # path/tobias_footprint_bw
    if not os.path.exists(tobias_footprint_bw_path):
        os.mkdir(tobias_footprint_bw_path)
    tobias_footprint_bw_name=tobias_footprint_bw_path+"/"+name+"_printscore.bw"
    os.system("TOBIAS FootprintScores --signal "+tobias_atac_path+"/"+name+"_corrected.bw --regions "+bedname+"  -o "+tobias_footprint_bw_name)
    #get _printscore.bw path/tobias_footprint_bw/GSE_printscore.bw

    #step4 Obtaining TOBIAS footprint bed files
    def regularize(score):
      final_score=float(int(score*100)/100)
      return final_score
    tobias_bw_name=name+"_printscore.bw"
    bw = pyBigWig.open(tobias_footprint_bw_name)  #path/tobias_footprint_bw/GSE_printscore.bw
    tobias_footprint_bed_path=savepath+"/tobias_footprint_bed"
    if not os.path.exists(tobias_footprint_bed_path):
        os.mkdir(tobias_footprint_bed_path)
    tobias_bed_name=tobias_footprint_bed_path+"/"+name+"_mergedFootprint.bed" #path/tobias_footprint_bed/GSE_mergedFootprint.bed
    of = open(tobias_bed_name, "w")
    startPos=0
    endPos=0
    for chrom, length in bw.chroms().items():
        intervals = bw.intervals(chrom)
        if intervals!=None:
            startChr=intervals[0]
            startChr_score=regularize(startChr[2])
            for interval in intervals:
              interval_score=regularize(interval[2])
              if interval_score!=startChr_score:
                startPos=startChr[0]
                endPos=interval[0]
                point=startChr_score
                of.write("{}\t{}\t{}\t{}\n".format(chrom, startPos, endPos,point))
                startChr=interval
                startChr_score  =regularize(startChr[2])
    bw.close()
    of.close()
    #get mergedFootprint.bed

    #step5 Obtaining TOBIAS top1500 footprint bed files
    # tobias_top1500_bed_path=savepath+"tobias_top1500_bed"
    # if not os.path.exists(tobias_top1500_bed_path):
    #     os.mkdir(tobias_top1500_bed_path)
    tobias_top1500_bed_name = savepath + '/top1500' + name + '.bed'  # path/tobias_top1500_bed/top1500GSE.bed
    nfile = open(tobias_top1500_bed_name, 'w')
    file = open(tobias_bed_name,"r")
    da = file.readlines()
    file.close()
    list1 = []
    t = 0
    # j=0
    # while j<1500:
    #     d1 = da[j + t].strip().split('\t')
    #     if int(d1[2]) - int(d1[1]) > 3:
    #         tup = (da[j],float(d1[-1]))
    #         list1.append(tup)
    #         j+=1
    #     else :
    #         t+=1
    for i in range(1500):
        d1 = da[i + t].strip().split('\t')
        if int(d1[2]) - int(d1[1]) > 3:
            tup = (da[i],float(d1[-1]))
            list1.append(tup)
        else:
            i -= 1
            t += 1
    list1 = sorted(list1, key=lambda tup:tup[1])[::-1]
    length = len(da)
    for i in range(1500 + t,length):
        d1 = da[i].strip().split('\t')
        if int(d1[2]) - int(d1[1]) > 3:
            tup = (da[i],float(d1[-1]))
            minn_score = list1[-1][1]
            if tup[1] > minn_score:
                list1.pop()
                list1.append(tup)
                list1 = sorted(list1, key=lambda tup:tup[1])[::-1]
        else:
            i -= 1
    for i in list1:
        nfile.write(i[0])
    nfile.close()

def get_hint_tobias(args):
    name = args.hash
    path = args.path+name  # path/
    if not os.path.exists(path):
        os.mkdir(path)
    savepath = args.savepath+name
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    bedname = path+'/'+name+'.bed' 
    bamname = path+'/'+name+'.bam'
    genome = args.genome

    t1 = threading.Thread(target=get_hint, args=(name,savepath,bedname,bamname,genome,))
    t2 = threading.Thread(target=get_tobias, args=(name,savepath,bedname,bamname,genome,))
    # t1 =Process(target=get_hint,args=(name,savepath,bedname,bamname,genome,)) #threading.Thread(target=get_hint, args=(name,savepath,bedname,bamname,genome,))
    # t2 =Process(target=get_tobias,args=(name,savepath,bedname,bamname,genome,))# threading.Thread(target=get_tobias, args=(name,savepath,bedname,bamname,genome,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    # step1.Obtaining hint-atac footprint bed files
    # hint_pre_path = savepath+"hint_pre"  # path/hint_pre
    # if not os.path.exists(hint_pre_path):
    #     os.mkdir(hint_pre_path)

    # os.system("rgt-hint footprinting --atac-seq --paired-end --organism="+genome+" --output-location="+savepath+" --output-prefix=hint_pre"+name+" "+bamname+" "+bedname)
    
    # test
    # bedfile = open(bedname)
    # beddata = bedfile.readlines()
    # bedfile.close()
    # savename = savepath +"/hint_pre"+name+".bed"
    # savefile = open(savename,'w')
    # for t in beddata:
    #     t=t.strip().split('\t')
    #     line = t[0]+'\t'+t[1]+'\t'+t[2]+'\n'
    #     savefile.writelines(line)
    # savefile.close()

    # get path/hint_pre/hint_preGSE.bed

    # step2.Obtaining standard hint-atac footprint bed files
    # hint_pre_name=savepath+"/hint_pre"+name+".bed" #hsah/hint_prehash.bed
    # hint_pre_file=open(hint_pre_name)
    # hint_data=hint_pre_file.readlines()
    # hint_pre_file.close()
    # # hint_path=savepath+"hint" #path/hint
    # # if not os.path.exists(hint_path):
    # #     os.mkdir(hint_path)
    hint_name=savepath+"/hint_"+name+".bed" #path/hint/hint_GSE.bed
    # hint_file=open(hint_name,'w')
    # for t in hint_data:
    #     t=t.strip().split('\t')
    #     line = t[0]+'\t'+t[1]+'\t'+t[2]+'\n'
    #     hint_file.writelines(line)
    # hint_file.close()
    # #get hint.bed  path/hint/hint_GSE.bed

    # #step3 Obtaining TOBIAS footprint bw files
    # tobias_atac_path = savepath + "/tobias_atac" # path/tobias_atac
    # if not os.path.exists(tobias_atac_path):
    #     os.mkdir(tobias_atac_path)
    # os.system("TOBIAS ATACorrect --bam "+bamname+" --genome ./genome/"+genome+".fa --peaks "+bedname+" --outdir "+tobias_atac_path+" --cores 8 ")
    # #get _corrected.bw  path/tobias_atac/GSE_corrected.bw
    # tobias_footprint_bw_path=savepath+"/tobias_footprint_bw" # path/tobias_footprint_bw
    # if not os.path.exists(tobias_footprint_bw_path):
    #     os.mkdir(tobias_footprint_bw_path)
    # tobias_footprint_bw_name=tobias_footprint_bw_path+"/"+name+"_printscore.bw"
    # os.system("TOBIAS FootprintScores --signal "+tobias_atac_path+"/"+name+"_corrected.bw --regions "+bedname+"  -o "+tobias_footprint_bw_name+" --cores 8 ")
    # #get _printscore.bw path/tobias_footprint_bw/GSE_printscore.bw

    # #step4 Obtaining TOBIAS footprint bed files
    # def regularize(score):
    #   final_score=float(int(score*100)/100)
    #   return final_score
    # tobias_bw_name=name+"_printscore.bw"
    # bw = pyBigWig.open(tobias_footprint_bw_name)  #path/tobias_footprint_bw/GSE_printscore.bw
    # tobias_footprint_bed_path=savepath+"/tobias_footprint_bed"
    # if not os.path.exists(tobias_footprint_bed_path):
    #     os.mkdir(tobias_footprint_bed_path)
    # tobias_bed_name=tobias_footprint_bed_path+"/"+name+"_mergedFootprint.bed" #path/tobias_footprint_bed/GSE_mergedFootprint.bed
    # of = open(tobias_bed_name, "w")
    # startPos=0
    # endPos=0
    # for chrom, length in bw.chroms().items():
    #     intervals = bw.intervals(chrom)
    #     if intervals!=None:
    #         startChr=intervals[0]
    #         startChr_score=regularize(startChr[2])
    #         for interval in intervals:
    #           interval_score=regularize(interval[2])
    #           if interval_score!=startChr_score:
    #             startPos=startChr[0]
    #             endPos=interval[0]
    #             point=startChr_score
    #             of.write("{}\t{}\t{}\t{}\n".format(chrom, startPos, endPos,point))
    #             startChr=interval
    #             startChr_score  =regularize(startChr[2])
    # bw.close()
    # of.close()
    # #get mergedFootprint.bed

    # #step5 Obtaining TOBIAS top1500 footprint bed files
    # # tobias_top1500_bed_path=savepath+"tobias_top1500_bed"
    # # if not os.path.exists(tobias_top1500_bed_path):
    # #     os.mkdir(tobias_top1500_bed_path)
    tobias_top1500_bed_name = savepath + '/top1500' + name + '.bed'  # path/tobias_top1500_bed/top1500GSE.bed
    # nfile = open(tobias_top1500_bed_name, 'w')
    # file = open(tobias_bed_name,"r")
    # da = file.readlines()
    # file.close()
    # list1 = []
    # t = 0
    # for i in range(1500):
    #     d1 = da[i + t].strip().split('\t')
    #     if int(d1[2]) - int(d1[1]) > 3:
    #         tup = (da[i],float(d1[-1]))
    #         list1.append(tup)
    #     else:
    #         i -= 1
    #         t += 1
    # list1 = sorted(list1, key=lambda tup:tup[1])[::-1]
    # length = len(da)
    # for i in range(1500 + t,length):
    #     d1 = da[i].strip().split('\t')
    #     if int(d1[2]) - int(d1[1]) > 3:
    #         tup = (da[i],float(d1[-1]))
    #         minn_score = list1[-1][1]
    #         if tup[1] > minn_score:
    #             list1.pop()
    #             list1.append(tup)
    #             list1 = sorted(list1, key=lambda tup:tup[1])[::-1]
    #     else:
    #         i -= 1
    # for i in list1:
    #     nfile.write(i[0])
    # nfile.close()

    #  get top1500 .bed  path/tobias_top1500_bed/top1500GSE.bed
    #step6 merge hint_tobias footprint bed files
    pre_hint_tobias_path=savepath+"/pre_hint_tobias"
    if not os.path.exists(pre_hint_tobias_path):
        os.mkdir(pre_hint_tobias_path)
    pre_hint_tobias=pre_hint_tobias_path+"/"+name+"_pre_hint_tobias.bed" # path/pre_hint_tobias/GSE_pre_hint_tobias.bed
    os.system("bedtools intersect -a "+hint_name+" -b "+tobias_top1500_bed_name+"> "+pre_hint_tobias)
    sort_hint_tobias_path=savepath+"/sort_hint_tobias"
    if not os.path.exists(sort_hint_tobias_path):
        os.mkdir(sort_hint_tobias_path)
    sort_hint_tobias=sort_hint_tobias_path+"/"+name+"_sort_hint_tobias.bed" # path/sort_hint_tobias/GSE_sort_hint_tobias.bed
    os.system("bedtools sort -i "+pre_hint_tobias+" >"+sort_hint_tobias)
    # final_hint_tobias_path=savepath+"final_hint_tobias"  #path/final_hint_tobias
    # if not os.path.exists(final_hint_tobias_path):
    #     os.mkdir(final_hint_tobias_path)
    hint_tobias_name=savepath+"/"+name+"_final_hint_tobias.bed"  # path/final_hint_tobias/GSE_final_hint_tobias.bed
    os.system("bedtools merge -i "+sort_hint_tobias+" > "+hint_tobias_name)

if __name__ == '__main__':
    get_hint_tobias(args)
    path = args.savepath+args.hash
    if os.path.exists(path+"/tobias_atac"):
        shutil.rmtree(path+"/tobias_atac")
    if os.path.exists(path+"/tobias_footprint_bw"):
        shutil.rmtree(path+"/tobias_footprint_bw")
    if os.path.exists(path+"/tobias_footprint_bed"):
        shutil.rmtree(path+"/tobias_footprint_bed")
    if os.path.exists(path+"/pre_hint_tobias"):
        shutil.rmtree(path+"/pre_hint_tobias")
    if os.path.exists(path+"/sort_hint_tobias"):
        shutil.rmtree(path+"/sort_hint_tobias")