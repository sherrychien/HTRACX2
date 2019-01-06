
# coding: utf-8

import numpy as np
import HAT_model as HAT
import sequence as seq
from scipy import stats
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import copy
import random
np.set_printoptions(threshold=np.nan)

#generate sequence with different levels of scrambling
def seq_scramble(t, arr_size, scram_len, filler_len, intact, noise_level):
    seq = np.zeros((0,arr_size))
    label = []
    for n in range (t):
        for i in range(filler_len):
                filler = np.random.uniform(-1,1,size=arr_size)
                seq = np.concatenate((seq,[filler]),0)
                label.append('x')
        shuffle = [intact[x:x+scram_len] for x in range(0,len(intact)-scram_len,scram_len)]
        b = np.random.permutation(shuffle)
        b = np.append(b.flatten(),intact[-scram_len:])
        b = b.astype(int)
        for x in range(len(b)):
            bipolar = np.zeros((arr_size))-1
            noise = np.random.uniform(-noise_level,noise_level,size=arr_size)
            bipolar[b[x]] = 1
            bipolar = bipolar + noise
            seq = np.concatenate((seq,[bipolar]),axis=0)
            label.append(b[x])
            
    return label, seq


def Hid_corr (seq1,seq2,test1,test2,str1,str2):
    s = np.shape(test1['Hid'])
    str1_Hid, str1_Hid1, str1_Hid2 = (np.zeros((0,s[1])) for i in range(3))
    str2_Hid, str2_Hid1, str2_Hid2 = (np.zeros((0,s[1])) for i in range(3))
    corr_Hid, corr_Hid1, corr_Hid2 = np.zeros((0))
    seq1 = seq1[2:]
    seq2 = seq2[2:]
    
    s1 = np.shape(test1['Hid'])
    s2 = np.shape(test2['Hid'])
    
    test1_arr, test1_arr1, test1_arr2 = np.zeros((s1))
    test2_arr, test2_arr1, test2_arr2 = np.zeros((s2))

    for i in range(0,s1[1]):
        for j in range(0,s1[0]):
            test1_arr[j,i]=test1['Hid'][j,i]-np.mean(test1['Hid'],0)[i]
            test1_arr1[j,i]=test1['Hid1'][j,i]-np.mean(test1['Hid1'],0)[i]
            test1_arr2[j,i]=test1['Hid2'][j,i]-np.mean(test1['Hid2'],0)[i]
    
    s2 = np.shape(test2['Hid'])
    for i in range(0,s2[1]):
        for j in range(0,s2[0]):
            test2_arr[j,i]=test2['Hid'][j,i]-np.mean(test2['Hid'],0)[i]
            test2_arr1[j,i]=test2['Hid1'][j,i]-np.mean(test2['Hid1'],0)[i]
            test2_arr2[j,i]=test2['Hid2'][j,i]-np.mean(test2['Hid2'],0)[i]
           
    for i in range(0,len(seq1)):
        if np.all(seq1[i:i+len(str1)]==str1):
            str1_Hid = np.append(str1_Hid,test1_arr[i:i+len(str1)],0)
            str1_Hid1 = np.append(str1_Hid1,test1_arr1[i:i+len(str1)],0)
            str1_Hid2 = np.append(str1_Hid2,test1_arr2[i:i+len(str1)],0)
    
    for i in range(0,len(seq2)):    
        if np.all(seq2[i:i+len(str2)]==str2):
            str2_Hid = np.append(str2_Hid,test2_arr[i:i+len(str2)],0)
            str2_Hid1 = np.append(str2_Hid1,test2_arr1[i:i+len(str2)],0)
            str2_Hid2 = np.append(str2_Hid2,test2_arr2[i:i+len(str2)],0)


    size = np.shape(str1_Hid)
    for i in range(0,size[0]):
        corr_Hid =  np.append(corr_Hid, np.corrcoef(str1_Hid[i],str2_Hid[i])[1,0])
        corr_Hid1 = np.append(corr_Hid1, np.corrcoef(str1_Hid1[i],str2_Hid1[i])[1,0])
        corr_Hid2 = np.append(corr_Hid2, np.corrcoef(str1_Hid2[i],str2_Hid2[i])[1,0])
    

    corr_Hid_mean = np.mean(corr_Hid)
    corr_Hid1_mean = np.mean(corr_Hid1)
    corr_Hid2_mean = np.mean(corr_Hid2)
    return {'corr':corr_Hid, 'corr1':corr_Hid1, 'corr2':corr_Hid2, 'corr_mean':corr_Hid_mean, 'corr1_mean':corr_Hid1_mean,'corr2_mean':corr_Hid2_mean }
           

def calc_corr(corr_target,test_intact,test_scram,test1,test2):
    corr0, corr1, corr2 = (np.zeros((0))for i in range(3))
    for i in range(0,len(corr_target)):
        corr = Hid_corr(test_intact,test_scram,test1,test2,corr_target[i],corr_target[i])
        corr0 = np.append(corr0,corr['corr_mean'])
        corr1 = np.append(corr1,corr['corr1_mean'])
        corr2 = np.append(corr2,corr['corr2_mean'])
            
    return {'corr_lev0':np.mean(corr0), 'corr_lev1':np.mean(corr1), 'corr_lev2':np.mean(corr2)}
    
array_size = 30
filler_len = 5
intact = np.arange(array_size)
intact2 = np.random.permutation(intact)
noise_level=0.3
train_len=600
train_seq, bipolar_seq = seq_scramble(train_len,array_size,array_size,filler_len,intact,noise_level)
train_shuffle, bipolar_seq_shuffle = seq_scramble(train_len,array_size,array_size,filler_len,intact2,noise_level)

intact_len = array_size
pscram_len = 6
sscram_len = 3
wscram_len = 2
test_len = 10
#testing sequence1 (same as training sequence)
test_intact, bipolar_iseq = seq_scramble(test_len,array_size,intact_len,filler_len,intact,noise_level)
test_pscram, bipolar_pseq = seq_scramble(test_len,array_size,pscram_len,filler_len,intact,noise_level)
test_sscram, bipolar_sseq = seq_scramble(test_len,array_size,sscram_len,filler_len,intact,noise_level)
test_wscram, bipolar_wseq = seq_scramble(test_len,array_size,wscram_len,filler_len,intact,noise_level)

#The target elements for analyzing temporal context effect
corr_tar_p = [intact[x+5:x+pscram_len] for x in range(0,len(intact),pscram_len)]
corr_tar_s = [intact[x+2:x+sscram_len] for x in range(0,len(intact),sscram_len)]
corr_tar_w = [intact[x+1:x+wscram_len] for x in range(0,len(intact),wscram_len)]

Model = 'HAT_full'

# train the network 
n_sbj = 100
D, D1 = (np.zeros((0,3,len(train_seq)-2))for i in range(2))
ip_noise = 0.3

D_i,D1_i,D_i2, D1_i2 = (np.zeros((0,3,len(test_intact)-2))for i in range(4))
D_p,D1_p,D_p2, D1_p2 = (np.zeros((0,3,len(test_pscram)-2))for i in range(4))
D_s, D1_s,D_s2, D1_s2 = (np.zeros((0,3,len(test_sscram)-2))for i in range(4))
D_w, D1_w,D_w2, D1_w2 = (np.zeros((0,3,len(test_wscram)-2))for i in range(4))

H_i, H1_i, H_i2, H1_i2 = (np.zeros((0,3,len(test_intact)-2,array_size))for i in range(4))
H_p, H1_p, H_p2, H1_p2 = (np.zeros((0,3,len(test_pscram)-2,array_size))for i in range(4))
H_s, H1_s, H_s2, H1_s2 = (np.zeros((0,3,len(test_sscram)-2,array_size))for i in range(4))
H_w, H1_w, H_w2, H1_w2 = (np.zeros((0,3,len(test_wscram)-2,array_size))for i in range(4))


full_corr_p, full_corr_w, full_corr_s, full_corr_p2, full_corr_w2, full_corr_s2=([] for i in range(6))
Hid_corr_p, Hid_corr_w, Hid_corr_s, Hid_corr_p2, Hid_corr_w2, Hid_corr_s2=([] for i in range(6))

tau = [0,0.4,0.8]
tau_test = [0,0.4,0.8]
learning_rate=[0.05,0.05,0.05]
ws = 0.5 #initial weights scale
k= [1,1,1]
Modeltype='full'
obj='max'
def test_wts(seq,wts,Modeltype,obj,k, tau):
        test_wts = HAT.HAT_test(seq,wts['IH_wts'], wts['HO_wts'], wts['IH_wts1'], wts['HO_wts1'], wts['IH_wts2'], wts['HO_wts2'],tau, Modeltype,obj,k,ip_noise)  
    
        return test_wts

for i in range(0,n_sbj):
    print(i)    
    s = np.shape(bipolar_seq)

    s = np.shape(bipolar_seq)
    IH_wts = ws*(2*np.random.rand(2*s[1]+1, s[1])-1)
    HO_wts = ws*(2*np.random.rand(s[1]+1, 2*s[1])-1)
    IH_wts1 = ws*(2*np.random.rand(2*s[1]+1, s[1])-1)
    HO_wts1 = ws*(2*np.random.rand(s[1]+1, 2*s[1])-1)
    IH_wts2 = ws*(2*np.random.rand(2*s[1]+1, s[1])-1)
    HO_wts2 = ws*(2*np.random.rand(s[1]+1, 2*s[1])-1)
    
    wts=HAT.HAT_learn(bipolar_seq,IH_wts,HO_wts,IH_wts1,HO_wts1,IH_wts2,HO_wts2,learning_rate,tau,Modeltype,obj,k,ip_noise)
    wts1= HAT.HAT_learn(bipolar_seq_shuffle,IH_wts,HO_wts,IH_wts1,HO_wts1,IH_wts2,HO_wts2,learning_rate,tau,Modeltype,obj,k,ip_noise)

    test_i = test_wts(bipolar_iseq,wts,Modeltype,obj,k, tau_test)
    test_p = test_wts(bipolar_pseq,wts,Modeltype,obj,k, tau_test)
    test_s = test_wts(bipolar_sseq,wts,Modeltype,obj,k, tau_test)
    test_w = test_wts(bipolar_wseq,wts,Modeltype,obj,k, tau_test)
    test1_i = test_wts(bipolar_iseq,wts1,Modeltype,obj,k, tau_test)
    test1_p = test_wts(bipolar_pseq,wts1,Modeltype,obj,k, tau_test)
    test1_s = test_wts(bipolar_sseq,wts1,Modeltype,obj,k, tau_test)
    test1_w = test_wts(bipolar_wseq,wts1,Modeltype,obj,k, tau_test)

    
    # calculate the correlation
    
    print("calculate the correlation")
    corr_p_l0,corr_p_l1,corr_p_l2 = ([]for i in range(3))
    corr_s_l0,corr_s_l1,corr_s_l2 = ([]for i in range(3))
    corr_w_l0,corr_w_l1,corr_w_l2 = ([]for i in range(3))

    corr_p = calc_corr(corr_tar_p,test_intact,test_pscram,test_i,test_p)
    corr_s = calc_corr(corr_tar_s,test_intact,test_sscram,test_i,test_s)
    corr_w = calc_corr(corr_tar_w,test_intact,test_wscram,test_i,test_w)
    corr1_p = calc_corr(corr_tar_p,test_intact,test_pscram,test1_i,test1_p)
    corr1_s = calc_corr(corr_tar_s,test_intact,test_sscram,test1_i,test1_s)
    corr1_w = calc_corr(corr_tar_w,test_intact,test_wscram,test1_i,test1_w)    
    
    corr_p_l0 = np.append(corr_p_l0,[corr_p['corr_lev0'],corr1_p['corr_lev0']],0)
    corr_p_l1 = np.append(corr_p_l1,[corr_p['corr_lev1'],corr1_p['corr_lev1']],0)
    corr_p_l2 = np.append(corr_p_l2,[corr_p['corr_lev2'],corr1_p['corr_lev2']],0)
    full_corr_p.append([corr_p_l0,corr_p_l1,corr_p_l2])
    corr_s_l0 = np.append(corr_s_l0,[corr_s['corr_lev0'],corr1_s['corr_lev0']],0)
    corr_s_l1 = np.append(corr_s_l1,[corr_s['corr_lev1'],corr1_s['corr_lev1']],0)
    corr_s_l2 = np.append(corr_s_l2,[corr_s['corr_lev2'],corr1_s['corr_lev2']],0)
    full_corr_s.append([corr_s_l0,corr_s_l1,corr_s_l2])
    corr_w_l0 = np.append(corr_w_l0,[corr_w['corr_lev0'],corr1_w['corr_lev0']],0)
    corr_w_l1 = np.append(corr_w_l1,[corr_w['corr_lev1'],corr1_w['corr_lev1']],0)
    corr_w_l2 = np.append(corr_w_l2,[corr_w['corr_lev2'],corr1_w['corr_lev2']],0)
    full_corr_w.append([corr_w_l0,corr_w_l1,corr_w_l2])
    
#record the delta 
    D_i = np.append(D_i, [[test_i['D'],test_i['D1'],test_i['D2']]],0)
    D1_i = np.append(D1_i, [[test1_i['D'],test1_i['D1'],test1_i['D2']]],0)
    D_p = np.append(D_p, [[test_p['D'],test_p['D1'],test_p['D2']]],0)
    D1_p = np.append(D1_p, [[test1_p['D'],test1_p['D1'],test1_p['D2']]],0)
    D_s = np.append(D_s, [[test_s['D'],test_s['D1'],test_s['D2']]],0)
    D1_s = np.append(D1_s, [[test1_s['D'],test1_s['D1'],test1_s['D2']]],0)
    D_w = np.append(D_w, [[test_w['D'],test_w['D1'],test_w['D2']]],0)
    D1_w = np.append(D1_w, [[test1_w['D'],test1_w['D1'],test1_w['D2']]],0)
    
    H_i = np.append(H_i, [[test_i['Hid'],test_i['Hid1'],test_i['Hid2']]],0)
    H1_i = np.append(H1_i, [[test1_i['Hid'],test1_i['Hid1'],test1_i['Hid2']]],0)
    H_p = np.append(H_p, [[test_p['Hid'],test_p['Hid1'],test_p['Hid2']]],0)
    H1_p = np.append(H1_p, [[test1_p['Hid'],test1_p['Hid1'],test1_p['Hid2']]],0)
    H_s = np.append(H_s, [[test_s['Hid'],test_s['Hid1'],test_s['Hid2']]],0)
    H1_s = np.append(H1_s, [[test1_s['Hid'],test1_s['Hid1'],test1_s['Hid2']]],0)
    H_w = np.append(H_w, [[test_w['Hid'],test_w['Hid1'],test_w['Hid2']]],0)
    H1_w = np.append(H1_w, [[test1_w['Hid'],test1_w['Hid1'],test1_w['Hid2']]],0)
    

n_groups = 3
fig,ax=plt.subplots(figsize=(10,5))
index = np.arange(n_groups)
bar_width = 0.25
opacity = 0.5
error_config = {'ecolor': '0.3'}
mean_w=np.mean(full_corr_w,0)
mean_s=np.mean(full_corr_s,0)
mean_p=np.mean(full_corr_p,0)
sem_w=np.std(full_corr_w,0)
sem_s=np.std(full_corr_s,0)
sem_p=np.std(full_corr_p,0)

mean1 =[mean_p[0][0],mean_p[1][0],mean_p[2][0]]
sem1 = [sem_p[0][0],sem_p[1][0],sem_p[2][0]]
mean2 =[mean_s[0][0],mean_s[1][0],mean_s[2][0]]
sem2 = [sem_s[0][0],sem_s[1][0],sem_s[2][0]]
mean3 =[mean_w[0][0],mean_w[1][0],mean_w[2][0]]
sem3 = [sem_w[0][0],sem_w[1][0],sem_w[2][0]]

rects1 = plt.bar(index, mean1, bar_width,
                 alpha=opacity,error_kw=dict(lw=3, capthick=2),align="center",
                 color='blue', yerr=sem1,
                 label='paragraph_scramble')

rects2 = plt.bar(index + bar_width, mean2, bar_width,
                 alpha=opacity,error_kw=dict(lw=3, capthick=2),align="center",
                 color='g', yerr=sem2,
                 label='sentence_scramble')

rects3 = plt.bar(index + 2*bar_width, mean3, bar_width,
                 alpha=opacity,error_kw=dict(lw=3, capthick=2),align="center",
                 color='y', yerr=sem3,
                 label='words_scramble')

ax.set_xlabel('LEVEL', fontsize=15)
ax.set_ylabel('CORRELATION', fontsize=15)

ax.set_xticks(index + 2*bar_width / 2)
ax.set_xticklabels(('1', '2', '3'),fontsize=20)
ax.set_title('train_seq_%d_sbj_%d'%(train_len,n_sbj))
y=np.arange(0,1.2,0.2)
ax.set_yticklabels(y.round(decimals=1),fontsize=20)
ax.set_ylim([0,1])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.labelpad = 10
ax.yaxis.labelpad = 20
plt.tight_layout()
plt.show()


n_groups = 3
fig,ax=plt.subplots(figsize=(10,5))
index = np.arange(n_groups)
bar_width = 0.25
opacity = 0.5
error_config = {'ecolor': '0.3'}

mean1 =[mean_p[0][1],mean_p[1][1],mean_p[2][1]]
sem1 = [sem_p[0][1],sem_p[1][1],sem_p[2][1]]
mean2 =[mean_s[0][1],mean_s[1][1],mean_s[2][1]]
sem2 = [sem_s[0][1],sem_s[1][1],sem_s[2][1]]
mean3 =[mean_w[0][1],mean_w[1][1],mean_w[2][1]]
sem3 = [sem_w[0][1],sem_w[1][1],sem_w[2][1]]


rects1 = plt.bar(index, mean1, bar_width,
                 alpha=opacity,error_kw=dict(lw=3, capthick=2),align="center",
                 color='blue', yerr=sem1,
                 label='paragraph_scramble')

rects2 = plt.bar(index + bar_width, mean2, bar_width,
                 alpha=opacity,error_kw=dict(lw=3, capthick=2),align="center",
                 color='g', yerr=sem2,
                 label='sentence_scramble')

rects3 = plt.bar(index + 2*bar_width, mean3, bar_width,
                 alpha=opacity,error_kw=dict(lw=3, capthick=2),align="center",
                 color='y', yerr=sem3,
                 label='words_scramble')

ax.set_xlabel('LEVEL', fontsize=15)
ax.set_ylabel('CORRELATION', fontsize=15)

ax.set_xticks(index + 2*bar_width / 2)
ax.set_xticklabels(('1', '2', '3'),fontsize=20)
ax.set_title('train_random_%d_sbj_%d'%(train_len,n_sbj))
y=np.arange(0,1.2,0.2)
ax.set_yticklabels(y.round(decimals=1),fontsize=20)
ax.set_ylim([0,1])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.labelpad = 10
ax.yaxis.labelpad = 20
plt.tight_layout()
plt.show()

