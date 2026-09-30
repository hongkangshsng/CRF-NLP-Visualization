import os, csv, math
from itertools import product
from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt

OUT="outputs"; os.makedirs(OUT,exist_ok=True)
LABELS=["N","V","ADV","ADJ"]

train_data=[
(["自然","語言","處理","很","有趣"],["N","N","V","ADV","ADJ"]),
(["機器","學習","改變","世界"],["N","N","V","N"]),
(["學生","認真","學習"],["N","ADV","V"]),
(["模型","快速","預測","標籤"],["N","ADV","V","N"]),
(["資料","科學","很","重要"],["N","N","ADV","ADJ"]),
(["研究","需要","仔細","分析"],["N","V","ADV","V"]),
(["演算法","有效","處理","資料"],["N","ADV","V","N"]),
(["工程師","設計","智慧","系統"],["N","V","ADJ","N"])]
test_data=[
(["自然","語言","分析","很","重要"],["N","N","V","ADV","ADJ"]),
(["模型","快速","處理","資料"],["N","ADV","V","N"]),
(["學生","仔細","分析","資料"],["N","ADV","V","N"])]

def state_features(word,label,lexical=True,suffix=True):
    f={f"BIAS|TAG={label}":1.0}
    if lexical:f[f"WORD={word}|TAG={label}"]=1.0
    if suffix:
        f[f"LAST={word[-1]}|TAG={label}"]=1.0
        f[f"LEN={min(len(word),3)}|TAG={label}"]=1.0
    return f
def feature_vector(words,tags,cfg):
    f=Counter(); prev="<START>"
    for w,t in zip(words,tags):
        f.update(state_features(w,t,cfg["lexical"],cfg["suffix"]))
        if cfg["transition"]:f[f"TRANS={prev}->{t}"]+=1
        prev=t
    if cfg["transition"]:f[f"TRANS={prev}-><END>"]+=1
    return f
def dot(w,f):return sum(w.get(k,0)*v for k,v in f.items())
def distribution(words,w,cfg):
    seqs=list(product(LABELS,repeat=len(words)))
    scores=np.array([dot(w,feature_vector(words,s,cfg)) for s in seqs])
    m=scores.max(); e=np.exp(scores-m); p=e/e.sum()
    return seqs,scores,p,m+math.log(e.sum())
def predict(words,w,cfg):
    seqs,scores,p,logz=distribution(words,w,cfg); j=int(np.argmax(scores))
    return list(seqs[j]),float(scores[j]),float(p[j]),logz
def train(data,cfg,epochs=35,lr=.12,l2=.01):
    w=defaultdict(float); hist=[]
    for ep in range(1,epochs+1):
        g=defaultdict(float); ll=0
        for x,y in data:
            gold=feature_vector(x,y,cfg); gs=dot(w,gold)
            seqs,_,p,logz=distribution(x,w,cfg); ll+=gs-logz
            for k,v in gold.items():g[k]+=v
            for s,ps in zip(seqs,p):
                for k,v in feature_vector(x,s,cfg).items():g[k]-=ps*v
        for k in set(w)|set(g):
            g[k]-=l2*w[k]; w[k]+=lr*g[k]/len(data)
        hist.append(ll/len(data))
        if ep==1 or ep%5==0 or ep==epochs:
            print(f"[TRAIN] epoch={ep:02d} avg_log_likelihood={hist[-1]:.6f}")
    return dict(w),hist
def evaluate(data,w,cfg):
    c=n=sc=0
    for x,y in data:
        yp,*_=predict(x,w,cfg); sc+=yp==y
        for a,b in zip(y,yp):c+=a==b;n+=1
    return c/n,sc/len(data)

configs={
"A_state_only":{"lexical":True,"suffix":True,"transition":False},
"B_transition_only":{"lexical":False,"suffix":False,"transition":True},
"C_full":{"lexical":True,"suffix":True,"transition":True},
"D_no_suffix":{"lexical":True,"suffix":False,"transition":True}}
results=[];models={};histories={}
for name,cfg in configs.items():
    print("\n"+"="*70+"\nEXPERIMENT:",name)
    w,h=train(train_data,cfg); tr=evaluate(train_data,w,cfg); te=evaluate(test_data,w,cfg)
    results.append([name,len(w),*tr,*te]);models[name]=w;histories[name]=h
    print(f"[RESULT] train_token={tr[0]:.3f} test_token={te[0]:.3f} test_sentence={te[1]:.3f}")

best=max(results,key=lambda r:r[4])[0];w=models[best];cfg=configs[best]
x,gold=test_data[0];pred,score,prob,logz=predict(x,w,cfg)
print("\nBEST MODEL:",best,"\nX:",x,"\nGold:",gold,"\nPred:",pred)
print(f"SCORE={score:.6f} logZ={logz:.6f} P(Y*|X)={prob:.6f}")
print("\nFEATURE CONTRIBUTIONS")
fv=feature_vector(x,pred,cfg);running=0
for k,v in sorted(fv.items(),key=lambda z:abs(w.get(z[0],0)),reverse=True):
    con=v*w.get(k,0);running+=con
    print(f"{k:34s} value={v:.0f} weight={w.get(k,0):+.4f} contribution={con:+.4f} running={running:+.4f}")

with open(f"{OUT}/experiment_results.csv","w",newline="",encoding="utf-8-sig") as f:
    z=csv.writer(f);z.writerow(["experiment","num_weights","train_token_acc","train_sentence_acc","test_token_acc","test_sentence_acc"]);z.writerows(results)
with open(f"{OUT}/learned_weights.csv","w",newline="",encoding="utf-8-sig") as f:
    z=csv.writer(f);z.writerow(["feature","weight"]);z.writerows(sorted(w.items(),key=lambda z:abs(z[1]),reverse=True))
with open(f"{OUT}/prediction_details.csv","w",newline="",encoding="utf-8-sig") as f:
    z=csv.writer(f);z.writerow(["sentence","word","gold","pred","correct"])
    for i,(x,y) in enumerate(test_data,1):
        yp,*_=predict(x,w,cfg)
        for word,a,b in zip(x,y,yp):z.writerow([i,word,a,b,int(a==b)])
with open(f"{OUT}/analysis.txt","w",encoding="utf-8") as f:
    f.write("CRF feature ablation analysis\n"+"="*50+"\n")
    for r in results:f.write(f"{r[0]}: weights={r[1]}, test_token={r[4]:.3f}, test_sentence={r[5]:.3f}\n")
    f.write("\nState features model X-to-Y compatibility. Transition features model adjacent-label dependencies. Positive weights raise sequence score; negative weights lower it. Results are educational because the dataset is intentionally small.\n")

plt.figure(figsize=(10,5));plt.bar([r[0] for r in results],[r[4] for r in results]);plt.ylim(0,1.05);plt.ylabel("Test token accuracy");plt.title("CRF feature selection vs performance");plt.xticks(rotation=15);plt.tight_layout();plt.savefig(f"{OUT}/01_feature_selection_accuracy.png",dpi=160);plt.close()
top=sorted(w.items(),key=lambda z:abs(z[1]),reverse=True)[:15]
plt.figure(figsize=(11,6));plt.barh([z[0] for z in top][::-1],[z[1] for z in top][::-1]);plt.xlabel("Learned weight");plt.title("Top learned CRF weights");plt.tight_layout();plt.savefig(f"{OUT}/02_top_weights.png",dpi=160);plt.close()
plt.figure(figsize=(9,5))
for name,h in histories.items():plt.plot(range(1,len(h)+1),h,label=name)
plt.xlabel("Epoch");plt.ylabel("Average conditional log-likelihood");plt.title("CRF training curves");plt.legend();plt.tight_layout();plt.savefig(f"{OUT}/03_training_curves.png",dpi=160);plt.close()
seqs,scores,p,_=distribution(x,w,cfg);order=np.argsort(p)[::-1][:12]
plt.figure(figsize=(11,6));plt.barh(["-".join(seqs[i]) for i in order][::-1],[p[i] for i in order][::-1]);plt.xlabel("P(Y|X)");plt.title("Top candidate sequence probabilities");plt.tight_layout();plt.savefig(f"{OUT}/04_sequence_probabilities.png",dpi=160);plt.close()
print("\nDone: CSV, TXT and PNG results exported to outputs/.")
