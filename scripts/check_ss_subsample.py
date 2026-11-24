import sys,time; sys.path.insert(0,'src')
from ccf_clustering import data, models
from sklearn.metrics import silhouette_score
import numpy as np, json
from sklearn.model_selection import KFold
df=data.encode(data.load_merged('data')); X=data.feature_matrix(df)
kf=KFold(10,shuffle=True,random_state=42); tr,_=next(kf.split(X)); Xt=X[tr]
out={}
for name in ['KMC','GMMC']:
    m=models.MODELS[name](k=2).fit(Xt)
    approx=[silhouette_score(Xt,m.labels_,sample_size=20000,random_state=s) for s in range(42,52)]
    t=time.time(); ex=silhouette_score(Xt,m.labels_); 
    out[name]={'exact':ex,'approx_seed42':approx[0],'approx_mean':float(np.mean(approx)),'approx_sd':float(np.std(approx,ddof=1)),'sec':time.time()-t}
    print(name,out[name],flush=True)
json.dump(out,open('results/round4/ss_exact_check_fold1.json','w'),indent=1)
