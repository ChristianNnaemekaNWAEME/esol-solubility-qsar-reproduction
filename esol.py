"""Reproduce Delaney's ESOL solubility model with RDKit and test it under outlier contamination."""
import numpy as np, pandas as pd, warnings, json
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from lib import featurize, scaffold_split, prep
warnings.filterwarnings('ignore')
df=pd.read_csv('delaney-processed.csv'); y_all=df['measured log solubility in mols per litre'].values
X,keep,scaf=featurize(df.smiles.values); y=y_all[keep]; d=df.iloc[keep].reset_index(drop=True)
rmse=lambda a,b: float(np.sqrt(mean_squared_error(a,b)))
out={}
# ---- Part A: Delaney (2004) published equation, recomputed with RDKit descriptors
def four(smi):
    m=Chem.MolFromSmiles(smi); n=m.GetNumHeavyAtoms()
    return [Crippen.MolLogP(m), Descriptors.MolWt(m), rdMolDescriptors.CalcNumRotatableBonds(m),
            sum(a.GetIsAromatic() for a in m.GetAtoms())/n]
F=np.array([four(s) for s in d.smiles]); logp,mw,rb,ap=F.T
pub=0.16-0.63*logp-0.0062*mw+0.066*rb-0.74*ap
pub_file=d['ESOL predicted log solubility in mols per litre'].values
out['A_published_equation_RDKit_descriptors']={'n':len(y),'R2':r2_score(y,pub),'RMSE':rmse(y,pub)}
out['A_published_equation_as_in_dataset_file']={'R2':r2_score(y,pub_file),'RMSE':rmse(y,pub_file)}
out['A_corr_between_RDKit_and_file_predictions']=float(np.corrcoef(pub,pub_file)[0,1])
# refit the 4-term model (scaffold split) and compare coefficients with the published ones
tr,te=scaffold_split(scaf,0)
lr=LinearRegression().fit(F[tr],y[tr])
out['A_refit_4term']={'intercept':float(lr.intercept_),'coef_logP_MW_RB_AP':[float(c) for c in lr.coef_],
  'published':{'intercept':0.16,'coef':[-0.63,-0.0062,0.066,-0.74]},'test_R2':r2_score(y[te],lr.predict(F[te])),'test_RMSE':rmse(y[te],lr.predict(F[te]))}
# ---- Part B: many descriptors, robust estimator in its native (continuous-response) setting
def slts_lasso_reg(Xtr,ytr,h_frac=0.9,alpha=0.02,starts=5,csteps=6,seed=0):
    rng=np.random.RandomState(seed); n=len(ytr); h=int(h_frac*n); best=(np.inf,None)
    for _ in range(starts):
        idx=rng.choice(n,size=max(h//2,50),replace=False)
        for _ in range(csteps):
            m=Lasso(alpha=alpha,max_iter=2000).fit(Xtr[idx],ytr[idx]); r=np.abs(ytr-m.predict(Xtr)); idx=np.argsort(r)[:h]
        obj=np.sum(r[idx]**2)
        if obj<best[0]: best=(obj,idx)
    idx=best[1]; return LassoCV(cv=5,max_iter=5000,alphas=30).fit(Xtr[idx],ytr[idx])
rows=[]
for seed in range(5):
    tr,te=scaffold_split(scaf,seed); Xtr,Xte=prep(X[tr].copy(),X[te].copy())
    for cont in (0.0,0.05,0.10,0.20):
        rng=np.random.RandomState(500+seed); ytr=y[tr].copy(); k=int(cont*len(ytr)); 
        if k: idx=rng.choice(len(ytr),k,replace=False); ytr[idx]=ytr[idx]+rng.choice([-1,1],k)*rng.uniform(4,8,k)   # gross outliers (4-8 log units)
        models={'LASSO':lambda:LassoCV(cv=5,max_iter=5000,alphas=30).fit(Xtr,ytr),
                'SLTS+LASSO (keep 90%)':lambda:slts_lasso_reg(Xtr,ytr,0.9,seed=seed),
                'SLTS+LASSO (keep 75%)':lambda:slts_lasso_reg(Xtr,ytr,0.75,seed=seed),
                'Random Forest':lambda:RandomForestRegressor(300,n_jobs=2,random_state=seed).fit(Xtr,ytr)}
        for name,f in models.items():
            m=f(); p=m.predict(Xte); nz=int(np.sum(m.coef_!=0)) if hasattr(m,'coef_') else -1
            rows.append(dict(seed=seed,contamination=cont,model=name,rmse=rmse(y[te],p),r2=r2_score(y[te],p),n_nonzero=nz))
        # published 4-term model refit on contaminated labels
        l4=LinearRegression().fit(F[tr],ytr); p=l4.predict(F[te])
        rows.append(dict(seed=seed,contamination=cont,model='Delaney 4-term (refit)',rmse=rmse(y[te],p),r2=r2_score(y[te],p),n_nonzero=4))
res=pd.DataFrame(rows); res.to_csv('results_esol_robust.csv',index=False)
tab=res.groupby(['model','contamination']).rmse.mean().unstack().round(3); tab['loss_0_to_20']=(tab[0.2]-tab[0.0]).round(3)
json.dump(out,open('results_esol_reproduction.json','w'),indent=2,default=float)
print(json.dumps(out,indent=2,default=float)); print(tab); print(res[res.contamination==0].groupby('model').n_nonzero.mean().round(1))
