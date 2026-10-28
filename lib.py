import numpy as np, pandas as pd, warnings
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
RDLogger.DisableLog('rdApp.*'); warnings.filterwarnings('ignore')

def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512)
    names=[d[0] for d in Descriptors._descList]
    X=[];keep=[];scaf=[]
    for i,s in enumerate(smiles):
        m=Chem.MolFromSmiles(s) if isinstance(s,str) else None
        if m is None or m.GetNumAtoms()==0: continue
        try:
            d=[f(m) for _,f in Descriptors._descList]
            fp=gen.GetCountFingerprintAsNumPy(m).astype(float)
            sc=MurckoScaffold.MurckoScaffoldSmiles(mol=m)
        except Exception: continue
        X.append(np.concatenate([d,np.log1p(fp)])); keep.append(i); scaf.append(sc)
    X=np.array(X,float); X[~np.isfinite(X)]=np.nan
    return X,np.array(keep),np.array(scaf)

def scaffold_split(scaf, seed, test_frac=0.2):
    rng=np.random.RandomState(seed)
    groups={}
    for i,s in enumerate(scaf): groups.setdefault(s,[]).append(i)
    keys=list(groups); rng.shuffle(keys)
    # big scaffold sets first (balanced scaffold split), randomized among ties via shuffle
    keys=sorted(keys,key=lambda k:-len(groups[k]))
    big=[k for k in keys if len(groups[k])>len(scaf)*test_frac/2]; small=[k for k in keys if k not in big]
    rng.shuffle(small); order=big+small
    n_tr=int((1-test_frac)*len(scaf)); tr=[]; te=[]
    # put big sets in train, then fill randomly; remaining -> test
    for k in order:
        (tr if len(tr)+len(groups[k])<=n_tr else te).extend(groups[k])
    return np.array(tr),np.array(te)

def prep(Xtr,Xte):
    med=np.nanmedian(Xtr,0); med=np.where(np.isfinite(med),med,0)
    for X in (Xtr,Xte):
        r,c=np.where(~np.isfinite(X)); X[r,c]=med[c]
    mu=Xtr.mean(0); sd=Xtr.std(0); keep=sd>1e-8
    f=lambda X: np.clip((X[:,keep]-mu[keep])/sd[keep],-5,5)
    return f(Xtr),f(Xte)

def cv_logit_l1(X,y,Cs=(0.003,0.01,0.03),seed=0):
    skf=StratifiedKFold(3,shuffle=True,random_state=seed); best=(-1,None)
    for C in Cs:
        sc=[]
        for a,b in skf.split(X,y):
            if y[a].min()==y[a].max(): continue
            m=LogisticRegression(penalty='l1',solver='liblinear',C=C,max_iter=200).fit(X[a],y[a])
            sc.append(roc_auc_score(y[b],m.decision_function(X[b])))
        s=np.mean(sc) if sc else -1
        if s>best[0]: best=(s,C)
    return best[1]

def lasso_logit(X,y,seed=0):
    C=cv_logit_l1(X,y,seed=seed)
    return LogisticRegression(penalty='l1',solver='liblinear',C=C,max_iter=300).fit(X,y)

def _strat_keep(r,y,h_frac):
    # class-stratified trimming: within each class keep the h_frac of records with the smallest residual,
    # so the rarer (toxic) class is not discarded wholesale
    keep=[]
    for c in np.unique(y):
        ic=np.where(y==c)[0]; k=max(int(h_frac*len(ic)),1)
        keep.append(ic[np.argsort(r[ic])[:k]])
    return np.concatenate(keep)

def slts_trim(X,y,h_frac=0.9,alpha=0.02,starts=3,csteps=4,seed=0):
    """Sparse LTS (concentration steps) on the 0/1 response: keeps the h observations
    with smallest absolute residual under an L1-penalised linear fit."""
    rng=np.random.RandomState(seed); n=len(y); h=int(h_frac*n); best=(np.inf,None)
    for _ in range(starts):
        idx=rng.choice(n,size=max(h//2,50),replace=False)
        if len(np.unique(y[idx]))<2: idx=np.arange(n)
        for _ in range(csteps):
            m=Lasso(alpha=alpha,max_iter=300,tol=1e-3,warm_start=False).fit(X[idx],y[idx])
            r=np.abs(y-m.predict(X)); idx=_strat_keep(r,y,h_frac)
        obj=np.sum(r[idx]**2)+alpha*np.abs(m.coef_).sum()
        if obj<best[0]: best=(obj,idx)
    return best[1]

def slts_lasso(X,y,h_frac=0.9,seed=0):
    idx=slts_trim(X,y,h_frac=h_frac,seed=seed)
    if y[idx].min()==y[idx].max(): idx=np.arange(len(y))
    m=lasso_logit(X[idx],y[idx],seed=seed); m.n_trimmed_=len(y)-len(idx); return m
