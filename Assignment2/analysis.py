"""Reproduce Assignment 2 fits, risk estimates and machine-readable results.

Run prepare.py first to inspect plots; predictions.md records the pre-fit views.
All input files are local. No network, sibling repository, or custom library is
needed. Monetary positions are dollar exposures to arithmetic returns.
"""
import json
import numpy as np
import pandas as pd
from scipy import stats, optimize
from prepare import ROOT, TAB, read, tails, main as prepare

SEED = 54502
NSIM = 100_000

def save(name, frame):
    frame.to_csv(TAB / (name + '.csv'))
    return frame

def risk(pnl, alpha=.05):
    """Loss VaR = -lower P&L quantile; ES = average worst alpha mass.

    Use the empirical inverse CDF (order statistic), not interpolation between
    observations. Fractional boundary mass keeps ES a coherent empirical risk
    measure for arbitrary n*alpha and ties. Negative VaR is retained if the
    cutoff is a profit: forcing it positive would change the risk measure.
    """
    x = np.sort(np.asarray(pnl))
    mass = len(x) * alpha
    k = int(np.floor(mass + 1e-10))
    fraction = max(0., mass-k)
    total = x[:k].sum() + (fraction*x[k] if fraction > 1e-10 else 0.)
    return float(-x[int(np.ceil(mass))-1]), float(-total/mass)

def aicc(ll, k, n):
    return -2*ll + 2*k + 2*k*(k+1)/(n-k-1)

def tfit(x):
    """Standardization and multiple starts stabilize the three-parameter MLE.

    Fitting tiny returns directly, especially with a demeaned loc near zero,
    can give an unhelpfully small initial Nelder-Mead simplex in location.
    Affine standardization preserves the MLE after transforming back.
    """
    center=float(np.mean(x)); spread=float(np.std(x))
    z=(np.asarray(x)-center)/spread
    candidates = [stats.t.fit(z, df, loc=0., scale=1.)
                  for df in [3., 8., 30.]]
    df,loc,scale=max(candidates, key=lambda p: np.sum(stats.t.logpdf(z, *p)))
    return df,center+spread*loc,spread*scale

def rj(a):
    """Rebonato-Jackel: clip eigenvalues, then rescale to unit diagonal."""
    val, vec = np.linalg.eigh(a)
    b = (vec*np.maximum(val, 0)) @ vec.T
    return b / np.sqrt(np.outer(np.diag(b), np.diag(b)))

def higham(a, tol=1e-12):
    """Nearest correlation matrix in unweighted Frobenius norm.

    Dykstra's correction is essential: plain alternating projections do not in
    general find the nearest point in the intersection. PSD is allowed to be
    singular. Tiny negative eigenvalues within tolerance are numerical zero.
    """
    y = a.copy(); correction = np.zeros_like(a)
    for it in range(10_000):
        old = y.copy()
        r = y-correction
        val, vec = np.linalg.eigh(r)
        x = (vec*np.maximum(val, 0)) @ vec.T
        correction = x-r
        y = x.copy(); np.fill_diagonal(y, 1.)
        if np.linalg.norm(y-old, 'fro') < tol and np.linalg.eigvalsh(y)[0] > -tol:
            return y, it+1
    raise RuntimeError('Higham did not converge')

def problem1():
    x = read(1).drop(columns='Day')
    cc = x.dropna().corr().to_numpy(); pw = x.corr().to_numpy()
    sd = x.std(ddof=1).to_numpy()
    w = np.array([-.4,-.3,-.2,-.1,1.])
    hi, iterations = higham(pw)
    rows = []
    for name, c in [('Complete',cc), ('Pairwise',pw), ('RJ',rj(pw)), ('Higham',hi)]:
        save('p1_corr_'+name.lower(), pd.DataFrame(c, index=x.columns, columns=x.columns))
        eig = np.linalg.eigvalsh(c)
        try:
            np.linalg.cholesky(c); chol = 'success'
        except np.linalg.LinAlgError:
            chol = 'fails'
        rows.append(dict(method=name, min_eigenvalue=eig[0], distance_pairwise=np.linalg.norm(c-pw),
                         tracking_variance=w @ (c*np.outer(sd,sd)) @ w, cholesky=chol))
        save('p1_eigenvalues_'+name.lower(), pd.DataFrame({'eigenvalue':eig}))
    result = save('p1_summary',pd.DataFrame(rows).set_index('method'))
    changes = [{'pair':f'{x.columns[i]}-{x.columns[j]}', 'pairwise':pw[i,j], 'higham':hi[i,j],
                'change':hi[i,j]-pw[i,j], 'joint_days':int(x.iloc[:,[i,j]].dropna().shape[0])}
               for i in range(5) for j in range(i+1,5)]
    save('p1_higham_changes',pd.DataFrame(changes).assign(abs_change=lambda f:f.change.abs()).sort_values('abs_change',ascending=False).set_index('pair'))
    assert result.loc['Pairwise','min_eigenvalue'] < 0
    assert result.loc['Higham','distance_pairwise'] <= result.loc['RJ','distance_pairwise']
    for c in [cc,rj(pw),hi]:
        assert np.allclose(np.diag(c),1) and np.linalg.eigvalsh(c)[0] > -1e-10
    return {'complete_rows':len(x.dropna()), 'higham_iterations':iterations,
            'repair_difference':float(np.linalg.norm(hi-rj(pw))),
            'full_history_sd':dict(zip(x.columns,sd))}

def problem2():
    raw = read(2).Price.pct_change().dropna().to_numpy()
    r = raw-raw.mean(); n=len(r)
    sigma = r.std(ddof=1)
    rows=[dict(method='Normal equal',sigma=sigma,VaR=-1e6*stats.norm.ppf(.05)*sigma)]
    ew=[]
    for lam in [.97,.94]:
        # Most recent observation has exponent zero. Use full-sample demeaning
        # (already performed), normalized finite-horizon weights, no Bessel fix.
        weights=lam**np.arange(n-1,-1,-1); weights/=weights.sum()
        s=np.sqrt(weights @ (r*r)); neff=1/(weights@weights)
        rows.append(dict(method=f'Normal EW {lam}',sigma=s,VaR=-1e6*stats.norm.ppf(.05)*s))
        ew.append(dict(lam=lam,neff=neff,half_life=np.log(.5)/np.log(lam),recent40_weight=weights[-40:].sum(),
                       sigma=s,sigma_se=s/np.sqrt(2*neff),VaR_se=-1e6*stats.norm.ppf(.05)*s/np.sqrt(2*neff)))
    df,loc,scale=tfit(r)
    rows += [dict(method='Student t',sigma=np.nan,VaR=-1e6*stats.t.ppf(.05,df,loc=loc,scale=scale)),
             dict(method='Historical',sigma=np.nan,VaR=risk(1e6*r)[0])]
    save('p2_var',pd.DataFrame(rows).set_index('method'))
    save('p2_ew',pd.DataFrame(ew).set_index('lam'))
    save('p2_regimes',pd.DataFrame([{'regime':name,'n':len(y),'mean':y.mean(),'sd':y.std(ddof=1),
                                  'excess_kurtosis':stats.kurtosis(y)}
                                  for name,y in [('Days 1-460',r[:-40]),('Days 461-500',r[-40:])]]).set_index('regime'))
    # Under zero-mean normal regimes, E[r^4] = 3 E[sigma^4].
    variances=np.array([np.var(r[:-40]),np.var(r[-40:])]); p=np.array([.92,.08])
    mixture_kurt=3*(p@(variances**2))/(p@variances)**2-3
    return dict(raw_mean=float(raw.mean()),t_df=df,t_loc=loc,t_scale=scale,
                normal_mixture_excess=float(mixture_kurt))

def problem3():
    x=read(3)[['A','B']]
    pnls={'A $1m':1e6*x.A,'B $1m':1e6*x.B,'A $2m':2e6*x.A,'A+B $2m':1e6*(x.A+x.B)}
    rows=[]
    for name,pnl in pnls.items():
        v5,e5=risk(pnl,.05); v1,e1=risk(pnl,.01)
        rows.append(dict(position=name,VaR_5=v5,ES_5=e5,VaR_1=v1,ES_1=e1,
                         Normal_VaR_5=-pnl.mean()-stats.norm.ppf(.05)*pnl.std(ddof=1)))
    table=save('p3_risk',pd.DataFrame(rows).set_index('position'))
    for alpha in ['5','1']:
        assert table.loc['A+B $2m','ES_'+alpha] <= table.loc[['A $1m','B $1m'],'ES_'+alpha].sum()+1e-8
    defaults=x<-.2
    return {'severe_A':int(defaults.A.sum()),'severe_B':int(defaults.B.sum()),
            'either':int(defaults.any(axis=1).sum()),'both':int(defaults.all(axis=1).sum()),
            'return_correlation':float(x.corr().iloc[0,1])}

def gaussian_logc(u,r):
    z=stats.norm.ppf(u)
    return stats.multivariate_normal.logpdf(z,cov=r)-stats.norm.logpdf(z).sum(axis=1)

def t_logc(u,r,df):
    z=stats.t.ppf(u,df)
    return stats.multivariate_t.logpdf(z,shape=r,df=df)-stats.t.logpdf(z,df).sum(axis=1)

def problem4():
    x=read(4); n=len(x); marg=[]; models={}; u=np.zeros(x.shape)
    for j,col in enumerate(x):
        normpars=stats.norm.fit(x[col]); tp=tfit(x[col])
        candidates=[('Normal',stats.norm,normpars,2),('Student t',stats.t,tp,3)]
        fitted=[]
        for name,dist,pars,k in candidates:
            ll=dist.logpdf(x[col],*pars).sum()
            row=dict(asset=col,model=name,df=pars[0] if k==3 else np.nan,
                     loc=pars[-2],scale=pars[-1],loglik=ll,AICc=aicc(ll,k,n))
            marg.append(row); fitted.append((row['AICc'],dist,pars,name))
        _,dist,pars,name=min(fitted,key=lambda z:z[0]); models[col]=(dist,pars,name)
        u[:,j]=dist.cdf(x[col],*pars)
    # Guard inverse CDF evaluation against exact numerical zero/one only.
    u=np.clip(u,1e-12,1-1e-12)
    save('p4_margins',pd.DataFrame(marg).set_index(['asset','model']))
    tau=np.eye(3)
    for i in range(3):
        for j in range(i):
            tau[i,j]=tau[j,i]=stats.kendalltau(x.iloc[:,i],x.iloc[:,j]).statistic
    r=np.sin(np.pi*tau/2)
    assert np.linalg.eigvalsh(r)[0]>0 # No repair needed for these data.
    save('p4_kendall_tau',pd.DataFrame(tau,index=x.columns,columns=x.columns))
    save('p4_R',pd.DataFrame(r,index=x.columns,columns=x.columns))
    # Fixed margins and fixed Kendall R, following Week 5: k=0 vs k=1.
    # A log-spaced grid checks that the bounded optimum is not a local artifact.
    objective=lambda df:-t_logc(u,r,df).sum()
    grid=np.geomspace(.5,200,50)
    scores=np.array([objective(df) for df in grid]); best=int(scores.argmin())
    assert 0<best<len(grid)-1
    fit=optimize.minimize_scalar(objective,bounds=(grid[best-1],grid[best+1]),method='bounded',options={'xatol':1e-7})
    assert fit.success
    df=float(fit.x)
    lg=gaussian_logc(u,r); lt=t_logc(u,r,df)
    save('p4_copula_grid',pd.DataFrame({'df':grid,'loglik':-scores}))
    save('p4_copulas',pd.DataFrame([dict(model=name,df=d,k=k,loglik=ll,AICc=aicc(ll,k,n),BIC=-2*ll+k*np.log(n))
                                   for name,d,k,ll in [('Gaussian',np.inf,0,lg.sum()),('Student t',df,1,lt.sum())]]).set_index('model'))
    rng=np.random.default_rng(SEED+4)
    # Common normal draws reduce some comparison noise. The t copula must share
    # ONE chi-square scale across all three assets within each simulated day.
    z=rng.standard_normal((NSIM,3)) @ np.linalg.cholesky(r).T
    ut=stats.t.cdf(z/np.sqrt(rng.chisquare(df,NSIM)[:,None]/df),df)
    ug=stats.norm.cdf(z)
    riskrows=[]; tailrows=[]; mcrows=[]
    for name,uv in [('Gaussian',ug),('Student t',ut)]:
        sim=np.column_stack([models[col][0].ppf(np.clip(uv[:,j],1e-12,1-1e-12),*models[col][1])
                             for j,col in enumerate(x)])
        pnl=1e6*sim.sum(axis=1)
        for alpha in [.05,.01]:
            v,e=risk(pnl,alpha); riskrows.append(dict(model=name,alpha=alpha,VaR=v,ES=e))
            # Ten independent batches give a rough Monte Carlo SE of the full
            # estimate, useful to avoid treating simulation dollars as exact.
            batch=np.array([risk(b,alpha) for b in np.array_split(pnl,10)])
            mcrows.append(dict(model=name,alpha=alpha,VaR_MC_SE=batch[:,0].std(ddof=1)/np.sqrt(10),ES_MC_SE=batch[:,1].std(ddof=1)/np.sqrt(10)))
        t=tails(pd.DataFrame(uv,columns=x.columns))*n/NSIM
        for pair,row in t.iterrows(): tailrows.append(dict(model=name,pair=pair,lower=row.lower,upper=row.upper))
    for alpha in [.05,.01]:
        v,e=risk(1e6*x.sum(axis=1),alpha); riskrows.append(dict(model='Historical',alpha=alpha,VaR=v,ES=e))
    save('p4_risk',pd.DataFrame(riskrows).set_index(['model','alpha']))
    save('p4_mc_error',pd.DataFrame(mcrows).set_index(['model','alpha']))
    save('p4_simulated_tails',pd.DataFrame(tailrows).set_index(['model','pair']))
    # Main central-day definition uses empirical ranks: each series is within
    # its own empirical 5th-95th percentiles. Also report fitted-U sensitivity.
    ranks=(x.rank()/(n+1)).to_numpy()
    central=((ranks>.05)&(ranks<.95)).all(axis=1)
    central_u=((u>.05)&(u<.95)).all(axis=1)
    diff=lt-lg
    save('p4_daily_likelihood',pd.DataFrame({'day':np.arange(1,n+1),'gauss_logc':lg,'t_logc':lt,'difference':diff,'central_rank':central,'central_fitted_u':central_u}).set_index('day'))
    rho=max(r[0,1],r[0,2],r[1,2]); lam=2*stats.t.cdf(-np.sqrt((df+1)*(1-rho)/(1+rho)),df+1)
    return dict(selected={col:model[2] for col,model in models.items()},copula_df=df,
                loglik_gain=float(diff.sum()),central_days=int(central.sum()),central_gain=float(diff[central].sum()),
                central_fraction=float(diff[central].sum()/diff.sum()),
                fitted_u_central_days=int(central_u.sum()),fitted_u_central_gain=float(diff[central_u].sum()),
                max_rho=float(rho),tail_dependence=float(lam))

def problem5():
    x=read(5)[['MKT','A','B']].pct_change().dropna()
    n=len(x); design=np.column_stack([np.ones(n),x.MKT])
    coef=np.linalg.lstsq(design,x[['A','B']],rcond=None)[0]
    e=x[['A','B']].to_numpy()-design@coef
    # Consistent n-1 divisors preserve the exact OLS covariance decomposition.
    # Regression MSE (n-2) is reported as a sensitivity, not mixed into it.
    residual_cov=np.cov(e,rowvar=False,ddof=1); market_var=x.MKT.var(ddof=1)
    residual_corr=np.corrcoef(e,rowvar=False)[0,1]
    save('p5_regressions',pd.DataFrame({'alpha':coef[0],'beta':coef[1],
                                      'residual_sd_n1':np.sqrt(np.diag(residual_cov)),
                                      'residual_sd_n2':np.sqrt((e*e).sum(axis=0)/(n-2))},index=['A','B']))
    save('p5_residual_covariance',pd.DataFrame(residual_cov,index=['A','B'],columns=['A','B']))
    beta=coef[1]; full=market_var*np.outer(beta,beta)+residual_cov
    independent=market_var*np.outer(beta,beta)+np.diag(np.diag(residual_cov))
    direct=x[['A','B']].cov().to_numpy()
    assert np.allclose(full,direct,atol=1e-15,rtol=1e-12)
    assert np.max(np.abs(design.T@e))<1e-10
    rng=np.random.default_rng(SEED+5)
    market=rng.normal(0,np.sqrt(market_var),NSIM); z=rng.standard_normal((NSIM,2))
    rows=[]
    for label,cov in [('Full residual covariance',residual_cov),('Independent residuals',np.diag(np.diag(residual_cov)))]:
        errors=z@np.linalg.cholesky(cov).T
        # Report fitted alphas but do not add them: the question assumes zero
        # expected returns. Both market and residual simulations are centered.
        sim=market[:,None]*beta+errors
        implied=market_var*np.outer(beta,beta)+cov
        for name,w in [('P1',np.array([1e6,1e6])),('P2',np.array([1e6,-1e6]))]:
            pnl=sim@w; v,_=risk(pnl)
            batch=np.array([risk(b)[0] for b in np.array_split(pnl,10)])
            rows.append(dict(assumption=label,portfolio=name,simulated_VaR=v,
                             analytical_VaR=-stats.norm.ppf(.05)*np.sqrt(w@implied@w),
                             delta_normal_VaR=-stats.norm.ppf(.05)*np.sqrt(w@direct@w),
                             MC_SE=batch.std(ddof=1)/np.sqrt(10)))
    save('p5_var',pd.DataFrame(rows).set_index(['assumption','portfolio']))
    save('p5_sample_covariance',pd.DataFrame(direct,index=['A','B'],columns=['A','B']))
    return dict(residual_correlation=float(residual_corr),market_sd=float(np.sqrt(market_var)),
                covariance_decomposition_max_error=float(np.max(np.abs(full-direct))))

def main():
    prepare()
    result={}
    for i,func in enumerate([problem1,problem2,problem3,problem4,problem5],1):
        result[f'problem{i}']=func()
        print(f'Problem {i} complete',flush=True)
    result['seed']=SEED; result['simulations']=NSIM
    (ROOT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
