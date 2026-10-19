"""Build the written PDF and editable Markdown from the computed CSV results.

Run analysis.py first. ReportLab is the only PDF-generation dependency; no
LaTeX, Quarto, browser, or network is needed. Explicit page breaks keep each
argument together. Tables are read from output files, not transcribed by hand.
"""
import json
import re
from pathlib import Path
from xml.sax.saxutils import escape
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

ROOT=Path(__file__).resolve().parent
R=json.loads((ROOT/'results.json').read_text())
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyA',fontName='Helvetica',fontSize=10,leading=12.7,spaceAfter=7))
styles.add(ParagraphStyle(name='NoteA',fontName='Helvetica',fontSize=8.5,leading=11,spaceAfter=7,textColor=colors.HexColor('#4c5965')))
styles.add(ParagraphStyle(name='TitleA',fontName='Helvetica-Bold',fontSize=23,leading=28,spaceAfter=12,textColor=colors.HexColor('#143b53')))
styles.add(ParagraphStyle(name='HeadA',fontName='Helvetica-Bold',fontSize=15,leading=19,spaceAfter=10,textColor=colors.HexColor('#143b53')))
styles.add(ParagraphStyle(name='SubA',fontName='Helvetica-Bold',fontSize=10.5,leading=13,spaceBefore=5,spaceAfter=5,textColor=colors.HexColor('#236f91')))
styles.add(ParagraphStyle(name='CellA',fontName='Helvetica',fontSize=8.2,leading=10.5))
styles.add(ParagraphStyle(name='HeaderA',fontName='Helvetica-Bold',fontSize=8.1,leading=10,textColor=colors.white))
story=[]; markdown=[]

def plain(s):
    return re.sub('<[^>]+>','',s).replace('&gt;','>').replace('&lt;','<').replace('&amp;','&')

def p(s,note=False):
    story.append(Paragraph(s,styles['NoteA' if note else 'BodyA']))
    markdown.append(plain(s)+'\n')

def h(s):
    story.append(Paragraph(escape(s),styles['SubA'])); markdown.append('## '+s+'\n')

def page(title,first=False):
    if not first: story.append(PageBreak())
    story.append(Paragraph(escape(title),styles['TitleA' if first else 'HeadA']))
    markdown.append('# '+title+'\n')

def load(name,index_col=0):
    return pd.read_csv(ROOT/'tables'/f'{name}.csv',index_col=index_col)

def fmt(x,spec='.6f'):
    if isinstance(x,(float,np.floating)):
        return '' if np.isnan(x) else format(x,spec)
    return str(x)

def table(df,formats=None,index=True,widths=None):
    """Render compact tables; every cell is a wrapping Paragraph."""
    frame=df.reset_index() if index else df.copy()
    formats=formats or {}
    rows=[[str(c) if not str(c).startswith('Unnamed') else '' for c in frame.columns]]
    rows += [[fmt(v,formats.get(c,'.6f')) for c,v in zip(frame.columns,row)] for row in frame.itertuples(index=False,name=None)]
    cells=[[Paragraph(escape(s),styles['HeaderA' if i==0 else 'CellA']) for s in row] for i,row in enumerate(rows)]
    t=Table(cells,colWidths=widths or [504/len(rows[0])]*len(rows[0]),repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#236078')),
                          ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f1f5f7'),colors.white]),
                          ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),
                          ('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),4),
                          ('BOTTOMPADDING',(0,0),(-1,-1),4),('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#bbccd4'))]))
    story.extend([t,Spacer(1,9)])
    markdown.append('| '+' | '.join(rows[0])+' |\n| '+' | '.join(['---']*len(rows[0]))+' |\n'+'\n'.join('| '+' | '.join(row)+' |' for row in rows[1:])+'\n')

def fig(name,caption,width=504):
    path=ROOT/'figures'/name
    iw,ih=ImageReader(str(path)).getSize()
    story.append(Image(str(path),width=width,height=width*ih/iw))
    p(escape(caption),note=True)
    markdown.append(f'![{caption}](figures/{name})\n')

def money(v): return f'${v:,.0f}'

page('Assignment 2',True)
p('<b>Covariance, VaR, and Copulas</b><br/>FinTech 545 - Quantitative Risk Management<br/>Yiting Sun | October 8, 2026')
h('Conventions and reproducibility')
p('Each problem follows Predict, Fit, and Reconcile. Predictions were recorded in predictions.md after inspecting raw-data summaries and plots, before model fitting. The script prepare.py isolates that inspection step. All five supplied CSV files are included unchanged in data/.')
p('Returns are arithmetic. Descriptive moments use denominator n and report excess kurtosis. Unless stated otherwise, covariance and sample standard deviation use n - 1. VaR is minus the lower P&amp;L quantile; ES is minus the average of the worst alpha fraction. Losses are positive; a negative VaR is a profit at the cutoff, not an absolute-valued loss. Historical quantiles use the empirical inverse CDF, without interpolation. Simulations use 100,000 scenarios and fixed random seeds.')
h('Problem 1 - Predict (a-c)')
p('The joint-observation table is computed before correlations. Its diagonal gives each series\' full observed history; <b>all five are observed together on only 81 days</b>.')
table(load('p1_joint_counts'),formats={c:'.0f' for c in ['A','B','C','D','IDX']})
p('<b>(a)</b> Complete-case correlation is a standardized Gram matrix on one common sample, so v\'Rv is a sum of squared standardized portfolio observations and cannot be negative. It is guaranteed PSD, but can be singular. Pairwise deletion combines entries estimated from different samples and different sample means; there need not be one common Gram matrix. It can therefore be indefinite.')
p('<b>(b)</b> If IDX were exactly the fixed basket, one linear dependence would make the covariance and correlation matrices singular. Small tracking error makes the smallest population eigenvalue close to zero. This leaves little margin for inconsistent pairwise estimates: even a small estimation error can push that eigenvalue below zero.')
p('<b>(c)</b> Correlations involving D should be least reliable because they use only 83-96 joint days. C-D has the fewest, 83. A and B have 242 joint days. These counts describe sampling precision, not a guarantee that the repair will target those entries.')

page('Problem 1 - Fit: correlation matrices')
h('(d) Complete-case correlation')
table(load('p1_corr_complete'),formats={c:'.5f' for c in ['A','B','C','D','IDX']})
h('(d) Pairwise correlation')
table(load('p1_corr_pairwise'),formats={c:'.5f' for c in ['A','B','C','D','IDX']})
eigs=pd.DataFrame({'Complete case':load('p1_eigenvalues_complete').iloc[:,0].to_numpy(),'Pairwise':load('p1_eigenvalues_pairwise').iloc[:,0].to_numpy()},index=np.arange(1,6))
eigs.index.name='Eigenvalue'
table(eigs)
p('Ordinary NumPy Cholesky <b>succeeds</b> for the complete-case matrix and <b>fails</b> for the pairwise matrix (not positive definite). The complete-case minimum eigenvalue is 0.001506, while the pairwise minimum is -0.011072. Here complete-case correlation is strictly positive definite; PSD alone would not have guaranteed an ordinary Cholesky factorization.')
h('(e) Tracking-portfolio variance before repair')
p('In the order A, B, C, D, IDX, let w = (-0.40, -0.30, -0.20, -0.10, 1). Use each series\' entire available history for its sample standard deviation, collected in D = diag(s). Then Sigma = D R_pair D and Var(w\'r) = w\' Sigma w.')
sd=pd.DataFrame([R['problem1']['full_history_sd']],index=['Full-history SD'])
table(sd,formats={c:'.8f' for c in sd.columns},widths=[99,81,81,81,81,81])
p('<b>Tracking variance = -1.799370 x 10<super>-6</super></b> in squared-return units. It is not a negative dollar loss or a low-risk forecast: it is an invalid variance produced by an inconsistent covariance estimate.')

page('Problem 1 - Repair and reconcile')
h('(f) Two repairs, with the same full-history standard deviations')
summary=load('p1_summary').loc[['Pairwise','RJ','Higham'],['min_eigenvalue','distance_pairwise','tracking_variance']]
summary.columns=['Minimum eigenvalue','Frobenius distance','Tracking variance']
table(summary,formats={'Minimum eigenvalue':'.6e','Frobenius distance':'.8f','Tracking variance':'.6e'})
p('Rebonato-Jackel clips negative eigenvalues at zero and rescales the result to unit diagonal. Higham alternates projection onto the PSD cone and the unit-diagonal set, with Dykstra\'s correction, to minimize the unweighted Frobenius distance. Higham converges in 34 iterations at tolerance 10<super>-12</super>. Its -4.26 x 10<super>-13</super> eigenvalue is numerical zero. A PSD repair can sit on the singular boundary; successful ordinary Cholesky is not its acceptance criterion.')
h('(g) Why the negative variance is impossible')
p('Positive semidefiniteness means v\' Sigma v &gt;= 0 for <i>every</i> real vector v. The tracking weights exhibit a concrete direction that violates this definition. Scaling an indefinite correlation matrix by a positive diagonal does not remove the problem. Both repairs restore a nonnegative tracking variance, about 6.8 x 10<super>-7</super>. This is algebraic consistency, not proof that either repaired estimate is the true covariance.')
h('(h) Which entries Higham changes most')
changes=load('p1_higham_changes').iloc[:5][['pairwise','higham','change','joint_days']]
changes.columns=['Pairwise','Higham','Change','Joint days']
table(changes,formats={'Pairwise':'.6f','Higham':'.6f','Change':'+.6f'})
p('The largest moves involve IDX, especially A-IDX, despite their relatively long histories. C-D, the least-supported pair, moves by only +0.001205. Thus the repair does not follow the reliability ranking in (c). It uses the geometry of the negative-eigenvalue direction and the unit-diagonal constraint. Entries whose adjustment most efficiently restores a feasible joint correlation structure move most. No observation counts or estimation-error weights enter this objective.')
h('(i) Estimator choice versus repair choice')
p('The complete-case/pairwise Frobenius gap is <b>0.423935</b>. The two repair sizes are only 0.016182 and 0.015520, and the repaired matrices differ by 0.004579. The estimator gap is about 27 times the Higham repair size. For these data the choice of estimator matters much more than which of these two repairs is used. Complete cases sacrifice substantial history and change the sample period, whereas pairwise deletion uses more data at the price of possible incoherence.')

page('Problem 2 - Predict from the return history')
fig('p2_returns.png','Figure 1. Five hundred demeaned arithmetic returns; the last roughly 40 days show larger fluctuations.')
h('(a) Pre-fit ranking')
p('My ascending prediction was <b>historical &lt; fitted t &lt; equal-weight normal &lt; EW normal (0.97) &lt; EW normal (0.94)</b>. I expected the recent volatile period to dominate the EW estimators, especially the faster 0.94 estimator. The first three were tentative: full-sample 5% historical and t cutoffs can reflect the old calm regime, while a normal fit spreads the effect of extreme returns across its variance. This original ranking is retained even though the historical estimate turns out differently.')
h('(b) Effective sample size, half-life and regime weights')
p('I identify the last 40 returns (days 461-500) as an approximate high-volatility regime by visual inspection; this is not an optimized or known structural break. With the newest return assigned lag zero, w_j = (1 - lambda) lambda<super>j</super> / (1 - lambda<super>500</super>). Then n_eff = 1 / sum(w_j<super>2</super>), half-life = log(0.5) / log(lambda), and recent-40 weight = (1 - lambda<super>40</super>) / (1 - lambda<super>500</super>).')
ew=load('p2_ew')[['neff','half_life','recent40_weight']].copy()
ew.columns=['Effective n','Half-life (days)','Recent 40 weight']
table(ew,formats={'Effective n':'.3f','Half-life (days)':'.3f','Recent 40 weight':'.4f'})
p('The same 40 observations receive only 8% under equal weighting, compared with 70.43% at 0.97 and 91.58% at 0.94. The infinite-history effective-n approximation (1 + lambda)/(1 - lambda) is practically identical here.')
h('(c) First four moments and the normal-regime prediction')
table(load('p2_moments'),formats={'mean':'.3e','variance':'.8f','skewness':'.6f','excess_kurtosis':'.6f'})
p('Excess kurtosis is 2.132, whereas a normal population has zero; skewness is only -0.156. A finite normal sample can produce nonzero kurtosis, but this magnitude and the visible time clustering make one stable normal regime unconvincing. A mixture of calm and volatile periods can create heavy unconditional tails even if each regime is approximately normal. The raw mean removed from returns is 0.000161876 per day.')

page('Problem 2 - Fit and reconcile')
h('(d) Five one-day VaR estimates on $1,000,000')
v2=load('p2_var'); table(v2[['VaR']].rename(columns={'VaR':'VaR ($)'}),formats={'VaR ($)':',.2f'},widths=[260,244])
t=R['problem2']
p(f"The fitted Student t has df = <b>{t['t_df']:.6f}</b>, location = {t['t_loc']:.9f}, and scale = {t['t_scale']:.9f}. Scale is not its standard deviation. All three parameters are fitted by maximum likelihood; removing the sample mean does not force the t location to zero. Normal VaR uses zero mean and sample SD; EW variance uses normalized weighted squared demeaned returns without a Bessel correction.")
h('(e) Actual order and the missed feature')
p('The actual order is <b>t &lt; equal-weight normal &lt; historical &lt; EW 0.97 &lt; EW 0.94</b>. The recent-volatility prediction was correct, but historical VaR was not the smallest. A 5% quantile depends on the location of the 25th worst return, not only on how many extraordinary events occur. Moderate losses and recent large losses together place it below the normal and t cutoffs. A heavy-tailed fitted t also need not have a larger 5% VaR: its estimated center and scale can give a narrower moderate quantile while leaving more probability in the far tails.')
h('(f) Within-regime volatility explains pooled kurtosis')
reg=load('p2_regimes')[['n','sd','excess_kurtosis']].copy(); reg.columns=['Days','Sample SD','Excess kurtosis']
table(reg,formats={'Sample SD':'.6f','Excess kurtosis':'.6f'})
p(f"Volatility rises from 1.018% to 2.429%, about 2.39 times larger, while excess kurtosis within the two visual regimes is close to zero. For zero-mean normal regimes with weights p_j, pooled excess kurtosis is 3 sum(p_j sigma_j<super>4</super>) / [sum(p_j sigma_j<super>2</super>)]<super>2</super> - 3. Substituting the two within-regime variances gives about {t['normal_mixture_excess']:.3f}, broadly explaining the observed 2.132. The comparison is approximate because the finite sample has different regime means and moments. The single fitted t describes the pooled mixture, not tomorrow's conditional volatility or evidence of one permanent t regime.")
h('(g) EW estimation noise and an explicitly subjective choice')
noise=load('p2_ew')[['sigma_se','VaR_se']].copy(); noise.columns=['Approx. SD SE','Approx. VaR SE ($)']
table(noise,formats={'Approx. SD SE':'.6f','Approx. VaR SE ($)':',.0f'})
p('The EW VaR gap is $4,088. Its corresponding volatility gap is 0.002486. This exceeds the 0.97 estimator\'s one-SE scale ($3,021), but is smaller than the 0.94 estimator\'s ($4,813); it is not overwhelmingly larger than estimation noise. A naive independent-error combination is $5,683, but these two estimators use the same observations and are correlated, so that is not a valid significance test. The Week 3 formula itself is only a stationary approximation after a regime shift.')
p('<b>Opinion:</b> I would use 0.94 for a trading limit to respond quickly to the persistent increase. I would use 0.97 for a more stable capital number, supplemented by stress scenarios. Its lower estimate can lag a new regime; stability alone does not make it conservative.')

page('Problem 3 - Predict and fit at 5%')
fig('p3_pnl.png','Figure 2. Annual P&L distributions. A log count scale makes the low-probability default clusters visible.')
h('(a) First four bond-return moments')
table(load('p3_moments'),formats={'mean':'.6f','variance':'.6f','skewness':'.4f','excess_kurtosis':'.4f'})
p('Each distribution has strongly negative skewness and excess kurtosis above 22. Most scenarios are near the non-default payoff return (100/90 - 1 = 11.11%), while a small cluster contains large default losses. A normal model cannot preserve that separation and its 5% cutoff can fall in a low-density gap. Mean and variance alone miss the default-threshold mechanism.')
h('(b) Prediction from severe-loss counts, before risk calculations')
p('A loses more than 20% in 406/10,000 scenarios (4.06%); B does so in 393 (3.93%). At least one does so in 782 (7.82%), with 17 joint events (0.17%). Thus 765 scenarios have exactly one severe loss. The concentrated 5% cutoff should remain outside the default cluster; the diversified cutoff should enter it. I predict <b>higher diversified 5% VaR but lower ES</b>: more scenarios contain a loss, but almost all default events expose only half the capital. Counts identify the threshold crossing; the P&amp;L plot also supports the loss-severity prediction for ES.')
h('(c-d) Historical risk and normal VaR, alpha = 5%')
v3=load('p3_risk')
table(v3[['VaR_5','ES_5','Normal_VaR_5']].rename(columns={'VaR_5':'Historical VaR ($)','ES_5':'Historical ES ($)','Normal_VaR_5':'Normal VaR ($)'}),formats={k:',.0f' for k in ['Historical VaR ($)','Historical ES ($)','Normal VaR ($)']})
p('Normal VaR is -mean(P&amp;L) + 1.644854 SD(P&amp;L), using each position\'s sample P&amp;L moments. Historical ES averages the worst 500 scenarios. A negative historical VaR for either single bond means that the 5% lower P&amp;L quantile is still a profit; it must not be reported as a positive loss by taking an absolute value.')

page('Problem 3 - Reconcile and change the tail level')
h('(e) Historical results at alpha = 1%')
table(v3[['VaR_1','ES_1']].rename(columns={'VaR_1':'VaR ($)','ES_1':'ES ($)'}),formats={'VaR ($)':',.0f','ES ($)':',.0f'})
h('(f) Explicit subadditivity checks at 5%')
va=v3.loc['A $1m','VaR_5']; vb=v3.loc['B $1m','VaR_5']; vc=v3.loc['A+B $2m','VaR_5']
ea=v3.loc['A $1m','ES_5']; eb=v3.loc['B $1m','ES_5']; ec=v3.loc['A+B $2m','ES_5']
p(f'VaR(A) + VaR(B) = {money(va)} + {money(vb)} = <b>{money(va+vb)}</b>, but VaR(A+B) = <b>{money(vc)}</b>. The required inequality VaR(A+B) &lt;= VaR(A) + VaR(B) fails by {money(vc-va-vb)}. This is a direct subadditivity violation, separate from comparing the two equal-capital choices.')
p(f'ES(A) + ES(B) = {money(ea)} + {money(eb)} = <b>{money(ea+eb)}</b>, whereas ES(A+B) = <b>{money(ec)}</b>. ES is subadditive here, with a margin of {money(ea+eb-ec)}. Compared with putting all $2 million in A, diversification lowers ES from {money(v3.loc["A $2m","ES_5"])} to {money(ec)}, while increasing VaR from {money(v3.loc["A $2m","VaR_5"])} to {money(vc)}.')
p('The mechanism is the 5% cutoff crossing a gap between repayment and default outcomes. Each bond individually defaults less often than 5%, so its VaR misses the default cluster. Combining two mostly different default sets pushes the event probability above 5%, pulling the portfolio cutoff into that cluster. ES includes the severity of the whole worst 5% in both choices, and benefits when the severe exposures do not coincide.')
h('(g) Is the normal answer right for the right reason?')
p('Normal VaR favors the $1 million/$1 million split: $143,466 versus $279,941 for $2 million in A. The sample bond-return correlation is only 0.00220, so diversification really does lower variance and historical ES. In that broad severity sense the preference is sensible. But for the specified historical 5% VaR it gives the opposite ordering. A normal approximation smooths the separated payoff clusters into one bell curve; it cannot explain the actual quantile crossing. Thus it is not right for the right reason if the objective is 5% VaR.')
h('(h) Why 1% VaR behaves differently')
vsum=v3.loc[['A $1m','B $1m'],'VaR_1'].sum()
p(f'At 1%, both the concentrated and diversified cutoffs are already inside default outcomes. Diversified VaR is {money(v3.loc["A+B $2m","VaR_1"])} versus {money(v3.loc["A $2m","VaR_1"])} for all in A, so diversification now lowers VaR. Also VaR(A+B) = {money(v3.loc["A+B $2m","VaR_1"])} &lt; {money(vsum)} = VaR(A) + VaR(B): subadditivity holds for these positions at this level. Joint severe losses occur in only 0.17% of scenarios, below 1%, so the 1% diversified cutoff is still largely a one-default outcome. VaR\'s failure depends on the tail level, event probabilities and payoff geometry; it does not have to fail at every alpha. ES remains subadditive at 1%.')

page('Problem 4 - Predict margins and dependence')
h('(a) Moments and a single-observation sensitivity check')
table(load('p4_moments'),formats={'mean':'.6f','variance':'.7f','skewness':'.4f','excess_kurtosis':'.4f'})
p('The moment-based prediction is normal for X1, t for X2 and X3. X1 has near-zero skew and modest negative excess kurtosis; X2 and X3 have substantial excess kurtosis. X2\'s negative skew initially looks incompatible with a symmetric t, so I check its largest absolute deviation from the median.')
sensitivity=load('p4_outlier_sensitivity')[['removed_day','removed_value','skewness','excess_kurtosis']]
sensitivity.columns=['Removed day','Removed return','New skewness','New excess kurtosis']
table(sensitivity,formats={'Removed return':'.6f','New skewness':'.4f','New excess kurtosis':'.4f'})
p('Day 952 is extreme in all three series. Removing it changes X2\'s skewness from -0.614 to +0.056 and excess kurtosis from 7.083 to 1.710. Negative skew is therefore highly sensitive to that day, while some heavy-tailed shape remains. This is a diagnostic only: <b>all observations are retained in every fit</b>.')
h('(b) Rank plots and simultaneous tails')
fig('p4_rank_pairs.png','Figure 3. Rank/(n+1) keeps empirical uniforms strictly inside (0,1); reference lines mark marginal 2.5% tails.')
table(load('p4_observed_tails'),widths=[168,168,168])
p('Under independence the expected count in either joint tail is 1,000 x 0.025<super>2</super> = <b>0.625 days</b> per pair. Observed counts of 6-12 and concentrated corners suggest stronger tail association than a loose central cloud. I predict the t copula wins. Positive Gaussian correlation itself raises counts above independence, so the independence comparison is suggestive rather than a test that uniquely establishes a t copula.')

page('Problem 4 - Fit margins and copulas')
h('(c) Maximum-likelihood margins and AICc')
marg=load('p4_margins',index_col=[0,1]).reset_index()
marg['df']=marg['df'].map(lambda v: 'normal limit' if v>1e8 else ('' if pd.isna(v) else f'{v:.4f}'))
table(marg[['asset','model','df','loc','scale','AICc']],formats={'loc':'.7f','scale':'.7f','AICc':'.3f'},index=False,widths=[35,67,66,103,103,130])
p('AICc = -2 log L + 2k + 2k(k+1)/(n-k-1), with k = 2 for a normal margin and k = 3 for a t margin. The choices agree with the prediction: <b>normal X1, t X2, t X3</b>. X1\'s t likelihood approaches the normal limit at very large degrees of freedom and does not earn its extra parameter. X2 and X3 improve AICc by about 109.69 and 62.32, respectively. All locations are estimated, and every scale is the distribution scale, not necessarily its SD.')
h('(d) Fitted uniforms and a common Kendall correlation matrix')
p('Transform each observed return with its selected fitted CDF: U_ij = F_j(r_ij). Estimate each Kendall tau on the original observations, then set R_ij = sin(pi tau_ij / 2). Monotone marginal transformations preserve these ranks. R is already positive definite; no repair is required.')
table(load('p4_R'),formats={c:'.6f' for c in ['X1','X2','X3']})
cop=load('p4_copulas')[['df','k','loglik','AICc','BIC']].copy(); cop.columns=['df','k','Log likelihood','AICc','BIC']
table(cop,formats={'df':'.5f','Log likelihood':'.4f','AICc':'.4f','BIC':'.4f'},widths=[84,72,30,110,104,104])
p('The Gaussian copula density is the multivariate normal density of z_j = Phi<super>-1</super>(u_j), divided by the product of univariate normal densities. For the t copula, replace both densities and inverse CDFs by t versions with common df. I sum log densities and optimize the t df by a one-dimensional search, with a log-spaced grid check. The fitted df is <b>3.65993</b>.')
p('As instructed in Week 5, margins and Kendall R are frozen for this comparison, so the remaining parameter counts are k_Gauss = 0 and k_t = 1. These are conditional copula information criteria, not full joint marginal-plus-copula likelihood scores. Counting the three R entries as newly fitted parameters would be a different convention; it is not the one used here.')
h('(g) Which model wins?')
co=load('p4_copulas')
p(f'The t copula gains <b>{R["problem4"]["loglik_gain"]:.4f}</b> log-likelihood units. Its AICc is lower by <b>{co.loc["Gaussian","AICc"]-co.loc["Student t","AICc"]:.4f}</b> and BIC by <b>{co.loc["Gaussian","BIC"]-co.loc["Student t","BIC"]:.4f}</b>. Both strongly favor t, in agreement with the pre-fit prediction. This is an in-sample model comparison under the chosen margins, not a guarantee about future tail calibration.')

page('Problem 4 - Simulated and historical portfolio risk')
h('(e) 100,000 scenarios per copula; $1 million in each asset')
v4=load('p4_risk',index_col=[0,1]).reset_index(); v4['alpha']=v4.alpha.map(lambda x:f'{x:.0%}')
v4=v4.rename(columns={'alpha':'Tail level','VaR':'VaR ($)','ES':'ES ($)'})
table(v4,index=False,formats={'VaR ($)':',.0f','ES ($)':',.0f'},widths=[150,94,130,130])
p('For Gaussian simulation, draw correlated normal scores and apply Phi. For t simulation, divide the same correlated normal scores by one shared sqrt(chi-square(df)/df) per day, then apply the t CDF. Apply each fitted marginal inverse CDF to those uniforms and sum the three dollar P&amp;Ls. Sharing the scale within a t scenario is essential for tail dependence. Marginal means are retained because this problem does not impose zero expected returns.')
h('(f-g) Implied joint tail days per 1,000 days')
obs=load('p4_observed_tails'); simulated=load('p4_simulated_tails',index_col=[0,1])
counts=[]
for pair in obs.index:
    counts.append([pair,obs.loc[pair,'lower'],obs.loc[pair,'upper'],*simulated.loc[('Gaussian',pair)].values,*simulated.loc[('Student t',pair)].values])
countframe=pd.DataFrame(counts,columns=['Pair','Data low','Data high','Gauss low','Gauss high','t low','t high'])
table(countframe,index=False,formats={c:'.2f' for c in countframe.columns[3:]},widths=[78,62,64,76,76,74,74])
p('Counts use true simulated marginal uniforms below 0.025 or above 0.975, then multiply scenario frequencies by 1,000. They are expected counts, not rounded actual days. The t copula predicts about 7.5-9.6 joint-tail days; Gaussian predicts about 4.2-6.1. The t model generally better captures the larger observed counts, especially upper tails, but overpredicts some lower counts and misses the X1-X3 upper count of 12. Both copulas are symmetric, so they cannot systematically fit lower/upper asymmetry. One thousand observations leave these rare-event counts noisy.')
h('(h) Which risk number changes most?')
vr=load('p4_risk',index_col=[0,1])
change=[]
for a in [.05,.01]:
    for metric in ['VaR','ES']:
        g=vr.loc[('Gaussian',a),metric]; tv=vr.loc[('Student t',a),metric]
        change.append([f'{a:.0%} {metric}',tv-g,100*(tv/g-1)])
table(pd.DataFrame(change,columns=['Measure','t minus Gaussian ($)','Change (%)']),index=False,formats={'t minus Gaussian ($)':'+,.0f','Change (%)':'+.2f'},widths=[168,168,168])
p('The biggest move is <b>1% ES</b>, up about $7,490 (7.93%). A common t scale increases the chance of simultaneous extreme marginal losses, reducing diversification precisely in the deepest tail. ES averages their magnitudes rather than just locating a cutoff. The 5% VaR barely moves because margins and central rank association are fixed, and that moderate quantile is less sensitive to the extreme joint tail. Tail dependence does not imply every finite-level VaR must rise monotonically.')

page('Problem 4 - What the likelihood gain is scoring')
h('(i) Daily log-likelihood contributions')
rp=R['problem4']
p('For every day I compute d_i = log c_t(U_i; R, df) - log c_Gauss(U_i; R). All 1,000 daily values are saved in tables/p4_daily_likelihood.csv. The main definition of a central day uses empirical ranks: all three series must lie strictly inside their own 5th-95th percentile range. This matches the question\'s description of observations sitting outside their own marginal tails.')
central=pd.DataFrame({'Days':[rp['central_days'],1000-rp['central_days'],1000],
                      'Log-likelihood gain':[rp['central_gain'],rp['loglik_gain']-rp['central_gain'],rp['loglik_gain']],
                      'Share (%)':[100*rp['central_fraction'],100*(1-rp['central_fraction']),100]},index=['All margins central','At least one outer-5%','Total'])
table(central,formats={'Log-likelihood gain':'.4f','Share (%)':'.2f'},widths=[200,66,134,104])
p(f'<b>{rp["central_fraction"]:.2%} of the net likelihood improvement comes from the {rp["central_days"]} days when no asset is in either outer 5% tail.</b> The rest comes from the {1000-rp["central_days"]} days with at least one tail observation; those need not be joint tail days. As a sensitivity, defining central days by fitted uniforms instead yields {rp["fitted_u_central_days"]} days and a gain of {rp["fitted_u_central_gain"]:.4f}, leading to the same conclusion.')
p('Information criteria score the entire joint density, not only joint crashes. The t copula changes central concentration and the distribution of moderate co-movements as well as asymptotic tail dependence. Its strong AICc/BIC advantage therefore cannot be interpreted as 101.56 units of evidence coming only from extreme-tail events. Some daily contributions can be negative; these reported shares are contributions to the net gain, not proportions of days that favor t.')
h('(j) Tail dependence for the most correlated pair')
p('The largest fitted correlation is X1-X2, rho = 0.583263. For a t copula, the lower and upper tail-dependence coefficients are identical:')
p('<b>lambda_L = lambda_U = 2 T_(df+1)(-sqrt[(df+1)(1-rho)/(1+rho)]).</b>')
p(f'At df = {rp["copula_df"]:.5f}, this gives <b>{rp["tail_dependence"]:.6f}</b>. It is the limiting conditional probability that one variable is in an ever more extreme tail given the other is in the same tail. The Gaussian copula coefficient is <b>zero</b> because rho is strictly less than one. Zero asymptotic dependence does not mean independent returns or zero finite-2.5% co-exceedances.')
h('Monte Carlo precision and finite historical tails')
mc=load('p4_mc_error',index_col=[0,1]).reset_index(); mc['alpha']=mc.alpha.map(lambda x:f'{x:.0%}')
mc.columns=['Copula','Tail level','VaR MC SE ($)','ES MC SE ($)']
table(mc,index=False,formats={'VaR MC SE ($)':',.0f','ES MC SE ($)':',.0f'},widths=[150,94,130,130])
p('These are rough standard errors of a full 100,000-scenario estimate, using ten independent batches and dividing their standard deviation by sqrt(10). They quantify simulation noise only, not parameter or model uncertainty. The historical 1% ES averages just ten days and includes day 952, so its close agreement with the t estimate should not be treated as precise validation. Common random normal draws reduce some comparison noise, but the two copulas still have different random scale structures.')

page('Problem 5 - Predict and fit the factor models')
h('(a) Prediction from the portfolio variance formula')
p('Write r_A = alpha_A + beta_A r_M + epsilon_A and likewise for B, with the market uncorrelated with the regression errors. Let c = Cov(epsilon_A, epsilon_B). An omitted shared industry factor suggests c &gt; 0. In dollar units, the residual part of P1 variance is 10<super>12</super>(s_A<super>2</super> + s_B<super>2</super> + 2c), whereas P2 uses 10<super>12</super>(s_A<super>2</super> + s_B<super>2</super> - 2c).')
p('Setting c to zero should therefore <b>understate P1 VaR and overstate P2 VaR</b>. Under zero-mean normal P&amp;L, VaR is proportional to SD, so the variance direction is the VaR direction. I expect P2 to be more sensitive in percentage terms because similar market betas largely cancel in a long-short position, leaving residual differences as its main risk.')
h('(b) OLS regression on 500 arithmetic returns')
reg5=load('p5_regressions'); reg5.index.name='Stock'
reg5.columns=['Intercept alpha','Market beta','Residual SD (n-1)','Residual SD (n-2)']
table(reg5,formats={'Intercept alpha':'.9f','Market beta':'.6f','Residual SD (n-1)':'.8f','Residual SD (n-2)':'.8f'},widths=[45,107,100,126,126])
p(f'The market sample SD is {R["problem5"]["market_sd"]:.8f}; the residual correlation is <b>{R["problem5"]["residual_correlation"]:.6f}</b>. This strong positive residual association confirms that the market alone leaves substantial shared movement unexplained. It is consistent with a shared omitted factor, though the data do not identify that factor as an industry effect.')
h('Residual covariance used for simulation')
table(load('p5_residual_covariance'),formats={'A':'.9f','B':'.9f'},widths=[70,217,217])
p('I use denominator n - 1 for the market variance, sample stock covariance and covariance of the centered OLS residuals. This common convention gives an exact sample covariance decomposition. The n - 2 residual SDs shown above are the usual regression-error estimates with two fitted coefficients; they are not mixed with n - 1 covariances in the simulation. That choice is explicit because inconsistent degrees-of-freedom corrections can create a small artificial mismatch.')
h('Zero expected returns and the simulation construction')
p('I report the fitted alphas, but omit them from simulated returns to honor the instruction to assume zero expected returns. Draw independent zero-mean market and bivariate residual vectors; use either the full residual covariance or its diagonal. Form simulated stock returns as beta_i r_M + epsilon_i. A shared random market and common base normal residual draws are used for the two assumptions to reduce comparison noise.')

page('Problem 5 - Portfolio VaR and reconciliation')
h('(c-d) 100,000 simulated days and direct delta-normal VaR')
v5=load('p5_var',index_col=[0,1]).reset_index()
v5['assumption']=v5.assumption.replace({'Full residual covariance':'Full residuals','Independent residuals':'Independent'})
v5.columns=['Residual assumption','Portfolio','Simulated VaR ($)','Model formula ($)','Delta-normal ($)','MC SE ($)']
table(v5,index=False,formats={c:',.0f' for c in v5.columns[2:]},widths=[98,54,92,96,96,68])
p('The model formula uses Sigma_model = beta beta\' Var(r_M) + Sigma_e and VaR = 1.644854 sqrt(w\' Sigma_model w). Direct delta-normal VaR uses the stock-return sample covariance instead. Both use zero mean and alpha = 5%. MC SE is the same ten-batch approximation as in Problem 4. Differences of tens or hundreds of dollars between simulated and analytical answers are consistent with Monte Carlo noise.')
h('(e) Direction and size of the omitted-correlation effect')
vv=load('p5_var',index_col=[0,1]); p1f=vv.loc[('Full residual covariance','P1'),'analytical_VaR']; p1i=vv.loc[('Independent residuals','P1'),'analytical_VaR']; p2f=vv.loc[('Full residual covariance','P2'),'analytical_VaR']; p2i=vv.loc[('Independent residuals','P2'),'analytical_VaR']
p(f'Both predictions are confirmed. Using analytical values to remove simulation noise, the shortcut reduces P1 VaR by {money(p1f-p1i)} ({100*(1-p1i/p1f):.2f}%), but raises P2 VaR by {money(p2i-p2f)} ({100*(p2i/p2f-1):.2f}%). P2 is much more exposed in relative terms. Its market loading is beta_A - beta_B = 0.099861 versus beta_A + beta_B = 1.970695 for P1. Ignoring positively correlated residuals discards the long-short hedge that cancels shared shocks.')
h('(f) Why full residual simulation agrees with delta-normal')
p('With an intercept, OLS makes each residual vector orthogonal to the observed market vector and to the constant. Thus, on the same rows and with consistent covariance divisors, <b>S_stocks = beta beta\' s_M<super>2</super> + S_residuals</b> exactly. In this calculation the maximum entrywise discrepancy is 1.63 x 10<super>-19</super>. This covariance identity is an OLS algebra result; normality is not needed for the identity itself.')
p('Under the imposed joint normal model, zero covariance between the market and residuals also implies independence. A sum of these normal components is multivariate normal with precisely the stock sample covariance. Any linear portfolio therefore has the same normal distribution, and consequently the same theoretical VaR, under full residual simulation and delta-normal. The displayed small simulation differences vanish as the number of draws grows.')
p('Agreement can fail if residual correlation is dropped, different windows or EW weights are used, covariance divisors differ, or alphas are added to only one model. With non-normal residuals, nonlinear factor relationships, or nonlinear portfolio payoffs, matching covariance no longer ensures matching tail quantiles. In a non-normal model OLS orthogonality also need not imply independence; independently simulating market and residuals may then destroy dependence that matters for VaR.')
h('Source and reproduction notes')
p('Problem statements and input data: class/Assignments/Assignment2/assignment.qmd and problem1.csv through problem5.csv. Methods and conventions follow the local Week 2-5 course materials, especially Week 3 exponential weighting/PSD repair and Week 5 copula comparison/OLS. No market data or external estimates were added. Run analysis.py, then build_report.py; README.md gives the complete setup commands.',note=True)

def footer(canvas,doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor('#ccd8dd')); canvas.setLineWidth(.4)
    canvas.line(54,43,558,43)
    canvas.setFont('Helvetica',8); canvas.setFillColor(colors.HexColor('#516773'))
    canvas.drawString(54,31,'FinTech 545 | Assignment 2 | Yiting Sun')
    canvas.drawRightString(558,31,str(doc.page))
    canvas.restoreState()

doc=SimpleDocTemplate(str(ROOT/'Assignment2_Report.pdf'),pagesize=(612,792),rightMargin=54,leftMargin=54,topMargin=43,bottomMargin=56,
                      title='Assignment 2 - Covariance, VaR, and Copulas',author='Yiting Sun')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
(ROOT/'report.md').write_text('\n'.join(markdown))
print('Wrote Assignment2_Report.pdf and report.md')
