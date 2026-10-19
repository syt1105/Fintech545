# Predictions recorded before fitting

These predictions were recorded after `prepare.py` produced counts, descriptive
moments and plots, and before `analysis.py` was run. They are kept unchanged so
the reconciliation can distinguish an initial expectation from a fitted result.

1. Pairwise correlations can be indefinite because entries use different samples;
   complete-case correlation is a Gram matrix and is PSD (strict positive
   definiteness is not guaranteed). The almost-redundant index makes one true
   eigenvalue close to zero, so small sampling discrepancies can cross the PSD
   boundary. Correlations involving D are least reliable (83-96 paired days),
   especially C-D (83); all five overlap on only 81 days.
2. The return plot suggests a volatility increase around day 461: use the final
   40 days as an approximate visual regime, not a fitted breakpoint. Expected
   ascending 5% VaR: historical, fitted t, equal-weight normal, EW normal 0.97,
   EW normal 0.94. The first three are a tentative ordering: historical 5% tails
   and a fitted t can mostly reflect the old calm regime, whereas a normal
   variance absorbs all large observations. EW gives the recent high-volatility
   period much greater weight. Excess kurtosis 2.13 seems inconsistent with a
   stable normal regime as a convincing model, though it is not mathematically
   impossible in a finite normal sample.
3. A has 406 severe-loss scenarios, B 393, and either defaults in 782 out of
   10,000. At 5%, concentration's cutoff should remain in the non-default region
   but diversification's cutoff should enter the one-default region. Predict
   larger diversified VaR but smaller diversified ES: losses occur more often,
   but most involve only half the investment. Only 17 scenarios have both severe
   losses. The very negative skew and excess kurtosis around 22-23 make a normal
   model a poor description of default risk.
4. Predict normal for X1 and t for X2 and X3. X2's skewness changes from -0.614
   to +0.056 after removing day 952; excess kurtosis falls from 7.083 to 1.710.
   Thus symmetric heavy tails are plausible and negative skew is not robust.
   Retain that observation in all fits. Each pair has 6-12 simultaneous tail
   days per side, compared with 0.625 under independence. The corners of the
   rank plots look tightly associated despite broad dispersion in the center.
   Predict the t copula will win, while recognizing positive Gaussian
   correlation also produces more joint tail days than independence.
5. For positive industry residual covariance c, removing it subtracts 2c from
   the variance of A+B and adds 2c to the variance of A-B (before dollar scaling).
   Predict understated P1 VaR and overstated P2 VaR under independent errors.
   P2 should be especially sensitive because similar market betas cancel.
