from __future__ import division
import pandas as pd
import numpy as np

df_G1 = pd.read_csv('ins/DH2 out.csv', index_col=0)
df_G2 = pd.read_csv('ins/DH1 out.csv', index_col=0)
df_in = pd.read_csv('ins/yields.csv')

predictor_year = 1997
missing_year = 1998

v1 = 0.981
v2 = 1.086
vd = 0.266

missing_hybrids = df_in[df_in['YR'] == missing_year]['NEWENT'].values.tolist()
predictor_hybrids = df_in[df_in['YR'] == predictor_year]['NEWENT'].values.tolist()
predictor_index2 = df_in[df_in['YR'] == predictor_year]['INDEX2'].values.tolist()

covariances = [] #list of lists. Each element is [missing name, predictor name, covariance]
dot_products = [] #list of lists. Each element is [missing name, dot_product]

#loop through the missing hybrids and, for each, loop through the predictor hybrids to calculate covariances
ph_length = len(predictor_hybrids)
for mh in missing_hybrids:
    mh_ls = mh.split('/')
    mh_female = mh_ls[0]
    mh_male = mh_ls[1]
    dot_prod = 0
    for i, ph in enumerate(predictor_hybrids):
        ph_ls = ph.split('/')
        ph_female = ph_ls[0]
        ph_male = ph_ls[1]
        gxx = df_G1.loc[mh_female, ph_female]
        fyy = df_G2.loc[mh_male, ph_male]
        covar = gxx*v1 + fyy*v2 + gxx*fyy*vd
        covariances.append([mh, ph, covar])
        dot_prod += covar*predictor_index2[i]
    dot_products.append([mh,dot_prod/ph_length])

df_covar = pd.DataFrame(covariances, columns=['missing','predictor','covar'])
df_covar = df_covar.pivot_table(index='missing', columns='predictor', values='covar')
df_covar.to_csv('outs/covariances.csv')
df_dot_prod = pd.DataFrame(dot_products, columns=['missing','dot_product'])
df_dot_prod.to_csv('outs/dot_products.csv', index=False)

#Now find V 
V = []
predictor_hybrids2 = list(predictor_hybrids)
for i1, ph1 in enumerate(predictor_hybrids):
    ph1_ls = ph1.split('/')
    ph1_female = ph1_ls[0]
    ph1_male = ph1_ls[1]
    for i2, ph2 in enumerate(predictor_hybrids2):
        ph2_ls = ph2.split('/')
        ph2_female = ph2_ls[0]
        ph2_male = ph2_ls[1]
        gxx = df_G1.loc[ph1_female, ph2_female]
        fyy = df_G2.loc[ph1_male, ph2_male]
        covar = gxx*v1 + fyy*v2 + gxx*fyy*vd
        if i1 == i2:
            covar += 1
        V.append([ph1, ph2, covar])

df_V = pd.DataFrame(V, columns=['ph1','ph2','covar'])
df_V = df_V.pivot_table(index='ph1', columns='ph2', values='covar')
df_V.to_csv('outs/V.csv')

#now calculate missing yields as ym = C*V-1*yp
yp = np.asarray(predictor_index2)
C = df_covar.values
V_np = df_V.values
V_inv = np.linalg.inv(V_np)
weights = C.dot(V_inv)
df_weights = pd.DataFrame(weights, index=missing_hybrids, columns=predictor_hybrids)
df_weights.to_csv('outs/weights.csv')
ym = weights.dot(yp)
df_ym = pd.DataFrame(ym, index=missing_hybrids, columns=['moretrue'])
df_dp = df_dot_prod.set_index('missing')
df_ym = pd.merge(df_ym, df_dp, left_index=True, right_index=True)
df_ym.to_csv('outs/yields_missing.csv')
