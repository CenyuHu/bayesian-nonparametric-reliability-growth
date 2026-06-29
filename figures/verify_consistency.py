#!/usr/bin/env python3
"""Cross-check all data sources for consistency."""
import json, numpy as np

with open('simulation_results_v3.json','r',encoding='utf-8') as f:
    sim = json.load(f)
with open('case_study_results.json','r',encoding='utf-8') as f:
    case = json.load(f)
with open('mcmc_diagnostics.json','r',encoding='utf-8') as f:
    mcmc = json.load(f)

s = sim['summary']
errors = []

def check(label, latex_val, json_val, tol=0.15):
    if abs(latex_val - json_val) > tol:
        errors.append(f"FAIL {label}: LaTeX={latex_val}, JSON={json_val}")
        return False
    return True

# ===== 1. JSON CONFIG =====
print("="*60)
print("1. CONFIGURATION")
print("="*60)
cfg = sim['config']
print(f"  simulation_results_v3.json: n_iter={cfg['n_iter']}, n_burn={cfg['n_burn']}")
mc = mcmc['config']
print(f"  mcmc_diagnostics.json:     n_iter={mc['n_iter']}, n_burn={mc['n_burn']}")
if cfg['n_iter'] != 20000 or cfg['n_burn'] != 10000:
    errors.append("FAIL sim config: n_iter/n_burn not 20000/10000")
if mc['n_iter'] != 20000 or mc['n_burn'] != 10000:
    errors.append("FAIL mcmc config: n_iter/n_burn not 20000/10000")
print(f"  K_trunc={cfg['k_trunc']}, K_bound(K=8)={cfg['truncation_error_bound']['bound_L1']:.2e}")
print(f"  K_bound(K=15)={cfg['truncation_error_K15']['bound_L1']:.2e}")

# ===== 2. CASE STUDY =====
print("\n" + "="*60)
print("2. CASE STUDY VALUES (case_study_results.json)")
print("="*60)
print(f"  psi: mean={case['psi']['mean']}, hpd=[{case['psi']['hpd_low']}, {case['psi']['hpd_high']}], prob_neg={case['psi']['prob_neg']}")
print(f"  rho2: mean={case['rho2']['mean']}, hpd=[{case['rho2']['hpd_low']}, {case['rho2']['hpd_high']}]")
print(f"  rho3: mean={case['rho3']['mean']}, hpd=[{case['rho3']['hpd_low']}, {case['rho3']['hpd_high']}]")
print(f"  prob(rho2<rho3)={case['prob_rho2_lt_rho3']}")
print(f"  occupied={case['occupied']}")
print(f"  mu_gamma={case['mu_gamma']}")

# LaTeX claims: psi=0.49, hpd=[-2.06,4.42], prob_neg=0.37
# rho2=0.51, hpd=[0.06,0.76], rho3=0.62, hpd=[0.11,0.82]
# prob(rho2<rho3)=0.63, occupied=[1.64,1.95,2.17]

# ===== 3. TABLE 3: RMSE n_l=5 =====
print("\n" + "="*60)
print("3. TABLE 3: RMSE n_l=5 (LaTeX vs JSON)")
print("="*60)
latex_t3 = {
    ('A','DDP-Joint'):[24.1,196.4,1686.5],('A','DDP-Only'):[13.7,69.9,577.1],
    ('A','Separate'):[19.7,122.4,1208.7],('A','LN-Bayes'):[20.8,80.3,456.0],
    ('A','Wbl-Bayes'):[15.5,83.3,316.5],('A','Ind-DPM'):[19.7,121.0,1195.7],
    ('B','DDP-Joint'):[7.2,61.9,479.7],('B','DDP-Only'):[3.2,19.9,198.2],
    ('B','Separate'):[5.8,36.3,346.4],('B','LN-Bayes'):[5.8,20.2,113.4],
    ('B','Wbl-Bayes'):[3.7,22.8,73.5],('B','Ind-DPM'):[5.8,36.6,348.4],
    ('C','DDP-Joint'):[10.6,176.6,1707.7],('C','DDP-Only'):[5.7,68.1,815.4],
    ('C','Separate'):[10.1,108.2,1221.0],('C','LN-Bayes'):[10.7,74.0,492.2],
    ('C','Wbl-Bayes'):[5.2,77.6,344.5],('C','Ind-DPM'):[10.1,108.7,1223.7],
    ('D','DDP-Joint'):[44.2,338.1,2380.5],('D','DDP-Only'):[23.0,102.9,859.1],
    ('D','Separate'):[31.1,188.9,1706.0],('D','LN-Bayes'):[29.9,111.7,630.7],
    ('D','Wbl-Bayes'):[26.9,136.0,418.9],('D','Ind-DPM'):[31.1,190.6,1705.8],
}
t3_ok = 0; t3_fail = 0
for (sc,m), lv in latex_t3.items():
    jv = s[sc]['5'][m]['rmse']
    for qi in range(3):
        if check(f"T3 {sc}/{m} q{qi}", lv[qi], jv[qi], 0.15):
            t3_ok += 1
        else:
            t3_fail += 1
print(f"  Result: {t3_ok} OK, {t3_fail} failed")

# ===== 4. TABLE 4: COVERAGE n_l=10 =====
print("\n" + "="*60)
print("4. TABLE 4: COVERAGE n_l=10 (LaTeX vs JSON)")
print("="*60)
latex_t4 = {
    ('A','DDP-Joint'):([0.75,1.00,1.00],1362),('A','DDP-Only'):([0.70,0.80,1.00],444),
    ('A','Separate'):([1.00,1.00,1.00],1213),('A','LN-Bayes'):([0.95,0.95,0.95],363),
    ('A','Wbl-Bayes'):([0.00,0.00,0.00],20),('A','Ind-DPM'):([1.00,1.00,1.00],1214),
    ('B','DDP-Joint'):([0.95,1.00,1.00],381),('B','DDP-Only'):([0.95,0.95,1.00],151),
    ('B','Separate'):([1.00,0.95,1.00],329),('B','LN-Bayes'):([0.85,0.80,0.95],86),
    ('B','Wbl-Bayes'):([0.00,0.00,0.00],16),('B','Ind-DPM'):([1.00,0.95,1.00],326),
    ('C','DDP-Joint'):([1.00,1.00,1.00],1452),('C','DDP-Only'):([0.95,0.85,0.90],485),
    ('C','Separate'):([1.00,0.90,1.00],1150),('C','LN-Bayes'):([0.80,0.85,0.95],424),
    ('C','Wbl-Bayes'):([0.00,0.00,0.00],17),('C','Ind-DPM'):([1.00,0.90,1.00],1139),
    ('D','DDP-Joint'):([0.80,1.00,1.00],1826),('D','DDP-Only'):([0.80,0.85,1.00],683),
    ('D','Separate'):([1.00,1.00,1.00],1569),('D','LN-Bayes'):([1.00,0.80,1.00],449),
    ('D','Wbl-Bayes'):([0.00,0.00,0.00],19),('D','Ind-DPM'):([1.00,1.00,1.00],1560),
}
t4_ok = 0; t4_fail = 0
for (sc,m), (lcov, lwid) in latex_t4.items():
    jcov = s[sc]['10'][m]['coverage']
    jwid = s[sc]['10'][m]['width_avg']
    for qi in range(3):
        if check(f"T4 {sc}/{m} cov_q{qi}", lcov[qi], jcov[qi], 0.015):
            t4_ok += 1
        else:
            t4_fail += 1
    if check(f"T4 {sc}/{m} width", lwid, jwid, 1.5):
        t4_ok += 1
    else:
        t4_fail += 1
print(f"  Result: {t4_ok} OK, {t4_fail} failed")

# ===== 5. TABLE 5: WAIC n_l=5 =====
print("\n" + "="*60)
print("5. TABLE 5: WAIC n_l=5 (LaTeX vs JSON)")
print("="*60)
latex_t5 = {
    ('A','DDP-Joint'):25.7,('A','DDP-Only'):17.7,('A','Separate'):15.3,
    ('A','LN-Bayes'):15.7,('A','Wbl-Bayes'):32.3,('A','Ind-DPM'):15.3,
    ('B','DDP-Joint'):27.7,('B','DDP-Only'):17.1,('B','Separate'):16.9,
    ('B','LN-Bayes'):18.3,('B','Wbl-Bayes'):27.0,('B','Ind-DPM'):17.2,
    ('C','DDP-Joint'):28.6,('C','DDP-Only'):19.9,('C','Separate'):19.6,
    ('C','LN-Bayes'):22.1,('C','Wbl-Bayes'):31.2,('C','Ind-DPM'):19.5,
    ('D','DDP-Joint'):30.2,('D','DDP-Only'):16.8,('D','Separate'):16.0,
    ('D','LN-Bayes'):16.6,('D','Wbl-Bayes'):35.8,('D','Ind-DPM'):16.1,
}
t5_ok = 0; t5_fail = 0
for (sc,m), lv in latex_t5.items():
    jv = s[sc]['5'][m]['waic']
    if check(f"T5 {sc}/{m}", lv, jv, 0.15):
        t5_ok += 1
    else:
        t5_fail += 1
print(f"  Result: {t5_ok} OK, {t5_fail} failed")

# ===== 6. LOGIC CHECKS =====
print("\n" + "="*60)
print("6. LOGIC & NARRATIVE CONSISTENCY CHECKS")
print("="*60)

# 6a. DDP-Joint RMSE > DDP-Only RMSE at n_l=5 (all scenarios, q50)
print("\n6a. Finding 2: DDP-Joint RMSE > DDP-Only at n_l=5?")
for sc in ['A','B','C','D']:
    dj = s[sc]['5']['DDP-Joint']['rmse'][1]
    do = s[sc]['5']['DDP-Only']['rmse'][1]
    ok = dj > do
    print(f"  Scen {sc}: DDP-Joint={dj:.1f} > DDP-Only={do:.1f}? {'OK' if ok else 'FAIL'}")
    if not ok: errors.append(f"DDP-Joint not > DDP-Only at Scen {sc}")

# 6b. DDP-Only RMSE < Ind-DPM at q50 (37-46%)
print("\n6b. Finding 1: DDP-Only q50 RMSE reduction vs Ind-DPM (37-46%)?")
for sc in ['A','B','C','D']:
    do = s[sc]['5']['DDP-Only']['rmse'][1]
    ind = s[sc]['5']['Ind-DPM']['rmse'][1]
    reduction = (ind - do) / ind * 100
    in_range = 37 <= reduction <= 46
    print(f"  Scen {sc}: reduction={reduction:.1f}% (37-46%?) {'OK' if in_range else 'WARN out of range'}")
    if not in_range: errors.append(f"DDP-Only reduction {reduction:.1f}% not in 37-46% at Scen {sc}")

# 6c. Weibull zero coverage
print("\n6c. Finding 4: Weibull zero coverage at n_l=10?")
for sc in ['A','B','C','D']:
    wcov = s[sc]['10']['Wbl-Bayes']['coverage']
    all_zero = all(c == 0.0 for c in wcov)
    print(f"  Scen {sc}: coverage={wcov} → {'OK all zero' if all_zero else 'FAIL NON-ZERO'}")
    if not all_zero: errors.append(f"Weibull non-zero coverage at Scen {sc}")

# 6d. WAIC: DDP-Joint > DDP-Only
print("\n6d. WAIC: DDP-Joint > DDP-Only at n_l=5?")
for sc in ['A','B','C','D']:
    dj = s[sc]['5']['DDP-Joint']['waic']
    do = s[sc]['5']['DDP-Only']['waic']
    delta = dj - do
    ok = delta > 0
    print(f"  Scen {sc}: ΔWAIC={delta:.1f} (>0?) {'OK' if ok else 'FAIL'}")
    if not ok: errors.append(f"DDP-Joint WAIC not > DDP-Only at Scen {sc}")

# 6e. Delta WAIC range (DDP-Joint vs DDP-Only)
deltas = []
for sc in ['A','B','C','D']:
    deltas.append(s[sc]['5']['DDP-Joint']['waic'] - s[sc]['5']['DDP-Only']['waic'])
print(f"\n6e. ΔWAIC(DDP-Joint - DDP-Only) range: {min(deltas):.1f}–{max(deltas):.1f}")
print(f"  LaTeX claims: 8.0–13.4")
if abs(min(deltas) - 8.0) > 0.2 or abs(max(deltas) - 13.4) > 0.2:
    errors.append(f"ΔWAIC range {min(deltas):.1f}-{max(deltas):.1f} ≠ 8.0-13.4")

# 6f. n_l=10 RMSE reduction
print("\n6f. Finding 1: DDP-Only q50 RMSE reduction vs Ind-DPM at n_l=10 (20-25%)?")
for sc in ['A','B','C','D']:
    do = s[sc]['10']['DDP-Only']['rmse'][1]
    ind = s[sc]['10']['Ind-DPM']['rmse'][1]
    reduction = (ind - do) / ind * 100
    in_range = 20 <= reduction <= 25
    print(f"  Scen {sc}: reduction={reduction:.1f}% (20-25%?) {'OK' if in_range else 'WARN'}")
    if not in_range: errors.append(f"n_l=10 reduction {reduction:.1f}% not in 20-25%")

# 6g. Case study psi non-identifiability logic
print("\n6g. Case study: psi posterior covers zero?")
psi_low = case['psi']['hpd_low']
psi_high = case['psi']['hpd_high']
psi_mean = case['psi']['mean']
prob_neg = case['psi']['prob_neg']
print(f"  psi mean={psi_mean}, HPD=[{psi_low}, {psi_high}], Pr(psi<0)={prob_neg}")
print(f"  HPD spans zero? {'OK' if psi_low < 0 < psi_high else 'FAIL'}")
print(f"  Pr(psi<0)={prob_neg} ≈ 0.37? {'OK' if abs(prob_neg-0.37)<0.02 else 'WARN'}")

# 6h. Reliability: DDP SD >> LN SD
print("\n6h. DDP reliability SD substantially larger than LN?")
for tkey in ['t300','t450']:
    for st in ['stage1','stage2','stage3']:
        ddp_sd = case['reliability'][tkey][st]['ddp_sd']
        ln_sd = case['reliability'][tkey][st]['ln_sd']
        ratio = ddp_sd / ln_sd
        print(f"  {tkey}/{st}: DDP_SD={ddp_sd:.3f}, LN_SD={ln_sd:.3f}, ratio={ratio:.1f}x")

# ===== 7. SUMMARY =====
print("\n" + "="*60)
print("7. OVERALL VERDICT")
print("="*60)
if errors:
    print(f"\n  {len(errors)} ERRORS FOUND:")
    for e in errors:
        print(f"    {e}")
else:
    print(f"\n  OK ALL CHECKS PASSED — no inconsistencies found")
print()
