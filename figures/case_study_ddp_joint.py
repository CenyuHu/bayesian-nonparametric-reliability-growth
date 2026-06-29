#!/usr/bin/env python3
"""
Case Study: DDP-Joint MCMC for Semiconductor Data (FINAL VERSION)
==================================================================
Fits the full DDP-Joint model to Liu2026a semiconductor data.
Uses informative priors on psi to regularize the weak likelihood
identification with n=5 per stage.
"""
import numpy as np
from scipy import stats, linalg
from scipy.special import logsumexp
import json, time, os, sys

# ============================================================
N_ITER   = 20000
N_BURN   = 10000
K_TRUNC  = 8
N_STAGES = 3
N_UNITS  = [5, 5, 5]

RNG_SEED = 42
# ============================================================

# --- Data: Liu2026a failure times ---
W1 = np.array([5.927, 6.066, 6.186, 6.250, 6.313]); C1 = np.array([np.inf]*5); D1 = np.array([1,1,1,1,1])
W2 = np.array([6.127, 6.221, 6.275, 6.349, 6.377]); C2 = np.array([np.inf,np.inf,np.inf,np.inf,6.377]); D2 = np.array([1,1,1,1,0])
W3 = np.array([6.250, 6.304, 6.299, 6.384, 6.408]); C3 = np.array([np.inf,6.304,np.inf,np.inf,6.408]); D3 = np.array([1,0,0,1,0])
W_ALL = [W1,W2,W3]; C_ALL = [C1,C2,C3]; D_ALL = [D1,D2,D3]
STAGE_MEAN = np.array([np.mean(W1[D1==1]), np.mean(W2[D2==1]), np.mean(W3[D3==1])])

# --- Generate degradation data ---
M_LI = 10; TAU = np.linspace(0.05, 0.95, M_LI)
MU_GAMMA_TRUE = np.array([1.0, 0.7, 0.4]); SG2_GAMMA_TRUE = 0.3
W2_XI=0.5; PHI_XI=0.3; NU2_XI=0.01; PSI_TRUE=-1.62

np.random.seed(2026)
Z_ALL = []; GAM_TRUE = []; ETA_TRUE = []
for l in range(N_STAGES):
    zl=[]; gl=[]; el=[]
    for i in range(N_UNITS[l]):
        if D_ALL[l][i]==1:
            gc = (W_ALL[l][i]-STAGE_MEAN[l])/PSI_TRUE
            gi = gc + np.random.normal(0, np.sqrt(SG2_GAMMA_TRUE)*0.3)
        else:
            gi = np.random.normal(MU_GAMMA_TRUE[l], np.sqrt(SG2_GAMMA_TRUE)*0.3)
        ei = np.random.normal(0, 0.5)
        Kx = np.zeros((M_LI,M_LI))
        for a in range(M_LI):
            for b in range(M_LI):
                d2=(TAU[a]-TAU[b])**2
                Kx[a,b]=W2_XI*np.exp(-d2/(2*PHI_XI**2))
                if a==b: Kx[a,b]+=NU2_XI
        Kx+=1e-10*np.eye(M_LI)
        Lx=linalg.cholesky(Kx, lower=True)
        xi=Lx@np.random.normal(0,1,M_LI)
        zi=ei+gi*TAU+xi
        zl.append(zi); gl.append(gi); el.append(ei)
    Z_ALL.append(np.array(zl)); GAM_TRUE.append(np.array(gl)); ETA_TRUE.append(np.array(el))

# --- GP precompute ---
K_XI = np.zeros((M_LI,M_LI))
for a in range(M_LI):
    for b in range(M_LI):
        d2=(TAU[a]-TAU[b])**2
        K_XI[a,b]=W2_XI*np.exp(-d2/(2*PHI_XI**2))
        if a==b: K_XI[a,b]+=NU2_XI
K_XI+=1e-8*np.eye(M_LI)
K_CH = linalg.cholesky(K_XI, lower=True)
K_INV = linalg.cho_solve((K_CH,True), np.eye(M_LI))
TAU_IT = TAU@K_INV@TAU
TAU_IV = TAU@K_INV
ONES_II = K_INV.sum()
TAU_IO = TAU_IV.sum()

print("="*60)
print("DDP-Joint MCMC — Final Version")
print(f"Iter={N_ITER} Burn={N_BURN} K={K_TRUNC}")
print(f"True psi={PSI_TRUE}")
print(f"Stage means: {STAGE_MEAN}")
print("="*60)

# ============================================================
# MCMC with informative psi prior: psi ~ N(-1.5, 1.0)
# This encodes the physical expectation of negative coupling.
# ============================================================
PSI_PRIOR_MEAN = -1.5
PSI_PRIOR_SD = 1.0

def run_chain(seed):
    rng = np.random.default_rng(seed)

    # Init
    all_w = np.concatenate([W_ALL[l][D_ALL[l]==1] for l in range(N_STAGES)])
    mu_s = np.percentile(all_w, np.linspace(5,95,K_TRUNC))
    s2_s = np.ones(K_TRUNC)*0.1

    beta_l = [np.ones(K_TRUNC-1)*0.5 for _ in range(N_STAGES)]
    pi_l = []
    for l in range(N_STAGES):
        b=beta_l[l]; p=np.zeros(K_TRUNC)
        p[0]=b[0]
        for k in range(1,K_TRUNC-1): p[k]=b[k]*np.prod(1-b[:k])
        p[K_TRUNC-1]=np.prod(1-b); p=p/p.sum()
        pi_l.append(p)

    c_li = []
    for l in range(N_STAGES):
        cl=[]
        for i in range(N_UNITS[l]):
            if D_ALL[l][i]==1: cl.append(np.argmin(np.abs(W_ALL[l][i]-mu_s)))
            else: cl.append(0)
        c_li.append(np.array(cl,dtype=int))

    gam_l=[]; eta_l=[]
    for l in range(N_STAGES):
        gl=[]; el=[]
        for i in range(N_UNITS[l]):
            z=Z_ALL[l][i]
            X=np.column_stack([np.ones(M_LI),TAU])
            coef=np.linalg.lstsq(X,z,rcond=None)[0]
            el.append(coef[0]); gl.append(max(0.1,coef[1]))
        gam_l.append(np.array(gl)); eta_l.append(np.array(el))

    state = {
        "mu": mu_s.copy(), "s2": s2_s.copy(),
        "beta": beta_l, "pi": pi_l, "c": c_li,
        "gam": gam_l, "eta": eta_l,
        "rho": rng.uniform(0.3,0.8,2),
        "alpha": rng.uniform(0.5,3.0,3),
        "psi": -0.5,
        "mu0": np.mean(all_w), "kappa0": 1.0, "b0": 1.0,
        "mu_eta": 0.0, "sigma2_eta": 0.5,
        "mu_gam": rng.uniform(0.4,1.2,3),
        "sigma2_gam": np.ones(3)*0.5,
    }

    trace = {"psi":[],"rho2":[],"rho3":[],"alpha1":[],"alpha2":[],"alpha3":[],
             "mu_s":[],"s2_s":[],"pi":[],"c":[],
             "mu_gam":[],"sigma2_gam":[],"n_occ":[]}

    t0 = time.time()
    for m in range(N_ITER):
        s=state
        mu=s["mu"]; s2=s["s2"]; beta=s["beta"]; pi=s["pi"]; c=s["c"]
        gam=s["gam"]; eta=s["eta"]; rho=s["rho"]; alpha=s["alpha"]
        psi=s["psi"]; mu0=s["mu0"]; kappa0=s["kappa0"]; b0=s["b0"]
        mu_eta=s["mu_eta"]; sigma2_eta=s["sigma2_eta"]
        mu_gam=s["mu_gam"]; sigma2_gam=s["sigma2_gam"]

        # ---- Step 1: Slice sampling for clusters ----
        for l in range(N_STAGES):
            for i in range(N_UNITS[l]):
                u = rng.uniform(0, pi[l][c[l][i]])
                adm = np.where(pi[l] > u)[0]
                ll = np.zeros(len(adm))
                for idx, k in enumerate(adm):
                    mk = mu[k] + psi*gam[l][i]
                    if D_ALL[l][i]==1:
                        ll[idx] = -0.5*np.log(2*np.pi*s2[k]) - 0.5*(W_ALL[l][i]-mk)**2/s2[k]
                    else:
                        ll[idx] = stats.norm.logcdf(np.clip((mk-C_ALL[l][i])/np.sqrt(s2[k]),-30,30))
                ll = np.clip(ll, -60, ll.max())
                prob = np.exp(ll - logsumexp(ll)); prob = prob/prob.sum()
                c[l][i] = adm[rng.choice(len(adm), p=prob)]

        # ---- Step 2: Stick-breaking weights ----
        n_lk = np.zeros((N_STAGES, K_TRUNC))
        for l in range(N_STAGES):
            for k in range(K_TRUNC): n_lk[l,k] = np.sum(c[l]==k)

        for l in range(N_STAGES):
            for k in range(K_TRUNC-1):
                sg = n_lk[l,k+1:].sum()
                ap = 1+n_lk[l,k]; bp = alpha[l]+sg
                if l==0:
                    beta[l][k] = rng.beta(ap, bp)
                else:
                    bp_prop = rng.beta(ap, bp); bc = beta[l][k]; bpr = beta[l-1][k]; rl = rho[l-1]
                    def lprior(bv, bpv, rr, al):
                        lo=rr*bpv; hi=rr*bpv+1-rr
                        if bv<=lo or bv>=hi: return -np.inf
                        if rr>=1.0: return 0.0
                        arg = np.clip((1-bv-rr*(1-bpv))/(1-rr), 1e-15, 1-1e-15)
                        return np.log(al)-np.log(1-rr)+(al-1)*np.log(arg)
                    lp_p = lprior(bp_prop,bpr,rl,alpha[l])
                    lp_c = lprior(bc,bpr,rl,alpha[l])
                    lq_p = stats.beta.logpdf(bp_prop,ap,bp)
                    lq_c = stats.beta.logpdf(bc,ap,bp)
                    if np.log(rng.random()) < (lp_p-lp_c)+(lq_c-lq_p):
                        beta[l][k] = bp_prop
            b=beta[l]; p=np.zeros(K_TRUNC)
            p[0]=b[0]
            for k in range(1,K_TRUNC-1): p[k]=b[k]*np.prod(np.clip(1-b[:k],1e-15,1))
            p[K_TRUNC-1]=max(0,1-p[:K_TRUNC-1].sum()); p=p/p.sum()
            pi[l]=p

        # ---- Step 3: Atom parameters ----
        a0=2.0
        for k in range(K_TRUNC):
            wk=[]; gk=[]
            for l in range(N_STAGES):
                for i in range(N_UNITS[l]):
                    if c[l][i]==k and D_ALL[l][i]==1: wk.append(W_ALL[l][i]-psi*gam[l][i])
            wk=np.array(wk); nk=len(wk)
            if nk>0:
                wb=wk.mean(); kp=kappa0+nk; mp=(kappa0*mu0+nk*wb)/kp
                ss=np.sum((wk-wb)**2) if nk>1 else 0.0
                ap=a0+nk/2.0; bp=b0+0.5*ss+0.5*(kappa0*nk/kp)*(wb-mu0)**2
                s2[k]=np.clip(1.0/rng.gamma(ap,1.0/max(bp,1e-10)),1e-6,10.0)
                mu[k]=rng.normal(mp,np.sqrt(s2[k]/kp))
            else:
                s2[k]=np.clip(1.0/rng.gamma(a0,1.0/b0),1e-6,10.0)
                mu[k]=rng.normal(mu0,np.sqrt(s2[k]/kappa0))

        # ---- Step 4: Random effects ----
        for l in range(N_STAGES):
            for i in range(N_UNITS[l]):
                z=Z_ALL[l][i]; k=c[l][i]
                # gamma
                pd=TAU_IT; md=TAU_IV@z - eta[l][i]*TAU_IO
                if D_ALL[l][i]==1:
                    pl=psi**2/s2[k]; ml=psi*(W_ALL[l][i]-mu[k])/s2[k]
                else:
                    zs=np.clip((C_ALL[l][i]-mu[k]-psi*gam[l][i])/np.sqrt(s2[k]),-30,30)
                    imr=stats.norm.pdf(zs)/max(stats.norm.cdf(zs),1e-15)
                    wa=C_ALL[l][i]+np.sqrt(s2[k])*max(-3.0,min(3.0,imr))
                    pl=psi**2/s2[k]; ml=psi*(wa-mu[k])/s2[k]
                pp=1.0/sigma2_gam[l]
                post_p=pd+pl+pp; post_v=1.0/max(post_p,1e-10)
                post_m=post_v*(md+ml+pp*mu_gam[l])
                gam[l][i]=np.clip(rng.normal(post_m,np.sqrt(post_v)),0.05,3.0)
                # eta
                za=z-gam[l][i]*TAU; mde=(K_INV@za).sum()
                ppe=1.0/sigma2_eta
                post_pe=ONES_II+ppe; post_ve=1.0/max(post_pe,1e-10)
                post_me=post_ve*(mde+ppe*mu_eta)
                eta[l][i]=rng.normal(post_me,np.sqrt(post_ve))

        # ---- Step 5: Stage-level and hierarchical params ----
        for l in range(N_STAGES):
            gl=gam[l]; nl=N_UNITS[l]
            pv=1.0/(nl/sigma2_gam[l]+1.0/100.0); pm=pv*(gl.sum()/sigma2_gam[l])
            mu_gam[l]=rng.normal(pm,np.sqrt(pv))
            ss=np.sum((gl-mu_gam[l])**2)
            sigma2_gam[l]=np.clip(1.0/rng.gamma(2.0+nl/2.0,1.0/max(1.0+ss/2.0,1e-10)),0.01,5.0)

        ae=np.concatenate([eta[l] for l in range(N_STAGES)]); ne=len(ae)
        pve=1.0/(ne/sigma2_eta+1.0/100.0); pme=pve*(ae.sum()/sigma2_eta)
        mu_eta=rng.normal(pme,np.sqrt(pve))
        sse=np.sum((ae-mu_eta)**2)
        sigma2_eta=np.clip(1.0/rng.gamma(2.0+ne/2.0,1.0/max(1.0+sse/2.0,1e-10)),0.01,10.0)

        # ---- Step 6: psi (MH with informative prior) ----
        psi_prop = psi + rng.normal(0, 0.25)
        def ll_psi(pv):
            ll=0.0
            for ll_l in range(N_STAGES):
                for ll_i in range(N_UNITS[ll_l]):
                    kk=c[ll_l][ll_i]; mk=mu[kk]+pv*gam[ll_l][ll_i]
                    if D_ALL[ll_l][ll_i]==1:
                        ll+=-0.5*np.log(2*np.pi*s2[kk])-0.5*(W_ALL[ll_l][ll_i]-mk)**2/s2[kk]
                    else:
                        ll+=stats.norm.logcdf(np.clip((mk-C_ALL[ll_l][ll_i])/np.sqrt(s2[kk]),-30,30))
            return ll
        llp=ll_psi(psi_prop); llc=ll_psi(psi)
        lpp=-0.5*(psi_prop-PSI_PRIOR_MEAN)**2/PSI_PRIOR_SD**2
        lpc=-0.5*(psi-PSI_PRIOR_MEAN)**2/PSI_PRIOR_SD**2
        if np.log(rng.random())<(llp-llc)+(lpp-lpc): psi=psi_prop

        # ---- Step 7: rho (MH on logit) ----
        for l_idx in range(2):
            l=l_idx+1; rc=rho[l_idx]
            if rng.random()<0.3:
                rp=rng.beta(1.5,1.5)
            else:
                lc=np.log(rc/(1-rc+1e-15)); lp=lc+rng.normal(0,1.0)
                rp=1.0/(1.0+np.exp(-lp))
            rp=np.clip(rp,0.02,0.98)
            def lprior_rho(bc,bpr,rr,al):
                ll=0.0
                for kk in range(K_TRUNC-1):
                    bck=bc[kk]; bpk=bpr[kk]; lo=rr*bpk; hi=rr*bpk+1-rr
                    if bck<=lo or bck>=hi: return -np.inf
                    if rr>=1.0: continue
                    arg=np.clip((1-bck-rr*(1-bpk))/(1-rr),1e-15,1-1e-15)
                    ll+=np.log(al)-np.log(1-rr)+(al-1)*np.log(arg)
                return ll
            lpp_r=lprior_rho(beta[l],beta[l-1],rp,alpha[l])
            lpc_r=lprior_rho(beta[l],beta[l-1],rc,alpha[l])
            lj_p=np.log(rp)+np.log(1-rp); lj_c=np.log(rc)+np.log(1-rc+1e-15)
            la=(lpp_r-lpc_r)+(lj_p-lj_c)
            if np.isfinite(la) and np.log(rng.random())<la: rho[l_idx]=rp

        # ---- Step 8: alpha (Escobar-West) ----
        for l in range(N_STAGES):
            ml=len(np.unique(c[l])); nl=N_UNITS[l]
            ea=rng.beta(alpha[l]+1,nl)
            pa=(2.0+ml-1)/(2.0+ml-1+nl*(2.0-np.log(ea+1e-15)))
            if rng.random()<pa:
                alpha[l]=rng.gamma(2.0+ml,1.0/max(2.0-np.log(ea+1e-15),1e-10))
            else:
                alpha[l]=rng.gamma(2.0+ml-1,1.0/max(2.0-np.log(ea+1e-15),1e-10))
            alpha[l]=np.clip(alpha[l],0.1,20.0)

        # ---- Step 9: Base measure ----
        pv0=1.0/(K_TRUNC/np.mean(s2)+1.0/100.0)
        pm0=pv0*(mu.sum()/np.mean(s2)); mu0=rng.normal(pm0,np.sqrt(pv0))
        if rng.random()<0.5:
            kp=np.clip(kappa0*np.exp(rng.normal(0,0.3)),0.1,50.0); kappa0=kp
        if rng.random()<0.5:
            bp_v=2.0*K_TRUNC/2.0+np.sum((mu-mu0)**2/s2)/2.0
            b0=np.clip(rng.gamma(bp_v,1.0/1.0),0.1,20.0)

        # Update state
        state = {"mu":mu,"s2":s2,"beta":beta,"pi":pi,"c":c,
                 "gam":gam,"eta":eta,"rho":rho,"alpha":alpha,"psi":psi,
                 "mu0":mu0,"kappa0":kappa0,"b0":b0,
                 "mu_eta":mu_eta,"sigma2_eta":sigma2_eta,
                 "mu_gam":mu_gam,"sigma2_gam":sigma2_gam}

        if m>=N_BURN:
            trace["psi"].append(psi); trace["rho2"].append(rho[0]); trace["rho3"].append(rho[1])
            trace["alpha1"].append(alpha[0]); trace["alpha2"].append(alpha[1]); trace["alpha3"].append(alpha[2])
            trace["mu_s"].append(mu.copy()); trace["s2_s"].append(s2.copy())
            trace["pi"].append([p.copy() for p in pi]); trace["c"].append([cl.copy() for cl in c])
            trace["mu_gam"].append(mu_gam.copy()); trace["sigma2_gam"].append(sigma2_gam.copy())
            trace["n_occ"].append([len(np.unique(c[l])) for l in range(N_STAGES)])

        if (m+1)%5000==0:
            t=time.time()-t0; no=[len(np.unique(c[l])) for l in range(N_STAGES)]
            print(f"  Iter {m+1}/{N_ITER} | {t:.0f}s | psi={psi:.3f} rho2={rho[0]:.3f} rho3={rho[1]:.3f} | K*={no}")

    return trace

# ============================================================
# Run single long chain
# ============================================================
print("\nRunning MCMC (single long chain)...")
trace = run_chain(RNG_SEED)

psi_arr = np.array(trace["psi"])
rho2_arr = np.array(trace["rho2"])
rho3_arr = np.array(trace["rho3"])

psi_m = psi_arr.mean(); psi_s = psi_arr.std()
psi_hpd_l = np.percentile(psi_arr,2.5); psi_hpd_u = np.percentile(psi_arr,97.5)
psi_pn = np.mean(psi_arr<0)

rho2_m = rho2_arr.mean(); rho3_m = rho3_arr.mean()
rho23_p = np.mean(rho2_arr<rho3_arr)

print("\n"+"="*60)
print("POSTERIOR SUMMARIES")
print("="*60)
print(f"\n  psi: mean={psi_m:.3f}, SD={psi_s:.3f}, 95%HPD=[{psi_hpd_l:.3f},{psi_hpd_u:.3f}]")
print(f"       Pr(psi<0)={psi_pn:.4f} (prior: N({PSI_PRIOR_MEAN},{PSI_PRIOR_SD}))")
print(f"\n  rho2: mean={rho2_m:.3f}, 95%HPD=[{np.percentile(rho2_arr,2.5):.3f},{np.percentile(rho2_arr,97.5):.3f}]")
print(f"  rho3: mean={rho3_m:.3f}, 95%HPD=[{np.percentile(rho3_arr,2.5):.3f},{np.percentile(rho3_arr,97.5):.3f}]")
print(f"  Pr(rho2<rho3)={rho23_p:.4f}")

mu_gam_arr = np.array(trace["mu_gam"])
print(f"\n  mu_gamma (degradation rate means):")
for l in range(N_STAGES):
    print(f"    Stage {l+1}: {mu_gam_arr[:,l].mean():.3f} (true: {MU_GAMMA_TRUE[l]:.1f})")

occ_arr = np.array(trace["n_occ"])
print(f"\n  Occupied clusters:")
for l in range(N_STAGES):
    print(f"    Stage {l+1}: {occ_arr[:,l].mean():.2f}")

# Reliability
N_POST = len(psi_arr)
print(f"\n  Reliability R_l(t) (posterior mean +/- SD):")
for tm in [300, 450, 500]:
    lt = np.log(tm)
    for l in range(N_STAGES):
        rv = np.zeros(N_POST)
        for s in range(N_POST):
            ps = trace["pi"][s][l]; ms = trace["mu_s"][s]; ss = trace["s2_s"][s]
            pv = trace["psi"][s]; mg = trace["mu_gam"][s][l]; sg2 = trace["sigma2_gam"][s][l]
            rel = 0.0
            for k in range(K_TRUNC):
                mk = ms[k]+pv*mg; vk = ss[k]+pv**2*sg2
                zk = np.clip((mk-lt)/np.sqrt(max(vk,1e-10)),-30,30)
                rel += ps[k]*stats.norm.cdf(zk)
            rv[s] = max(0,min(1,rel))
        print(f"    Stage {l+1}, t={tm}h: {rv.mean():.4f} +/- {rv.std():.4f}")

# Cluster weight evolution (posterior mean)
print(f"\n  Cluster weight evolution (top components by posterior mean):")
for l in range(N_STAGES):
    pi_mean = np.zeros(K_TRUNC)
    for s in range(N_POST):
        pi_mean += np.array(trace["pi"][s][l])
    pi_mean /= N_POST
    mu_mean = np.array([np.mean([trace["mu_s"][s][k] for s in range(N_POST)]) for k in range(K_TRUNC)])
    for k in np.argsort(pi_mean)[::-1]:
        if pi_mean[k] > 0.02:
            print(f"    Stage {l+1}, k={k}: pi={pi_mean[k]:.4f}, mu*={mu_mean[k]:.3f}")

# Save
result = {
    "config": {"n_iter": N_ITER, "n_burn": N_BURN, "k_trunc": K_TRUNC},
    "psi": {"mean": float(psi_m), "sd": float(psi_s),
            "hpd_low": float(psi_hpd_l), "hpd_high": float(psi_hpd_u),
            "prob_neg": float(psi_pn)},
    "rho2": {"mean": float(rho2_m),
             "hpd_low": float(np.percentile(rho2_arr,2.5)),
             "hpd_high": float(np.percentile(rho2_arr,97.5))},
    "rho3": {"mean": float(rho3_m),
             "hpd_low": float(np.percentile(rho3_arr,2.5)),
             "hpd_high": float(np.percentile(rho3_arr,97.5))},
    "prob_rho2_lt_rho3": float(rho23_p),
    "mu_gamma": [float(mu_gam_arr[:,l].mean()) for l in range(N_STAGES)],
    "occupied": [float(occ_arr[:,l].mean()) for l in range(N_STAGES)],
}

sp = os.path.join(os.path.dirname(__file__), "case_study_results.json")
with open(sp, "w") as f: json.dump(result, f, indent=2)
print(f"\nSaved: {sp}")
print("="*60)
