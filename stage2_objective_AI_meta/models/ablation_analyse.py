#!/usr/bin/env python3
"""
Ablation figure (MEE) built directly from the per-species CSV.
Horizontal bars so condition labels never overlap; each panel uses ONE colour
scheme so nothing is ambiguous.
  (A) recall on hidden cells per condition, with bootstrap 95% CI and the chance
      line. All bars one colour; the only bar meeting chance is annotated.
  (B) fragmentation and spread KS per condition (range size omitted: it is 0 by
      construction under area-matched top-N binarisation).
"""
import argparse, csv, collections
from pathlib import Path
import numpy as np
from scipy.stats import ks_2samp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ORDER = ["FULL","NO_HISTORY","NO_NETWORK","NO_ENV","NO_SPECIES_FEATS"]
LAB = {"FULL":"Full model","NO_HISTORY":"No records","NO_NETWORK":"No interactions",
       "NO_ENV":"No environment","NO_SPECIES_FEATS":"No traits"}
C_BAR="#4477AA"; C_BASE="#555555"; C_FRAG="#CC6677"; C_SPR="#117733"; C_FAIL="#AA3377"
plt.rcParams.update({"font.size":9,"axes.labelsize":9,"xtick.labelsize":8,
    "ytick.labelsize":8.5,"legend.fontsize":8,"axes.linewidth":0.8,
    "pdf.fonttype":42,"ps.fonttype":42,"savefig.dpi":300})

def boot(v, n=4000, seed=0):
    v=np.asarray([x for x in v if x==x],float); rng=np.random.default_rng(seed)
    bs=np.sort(rng.choice(v,(n,v.size),True).mean(1))
    return v.mean(), bs[int(.025*n)], bs[int(.975*n)]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--ks-bar",type=float,default=0.30); a=ap.parse_args()
    rows=list(csv.DictReader(open(a.csv)))
    g=collections.defaultdict(list)
    for r in rows: g[r["variant"]].append(r)
    present=[v for v in ORDER if v in g]

    rec,lo,hi,chance,frag,spr={},{},{},{},{},{}
    for v in present:
        rr=g[v]
        rec[v],lo[v],hi[v]=boot([float(x["rec_novel"]) for x in rr if x["rec_novel"]!=""])
        chance[v]=np.mean([float(x["chance_novel"]) for x in rr if x["chance_novel"]!=""])
        frag[v]=ks_2samp([float(x["truth_patches"]) for x in rr],
                         [float(x["pred_patches"]) for x in rr]).statistic
        spr[v]=ks_2samp([float(x["truth_spread"]) for x in rr],
                        [float(x["pred_spread"]) for x in rr]).statistic

    y=np.arange(len(present))[::-1]           # Full at top
    labs=[LAB[v] for v in present]
    fig,ax=plt.subplots(1,2,figsize=(6.6,2.7),gridspec_kw={"width_ratios":[1,1]})

    # ---- (A) recall ----
    A=ax[0]
    m=[rec[v]*100 for v in present]
    el=[(rec[v]-lo[v])*100 for v in present]; eh=[(hi[v]-rec[v])*100 for v in present]
    bars=A.barh(y,m,xerr=[el,eh],color=C_BAR,height=.62,capsize=3,
                error_kw=dict(lw=1,ecolor="#222"))
    ch=np.mean([chance[v] for v in present])*100
    A.axvline(ch,ls=(0,(4,3)),lw=1.1,color=C_BASE,zorder=0)
    A.text(ch,y.max()+0.62,"chance",ha="center",va="bottom",fontsize=8,color=C_BASE)
    # annotate the failing arm directly (no colour code needed)
    for i,v in enumerate(present):
        if v=="NO_HISTORY":
            A.text(rec[v]*100+eh[i]+0.4,y[i],"falls to chance",va="center",
                   ha="left",fontsize=7.8,color=C_FAIL,fontweight="bold")
    A.set_yticks(y); A.set_yticklabels(labs)
    A.set_xlabel("recall on hidden cells  [%]"); A.set_xlim(0,15)

    # ---- (B) shape KS ----
    B=ax[1]; h=.32
    B.barh(y+h/2,[frag[v] for v in present],h,color=C_FRAG,label="fragmentation")
    B.barh(y-h/2,[spr[v]  for v in present],h,color=C_SPR, label="spread")
    B.axvline(a.ks_bar,ls=(0,(4,3)),lw=1.1,color=C_BASE,zorder=0)
    B.text(a.ks_bar,y.max()+0.62,"agreement\nthreshold",ha="center",va="bottom",
           fontsize=7.6,color=C_BASE,linespacing=.9)
    B.set_yticks(y); B.set_yticklabels([])
    B.set_xlabel("KS distance from truth"); B.set_xlim(0,0.68)
    B.legend(handles=[Patch(color=C_FRAG,label="fragmentation"),
                      Patch(color=C_SPR,label="spread")],
             frameon=False,loc="lower right")

    for axx,tag in zip(ax,"AB"):
        axx.spines[["top","right"]].set_visible(False)
        axx.grid(axis="x",lw=.4,alpha=.35)
        axx.set_ylim(y.min()-0.6,y.max()+1.0)
        axx.text(-0.02,1.10,f"({tag})",transform=axx.transAxes,fontweight="bold",
                 fontsize=11,va="top",ha="right")
    fig.tight_layout()
    o=Path(a.out); o.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(o.with_suffix(".pdf"),bbox_inches="tight",facecolor="white")
    fig.savefig(o.with_suffix(".png"),dpi=300,bbox_inches="tight",facecolor="white")
    print("recall:",{LAB[v]:round(rec[v]*100,1) for v in present})
    print("saved",o.with_suffix(".pdf"))
if __name__=="__main__": main()