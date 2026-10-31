"""Create every report figure directly from the recorded run outputs."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.facecolor": "white", "figure.facecolor": "white"})
COLORS = {"plain": "#7B8894", "correlated": "#CB6443", "randomized": "#168278"}
NAMES = {"plain": "Plain", "correlated": "Correlated", "randomized": "Randomized"}


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    data = np.load(ROOT / "data/test_views.npz")
    labels = data["digit_labels"]
    selected = [int(np.flatnonzero(labels == digit)[0]) for digit in (0,1,2,5,8)]
    conflict = data["conflict"].reshape(len(labels),9,32,32)
    texture_ids = data["texture_ids"].reshape(len(labels),9)
    fig, axes = plt.subplots(3, 5, figsize=(7.1, 3.9))
    for j, idx in enumerate(selected):
        shape = int(labels[idx])
        tex = (shape+3)%10
        k = int(np.flatnonzero(texture_ids[idx]==tex)[0])
        imgs = [data["normal"][idx], data["aligned"][idx], conflict[idx,k]]
        titles = [f"shape {shape}", f"shape {shape} / texture {shape}", f"shape {shape} / texture {tex}"]
        for i, (img,title) in enumerate(zip(imgs,titles)):
            axes[i,j].imshow(img,cmap="gray",vmin=0,vmax=1,interpolation="nearest")
            axes[i,j].set_xticks([]); axes[i,j].set_yticks([])
            axes[i,j].set_title(title,fontsize=8,pad=4)
        if j == 0:
            for i, row in enumerate(("Normal", "Aligned", "Conflict")):
                axes[i,j].set_ylabel(row,fontsize=10,labelpad=12)
    fig.subplots_adjust(wspace=.18,hspace=.4,left=.1,right=.99,top=.93,bottom=.03)
    save(fig,"test_examples")

    fig,axes=plt.subplots(1,10,figsize=(7.1,1.15))
    disks=data["texture_only_disk"][:10]
    for t,ax in enumerate(axes):
        ax.imshow(disks[t],cmap="gray",vmin=0,vmax=1,interpolation="nearest")
        ax.set_title(f"{t}",fontsize=10,pad=3)
        ax.set_xticks([]);ax.set_yticks([])
        ax.set_xlabel(f"{(t%5)*36}deg\n{3 if t<5 else 6} cyc",fontsize=6.5,labelpad=3)
    fig.subplots_adjust(wspace=.12,left=.01,right=.99,top=.85,bottom=.25)
    save(fig,"texture_bank")

    summary=json.loads((ROOT/"results/summary.json").read_text())
    get=lambda c,v:next(r for r in summary if r["condition"]==c and r["view"]==v)
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.9),gridspec_kw={"width_ratios":[1.1,1]})
    conditions=["plain","correlated","randomized"]
    x=np.arange(3)
    for shift,view,label in [(-.2,"normal","Normal"),(.2,"aligned","Aligned")]:
        y=[100*get(c,view)["shape_accuracy_mean"] for c in conditions]
        sd=[100*get(c,view)["shape_accuracy_sd"] for c in conditions]
        axes[0].bar(x+shift,y,width=.36,label=label,color="#8E9BAC" if view=="normal" else "#445A72",
                    yerr=sd,capsize=2,error_kw={"linewidth":.8})
        for a,b in zip(x+shift,y):axes[0].text(a,b-6,f"{b:.1f}",ha="center",va="top",color="white",fontsize=8)
    axes[0].set_xticks(x,[NAMES[c] for c in conditions],fontsize=8)
    axes[0].set_ylim(0,105);axes[0].set_ylabel("Digit accuracy (%)")
    axes[0].set_title("Competence checks",fontsize=11)
    axes[0].legend(frameon=False,loc="upper center",bbox_to_anchor=(.5,-.16),ncol=2,fontsize=8)
    bottom=np.zeros(3)
    for key,label,color in [("shape_accuracy_mean","Shape","#168278"),
                            ("texture_agreement_mean","Texture","#CB6443"),
                            ("other_rate_mean","Other","#D7DCE2")]:
        values=np.array([100*get(c,"conflict")[key] for c in conditions])
        axes[1].bar(x,values,bottom=bottom,width=.58,label=label,color=color)
        if label=="Shape":
            for a,b in zip(x,values):axes[1].text(a,b/2,f"{b:.1f}",ha="center",va="center",color="white",fontsize=9)
        bottom+=values
    axes[1].set_xticks(x,[NAMES[c] for c in conditions],fontsize=8)
    axes[1].set_ylim(0,105);axes[1].set_ylabel("All conflict predictions (%)")
    axes[1].set_title("Cue-conflict decisions",fontsize=11)
    axes[1].legend(frameon=False,loc="upper center",bbox_to_anchor=(.5,-.16),ncol=3,fontsize=7)
    fig.tight_layout(w_pad=1.7)
    save(fig,"main_results")

    histories=json.loads((ROOT/"results/training_records.json").read_text())
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.6))
    for c in conditions:
        records=[r for r in histories if r["condition"]==c]
        epochs=[r["epoch"] for r in records[0]["history"]]
        for key,ax in [("train_loss",axes[0]),("normal_validation_accuracy",axes[1])]:
            values=np.array([[e[key] for e in r["history"]] for r in records])
            mean=values.mean(0);sd=values.std(0,ddof=1)
            if "accuracy" in key:mean*=100;sd*=100
            ax.plot(epochs,mean,color=COLORS[c],label=NAMES[c],linewidth=1.6)
            ax.fill_between(epochs,mean-sd,mean+sd,color=COLORS[c],alpha=.13)
    axes[0].set_ylabel("Training cross entropy");axes[0].set_xlabel("Epoch")
    axes[1].set_ylabel("Normal validation accuracy (%)");axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(20,100);axes[0].legend(frameon=False,fontsize=8)
    fig.tight_layout()
    save(fig,"training_curves")

    # Deterministic example of a decision change; explicitly illustrative rather
    # than a random sample. Main metrics always use every held-out original.
    cor=np.load(ROOT/"results/predictions_correlated_seed11_conflict.npz")
    rnd=np.load(ROOT/"results/predictions_randomized_seed11_conflict.npz")
    changed=np.flatnonzero((cor["prediction"]==cor["texture_label"]) &
                          (rnd["prediction"]==rnd["shape_label"]))
    selected_changes=changed[:4]
    fig,axes=plt.subplots(1,4,figsize=(6.7,2.15))
    for ax,k in zip(axes,selected_changes):
        ax.imshow(data["conflict"][k],cmap="gray",vmin=0,vmax=1,interpolation="nearest")
        s,t=int(cor["shape_label"][k]),int(cor["texture_label"][k])
        ax.set_title(f"Shape {s} / texture {t}",fontsize=9,pad=5)
        ax.set_xlabel(f"Correlated: {int(cor['prediction'][k])}\nRandomized: {int(rnd['prediction'][k])}",fontsize=9)
        ax.set_xticks([]);ax.set_yticks([])
    fig.tight_layout(w_pad=1)
    save(fig,"illustrative_decisions")
    print("created figures",*[p.name for p in sorted(OUT.glob('*.png'))])


if __name__=="__main__":
    main()
