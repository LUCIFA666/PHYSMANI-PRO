"""Train a small NGFF on official position tracks and evaluate held-out scenes.

Only validation scenes select the checkpoint. Test trajectories are evaluated
once, after training finishes. This is not the full paper benchmark.
"""

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from scripts.smoke_dynamics import make_model, world_points


def inputs(points):
    """The official release-from-rest convention; all positions are world-space."""
    first = points[:, 0]
    centers = first.mean(dim=-2)
    zeros = torch.zeros_like(centers)
    batch, objects, count, _ = first.shape
    padding = torch.ones(batch, objects, count, device=points.device)
    knn = torch.zeros(batch, objects, count, 8, dtype=torch.long, device=points.device)
    return first, centers, zeros, zeros.clone(), zeros.clone(), padding, knn


def new_model(config, device):
    model = make_model(device, keypoints=config["keypoints_per_object"],
                       hidden_dim=config["hidden_dim"], num_layers=config["num_layers"])
    model.dt = config["sampled_dt"]
    model.step_size = config["ode_step_size"]
    return model


@torch.no_grad()
def predict(model, tracks):
    model.eval()
    prediction = world_points(model(*inputs(tracks), pred_len=tracks.shape[1]-1))
    if not torch.isfinite(prediction).all().item():
        raise RuntimeError("non-finite rollout")
    torch.testing.assert_close(prediction[:, 0], tracks[:, 0], atol=2e-6, rtol=1e-6)
    return prediction


def metric(prediction, truth):
    """Macro-over-scenes metrics, excluding the observed initial frame."""
    p = prediction.detach().cpu().double()[:, 1:]
    g = truth.detach().cpu().double()[:, 1:]
    error = p-g
    p_mean, g_mean = p.mean(dim=1, keepdim=True), g.mean(dim=1, keepdim=True)
    a, b = p-p_mean, g-g_mean
    variance_p, variance_g = a.square().sum(dim=1), b.square().sum(dim=1)
    denominator = (variance_p*variance_g).sqrt()
    correlation = (a*b).sum(dim=1) / denominator.clamp_min(1e-12)
    # Constant GT coordinates carry no temporal-correlation information. A
    # constant prediction against moving GT gets correlation zero by convention.
    valid_gt = variance_g > 1e-10
    correlation = torch.where(variance_p > 1e-10, correlation, torch.zeros_like(correlation))
    rows = []
    for s in range(len(p)):
        center_error = error[s].mean(dim=-2)
        rows.append({
            "rmse": error[s].square().mean().sqrt().item(),
            "fpe": error[s,-1].norm(dim=-1).mean().item(),
            "r": correlation[s][valid_gt[s]].mean().item() if valid_gt[s].any() else None,
            "centroid_rmse": center_error.square().mean().sqrt().item(),
            "centroid_final_error": center_error[-1].norm(dim=-1).mean().item(),
            "r_valid_coordinates": int(valid_gt[s].sum().item()),
        })
    result = {key: float(np.mean([row[key] for row in rows if row[key] is not None]))
              for key in ("rmse", "fpe", "r", "centroid_rmse", "centroid_final_error")}
    result["per_scene"] = rows
    result["scenes"] = len(rows)
    return result


def validate_metrics():
    t = torch.arange(6, dtype=torch.float32).view(1, 6, 1, 1, 1)
    gt = t * torch.tensor([1.,2.,3.]).view(1,1,1,1,3)
    same = metric(gt, gt)
    assert same["rmse"] == 0 and same["fpe"] == 0 and abs(same["r"]-1) < 1e-10
    shifted = gt + torch.tensor([1.,0.,0.])
    score = metric(shifted,gt)
    assert abs(score["rmse"]-1/math.sqrt(3)) < 1e-10 and score["fpe"] == 1
    constant = metric(gt[:, :1].expand_as(gt),gt)
    assert constant["r"] == 0
    return {"exact_prediction": "passed", "known_translation_error": "passed",
            "constant_prediction_correlation": "passed"}


def baselines(tracks, dt, gravity=-5., floor=-0.94):
    start = tracks[:, :1]
    persistence = start.expand_as(tracks).clone()
    t = torch.arange(tracks.shape[1], device=tracks.device)*dt
    displacement = torch.zeros(tracks.shape[0], tracks.shape[1], tracks.shape[2], 1, 3, device=tracks.device)
    displacement[..., 2] = 0.5 * gravity * t.square().view(1,-1,1,1)
    ballistic = persistence + displacement
    # Stop each rigidly translated object when its initial bottom reaches the
    # known floor. No future observations or fitted test parameters are used.
    bottom = start[..., 2].amin(dim=-1, keepdim=True)
    minimum_shift = floor-bottom
    stopped = displacement.clone()
    stopped[..., 2] = torch.maximum(stopped[..., 2], minimum_shift)
    return {"constant_velocity_zero_release": persistence,
            "constant_acceleration_gravity": ballistic,
            "gravity_with_floor_stop": persistence+stopped}


def evaluate(predictions, truth, dt):
    result = {}
    for name, predicted in predictions.items():
        result[name] = {}
        for horizon in (0.2,0.6,1.2):
            frames = round(horizon/dt)+1
            result[name][str(horizon)] = metric(predicted[:,:frames],truth[:,:frames])
    return result


def load_split(entries, split, device):
    selected = [entry for entry in entries if entry["split"] == split]
    arrays = []
    for entry in selected:
        path = Path(entry["cache"])
        if not path.is_absolute():
            path = ROOT/path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["cache_sha256"]
        with np.load(path) as cached:
            points = cached["points"].copy()
            assert np.isfinite(points).all()
            assert points.shape == (31,3,256,3)
            arrays.append(points)
    return torch.tensor(np.stack(arrays), device=device), selected


def plot_results(output, history, test, predictions, entries, dt):
    figure, axes = plt.subplots(1,2,figsize=(12,4.2),layout="constrained")
    axes[0].plot([x["step"] for x in history], [x["train_mse"] for x in history],label="Training")
    val = [x for x in history if x.get("validation_mse") is not None]
    axes[0].plot([x["step"] for x in val],[x["validation_mse"] for x in val],label="Validation")
    axes[0].set(xlabel="Optimizer updates",ylabel="Point MSE",yscale="log",title="Learning on official tracks")
    axes[0].legend()
    time_grid = np.arange(test.shape[1])*dt
    styles = {"ngff_initial": ("#777777","--"), "ngff_trained": ("#1768ac","-"),
              "constant_velocity_zero_release": ("#e08018",":"),
              "constant_acceleration_gravity": ("#ac3434",":"),
              "gravity_with_floor_stop": ("#22854b","-.")}
    for name, predicted in predictions.items():
        error = (predicted-test).square().mean(dim=(0,2,3,4)).sqrt().cpu().numpy()
        color, style = styles[name]
        axes[1].plot(time_grid,error,label=name,color=color,linestyle=style)
    axes[1].set(xlabel="Forecast horizon (s)",ylabel="Point RMSE (simulation units)",title="Independent test scenes")
    axes[1].legend(fontsize=7)
    figure.savefig(output/"learning_and_test_error.png",dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(len(entries),3,figsize=(13,3.2*len(entries)),squeeze=False,layout="constrained")
    gt_centers = test.mean(dim=-2).cpu().numpy()
    for s,entry in enumerate(entries):
        for obj in range(3):
            ax = axes[s,obj]
            ax.plot(time_grid,gt_centers[s,:,obj,2],color="black",linewidth=2,label="Ground truth")
            for name in ("ngff_initial","ngff_trained","gravity_with_floor_stop"):
                centers=predictions[name].mean(dim=-2).cpu().numpy()
                color,style=styles[name]
                ax.plot(time_grid,centers[s,:,obj,2],color=color,linestyle=style,label=name)
            ax.set(title=entry["name"]+" / "+entry["objects"][obj],xlabel="Time (s)",ylabel="Centroid z (simulation units)")
            if s==0 and obj==0:ax.legend(fontsize=7)
    figure.savefig(output/"heldout_centroid_trajectories.png",dpi=160)
    plt.close(figure)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates",type=int,default=400)
    parser.add_argument("--output",type=Path,default=ROOT/"artifacts/quality_smoke")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    config={"purpose":"small learning-quality smoke; not full-paper reproduction",
            "seed":0,"hidden_dim":128,"num_layers":4,"keypoints_per_object":256,
            "objects":3,"frames":31,"sampled_dt":0.04,"ode_step_size":0.02,
            "training_scenes":8,"validation_scenes":2,"test_scenes":2,
            "actual_batch_size":8,"updates":args.updates,"lr":1e-4,"min_lr":1e-6,
            "warmup_updates":10,"gradient_clip_norm":1.,"validation_interval":20,
            "checkpoint_selection":"lowest validation point MSE over full 1.2s rollout",
            "effect_criterion":"test RMSE at least 10% lower than strongest simple baseline and FPE no worse",
            "metric_definitions":{
                "rmse":"per-scene sqrt(mean coordinate squared error)); macro mean; initial frame excluded",
                "fpe":"mean per-point final 3D Euclidean error; macro mean over scenes",
                "r":"per-point per-coordinate Pearson over forecast time; omit constant GT coordinates; constant predictions score zero",
                "units":"original simulation coordinates, not calibrated real-world meters"}}
    (args.output/"config.json").write_text(json.dumps(config,indent=2)+"\n")
    manifest_path=ROOT/"data/quality_smoke/manifest.json"
    manifest=json.loads(manifest_path.read_text())
    assert len(manifest["scenes"])==12,"prepare all 12 scenes before training"
    paths=[entry["source"] for entry in manifest["scenes"]]
    assert len(set(paths))==12,"scene leakage across splits"
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.set_num_threads(4)
    device=torch.device("cuda:0")
    report={"status":"running","started_utc":datetime.now(timezone.utc).isoformat(),
            "config":config,"data_revision":manifest["revision"],
            "data_manifest_sha256":hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "metric_self_checks":validate_metrics(),
            "environment":{"torch":torch.__version__,"cuda":torch.version.cuda,
                           "gpu":torch.cuda.get_device_name(0)},"history":[]}
    train,train_entries=load_split(manifest["scenes"],"train",device)
    validation,val_entries=load_split(manifest["scenes"],"validation",device)
    model=new_model(config,device)
    initial={name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
    torch.save({"state_dict":initial,"config":config},args.output/"initial.pt")
    initial_train=metric(predict(model,train),train)
    initial_val=metric(predict(model,validation),validation)
    report["initial_train"]=initial_train
    report["initial_validation"]=initial_val
    best_loss=initial_val["rmse"]**2
    best_step=0
    torch.save({"state_dict":initial,"config":config,"step":0},args.output/"best.pt")
    history=[{"step":0,"train_mse":initial_train["rmse"]**2,"validation_mse":best_loss}]
    optimizer=torch.optim.Adam(model.parameters(),lr=config["lr"])
    def lr_scale(step):
        if step<config["warmup_updates"]:return (step+1)/config["warmup_updates"]
        fraction=(step-config["warmup_updates"])/max(1,args.updates-config["warmup_updates"])
        return config["min_lr"]/config["lr"]+(1-config["min_lr"]/config["lr"])*0.5*(1+math.cos(math.pi*fraction))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_scale)
    initial_train_mse=F.mse_loss(predict(model,train),train).item()
    history[0]["train_mse"]=initial_train_mse
    best_loss=F.mse_loss(predict(model,validation),validation).item()
    history[0]["validation_mse"]=best_loss
    start=time.perf_counter()
    for step in range(1,args.updates+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        predicted=world_points(model(*inputs(train),pred_len=30))
        loss=F.mse_loss(predicted,train)
        if not torch.isfinite(loss).item():raise RuntimeError(f"non-finite loss at update {step}")
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config["gradient_clip_norm"],error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if step%10==0 or step==1:
            row={"step":step,"train_mse":loss.item(),"lr":scheduler.get_last_lr()[0],
                 "gradient_norm_before_clip":norm.item(),"elapsed_s":time.perf_counter()-start}
            if step%config["validation_interval"]==0 or step==args.updates:
                validation_loss=F.mse_loss(predict(model,validation),validation).item()
                row["validation_mse"]=validation_loss
                if validation_loss<best_loss:
                    best_loss=validation_loss;best_step=step
                    torch.save({"state_dict":model.state_dict(),"config":config,"step":step},args.output/"best.pt")
            history.append(row)
            print("TRAIN",json.dumps(row),flush=True)
            report["history"]=history
            (args.output/"progress.json").write_text(json.dumps(report,indent=2)+"\n")
    report["training_seconds"]=time.perf_counter()-start
    report["best_validation_step"]=best_step
    saved=torch.load(args.output/"best.pt",map_location=device,weights_only=True)
    model.load_state_dict(saved["state_dict"],strict=True)
    report["selected_train"]=metric(predict(model,train),train)
    report["selected_validation"]=metric(predict(model,validation),validation)
    # First and only evaluation pass on the independent test scenes.
    test,test_entries=load_split(manifest["scenes"],"test",device)
    initial_model=new_model(config,device)
    initial_model.load_state_dict(initial,strict=True)
    predictions=baselines(test,config["sampled_dt"])
    predictions["ngff_initial"]=predict(initial_model,test)
    predictions["ngff_trained"]=predict(model,test)
    report["test"]=evaluate(predictions,test,config["sampled_dt"])
    report["test_scenes"]=[entry["name"] for entry in test_entries]
    trained=report["test"]["ngff_trained"]["1.2"]
    baseline_names=[name for name in predictions if name not in ("ngff_initial","ngff_trained")]
    strongest=min(baseline_names,key=lambda name:report["test"][name]["1.2"]["rmse"])
    baseline=report["test"][strongest]["1.2"]
    report["learning_chain_passed"]=(report["selected_train"]["rmse"]<initial_train["rmse"]*0.8 and
                                      report["selected_validation"]["rmse"]<initial_val["rmse"]*0.8)
    report["effect_check"]={"strongest_baseline":strongest,
            "rmse_improvement_fraction":1-trained["rmse"]/baseline["rmse"],
            "supported_on_this_tiny_split":trained["rmse"]<baseline["rmse"]*0.9 and trained["fpe"]<=baseline["fpe"]}
    report["status"]="completed"
    report["finished_utc"]=datetime.now(timezone.utc).isoformat()
    (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    with (args.output/"test_metrics.csv").open("w") as file:
        writer=csv.DictWriter(file,fieldnames=["model","horizon","rmse","fpe","r","centroid_rmse","centroid_final_error","scenes"])
        writer.writeheader()
        for name, horizons in report["test"].items():
            for horizon, score in horizons.items():
                writer.writerow({"model":name,"horizon":horizon,**{k:score[k] for k in writer.fieldnames if k not in ("model","horizon")}})
    np.savez_compressed(args.output/"test_predictions.npz",truth=test.cpu().numpy(),
                        **{name:predicted.cpu().numpy() for name,predicted in predictions.items()})
    plot_results(args.output,history,test,predictions,test_entries,config["sampled_dt"])
    lines=["NGFF quality smoke: completed", "", "Official GSCollision revision: "+manifest["revision"],
           "Train/validation/test: 8/2/2 independent scenes; 256 points/object; 1.2s rollout.",
           "Best checkpoint chosen ONLY by validation; selected update: "+str(best_step),
           "Training wall time (seconds): "+str(report["training_seconds"]),
           "Learning chain passed: "+str(report["learning_chain_passed"]),
           "Effect versus strongest baseline: "+json.dumps(report["effect_check"]),
           "", "Model | horizon(s) | RMSE | FPE | temporal Pearson R"]
    for name,horizons in report["test"].items():
        for horizon,score in horizons.items():
            lines.append(f"{name} | {horizon} | {score['rmse']:.6f} | {score['fpe']:.6f} | {score['r']:.6f}")
    lines += ["", "Units are original simulation coordinates, not real-world meters.",
              "This tiny split does not establish paper-level accuracy or robot grasp success.",
              "Architecture width, point count, sequence subsampling, learning rate and update",
              "count differ from the full paper. See config.json and the cached data manifest.",
              "Fixed-gravity baselines use the known -5 simulation gravity and -0.94 floor.",
              "All methods forecast from the same initial release-from-rest observation."]
    (args.output/"SUMMARY.txt").write_text("\n".join(lines)+"\n")
    print("QUALITY RESULT",json.dumps(report["effect_check"]),flush=True)
    print("REPORT",args.output/"report.json",flush=True)


if __name__=="__main__":
    main()
