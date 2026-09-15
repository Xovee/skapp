"""GPU-resident auxiliary training for full-scale fixed-split attention experiments."""
from pathlib import Path
import argparse,gc,json,math,random,sys,time
import numpy as np,torch
from teachers import Teacher
from data import TeacherData
from datasets import NAMES, ROOT, dataset_path, digest, read_manifest


def cyclic_positions(query,within,epoch,multipliers,offsets,count):
    return (multipliers[query]*(epoch*count+within)+offsets[query])%500


def main():
    p=argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--dataset', choices=NAMES)
    source.add_argument('--dataset-path')
    p.add_argument('--data-root');p.add_argument('--output-dir')
    p.add_argument('--kind',choices=['all','single'],required=True);p.add_argument('--seed',type=int,default=2024)
    p.add_argument('--batch',type=int);p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--epochs',type=int,default=80);p.add_argument('--patience',type=int,default=8)
    p.add_argument('--pairs-per-query',type=int,default=32);p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    path=Path(a.dataset_path) if a.dataset_path else dataset_path(a.dataset,a.data_root)
    manifest=read_manifest(path)
    if a.dataset and a.dataset!=manifest['dataset']:raise ValueError('Dataset name mismatch')
    out=Path(a.output_dir) if a.output_dir else ROOT/'runs'/manifest['dataset']/'teachers'/a.kind
    out.mkdir(parents=True,exist_ok=False)
    if a.batch is None:a.batch=8192 if a.kind=='single' else 128
    if min(a.batch,a.epochs,a.patience)<=0:raise ValueError('Batch, epochs and patience must be positive')
    if not 1<=a.pairs_per_query<=500:raise ValueError('Invalid pair count')
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.set_num_threads(4)
    data=TeacherData(path,a.device)
    single=a.kind=='single';model=Teacher(1 if single else 500,efficient=True).to(a.device)
    model.mixed_precision=True;optim=torch.optim.Adam(model.parameters(),lr=a.lr)
    n=len(data.splits['train']['y']);nv=len(data.splits['valid']['y'])
    if single:
        choices=torch.tensor([i for i in range(1,500) if math.gcd(i,500)==1],device=a.device)
        multipliers=choices[torch.randint(len(choices),(n,),device=a.device)]
        offsets=torch.randint(500,(n,),device=a.device)
    history=[];best=float('inf');best_epoch=0;started=time.time()
    for epoch in range(1,a.epochs+1):
        epoch_start=time.time();model.train();train_count=n*(a.pairs_per_query if single else 1)
        order=torch.randperm(train_count,device=a.device);total=torch.zeros((),device=a.device)
        for start in range(0,train_count,a.batch):
            selected=order[start:start+a.batch]
            if single:
                query=selected//a.pairs_per_query;within=selected%a.pairs_per_query
                pos=cyclic_positions(query,within,epoch-1,multipliers,offsets,a.pairs_per_query)
                args,y=data.batch('train',query,pos)
            else:args,y=data.batch('train',selected)
            pred=model(*args).flatten().float();loss=(pred-y).square().mean()
            if single:loss=.5*loss+.5*(model.forward_query_only(*args[:2]).flatten().float()-y).square().mean()
            if not torch.isfinite(loss):raise ValueError('Non-finite training loss')
            optim.zero_grad(set_to_none=True);loss.backward();optim.step();total+=loss.detach()*len(y)
        model.eval();valid_count=nv*(500 if single else 1);valid_total=torch.zeros((),device=a.device)
        with torch.no_grad():
            for start in range(0,valid_count,a.batch if single else 64):
                end=min(start+(a.batch if single else 64),valid_count)
                if single:
                    flat=torch.arange(start,end,device=a.device);args,y=data.batch('valid',flat//500,flat%500)
                else:args,y=data.batch('valid',slice(start,end))
                loss=(model(*args).flatten().float()-y).square()
                if single:loss=.5*loss+.5*(model.forward_query_only(*args[:2]).flatten().float()-y).square()
                valid_total+=loss.sum()
        valid=float(valid_total.double()/valid_count)
        if valid<best:
            best=valid;best_epoch=epoch
            torch.save({'config': {'retrieval_num': 1 if single else 500, 'efficient': True},
                        'mixed_precision': model.mixed_precision, 'state_dict': model.state_dict(),
                        'dataset_manifest_sha256': digest(path/'dataset.json')},
                       out/'best_model.pth')
            (out/'best.json').write_text(json.dumps({'best_epoch':epoch,'best_validation_loss':best,'config':vars(a),
                'precision':'BF16 autocast for training and inference; FP32 parameters and losses',
                'single_training_sampling':'cyclic per-query affine permutation of 500 neighbors' if single else None,
                'single_validation_neighbors':500 if single else None},indent=2))
        row={'epoch':epoch,'train_objective':float(total.double()/train_count),'valid_objective':valid,
             'best_epoch':best_epoch,'best_validation_loss':best,'seconds':time.time()-epoch_start}
        history.append(row);(out/'epochs.json').write_text(json.dumps(history,indent=2))
        print(json.dumps(row),flush=True)
        if epoch-best_epoch>=a.patience:break
    (out/'summary.json').write_text(json.dumps({'kind':a.kind,'epochs':epoch,'best_epoch':best_epoch,
        'best_validation_loss':best,'seconds':time.time()-started,'config':vars(a)},indent=2))


if __name__=='__main__':main()
