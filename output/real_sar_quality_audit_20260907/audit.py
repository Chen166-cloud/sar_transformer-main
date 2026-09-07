from pathlib import Path
import csv
import hashlib
import json
import re
import time
import numpy as np
from scipy.io import loadmat
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA = ROOT / 'datasets/real_sar_dataset'
manifest_path = DATA / 'real_split_grouped_seed42.json'
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
rows, errors = [], []
started = time.time()
for split, names in manifest['files'].items():
    for name in names:
        try:
            payload = loadmat(DATA / name)
            x = payload['noisy']
            u, counts = np.unique(x, return_counts=True)
            finite = bool(np.isfinite(x).all())
            grid_n, grid_error = None, None
            if finite and len(u) > 1 and u[0] == 0 and u[-1] == 1:
                grid_n = int(round(1 / float(u[1]) ** 2))
                sq = u.astype(np.float64) ** 2 * grid_n
                grid_error = float(np.max(np.abs(sq - np.round(sq))))
            rows.append(dict(path=name, split=split, shape='x'.join(map(str,x.shape)),
                dtype=str(x.dtype), finite=finite, unique_values=len(u), minimum=float(x.min()),
                maximum=float(x.max()), mean=float(x.mean()), std=float(x.std()),
                zero_fraction=float(np.mean(x==0)), one_fraction=float(np.mean(x==1)),
                sqrt_grid_n=grid_n, sqrt_grid_max_error=grid_error,
                sqrt_grid_fit=grid_error is not None and grid_error < 0.001))
        except Exception as exc:
            errors.append({'path':name, 'error':repr(exc)})
    print(f'{split}: scanned {len(rows)} patches', flush=True)
with (OUT/'per_patch.csv').open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def distribution(key):
    a=np.array([r[key] for r in rows],dtype=float)
    return {k:float(v) for k,v in zip(['min','p25','median','p75','p95','max'],np.percentile(a,[0,25,50,75,95,100]))}
summary = {'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    'total':len(rows),'errors':errors,'shapes':sorted(set(r['shape'] for r in rows)),
    'dtypes':sorted(set(r['dtype'] for r in rows)), 'all_finite':all(r['finite'] for r in rows),
    'unique_values':distribution('unique_values'),
    'patches_at_most_64_values':sum(r['unique_values']<=64 for r in rows),
    'patches_at_most_256_values':sum(r['unique_values']<=256 for r in rows),
    'patches_min_zero_max_one':sum(r['minimum']==0 and r['maximum']==1 for r in rows),
    'mean_zero_fraction':float(np.mean([r['zero_fraction'] for r in rows])),
    'mean_one_fraction':float(np.mean([r['one_fraction'] for r in rows])),
    'one_fraction':distribution('one_fraction'),
    'sqrt_grid_fit_count':sum(r['sqrt_grid_fit'] for r in rows),
    'note':'sqrt grid is an observed numeric pattern, not proof of the processing history.'}

stem='4608_3584_y0_x256'
selected=next(r for r in rows if Path(r['path']).stem==stem)
x=loadmat(DATA/selected['path'])['noisy']
noisy_paths=sorted((ROOT/'test_results/real_sar').rglob('*05_45_texture_'+stem+'_noisy.png'))
comparisons=[]
for p in noisy_paths:
    png=np.asarray(Image.open(p).convert('L'))
    expected=np.round(np.clip(x,0,1)*255).astype(np.uint8)
    comparisons.append({'path':p.relative_to(ROOT).as_posix(), 'shape':list(png.shape),
        'exactly_round_raw_times_255':bool(png.shape==x.shape and np.array_equal(png,expected)),
        'unique_values':len(np.unique(png)),
        'max_absolute_error_unit':float(np.max(np.abs(png.astype(float)/255-x))) if png.shape==x.shape else None})
summary['selected_patch']=selected
summary['selected_export_comparisons']=comparisons
summary['existing_example_image_sizes']={p.name:list(Image.open(p).size) for p in (ROOT/'examples/realsar').glob('*.png')}

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
fig,axes=plt.subplots(2,3,figsize=(12,8),layout='constrained')
axes[0,0].imshow(x,cmap='gray',vmin=0,vmax=1,interpolation='nearest')
axes[0,0].set_title('Stored MAT: 256 x 256\n'+str(selected['unique_values'])+' distinct values')
export=np.asarray(Image.open(noisy_paths[0]).convert('L'))
axes[0,1].imshow(export,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
axes[0,1].set_title('Existing Noisy PNG: 256 x 256\n'+str(len(np.unique(export)))+' distinct values')
u,c=np.unique(x,return_counts=True)
axes[0,2].bar(u,c/x.size,width=0.003,color='#285b72')
axes[0,2].set_title('Stored MAT value distribution')
axes[0,2].set_xlabel('Stored value'); axes[0,2].set_ylabel('Fraction of pixels')
crop=x[80:144,80:144]
axes[1,0].imshow(crop,cmap='gray',vmin=0,vmax=1,interpolation='nearest')
axes[1,0].set_title('Same 64 x 64 MAT crop\nNearest-neighbour display')
axes[1,1].imshow(export[80:144,80:144],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
axes[1,1].set_title('Same 64 x 64 PNG crop\nNearest-neighbour display')
axes[1,2].hist([r['unique_values'] for r in rows],bins=40,color='#285b72')
axes[1,2].set_title('All '+str(len(rows))+' MAT patches')
axes[1,2].set_xlabel('Distinct values per patch'); axes[1,2].set_ylabel('Number of patches')
for ax in [axes[0,0],axes[0,1],axes[1,0],axes[1,1]]: ax.set_axis_off()
fig.suptitle('Real SAR quality audit | '+stem+'\nNumerical-data rendering only; no denoising or detail generation',fontsize=14)
fig.savefig(OUT/'quality_comparison.png',dpi=150)
plt.close(fig)

targets=[rows[0],selected, sorted(rows,key=lambda r:r['unique_values'])[len(rows)//2],max(rows,key=lambda r:r['unique_values'])]
fig,axes=plt.subplots(1,4,figsize=(13,4),layout='constrained')
for ax,r in zip(axes,targets):
    a=loadmat(DATA/r['path'])['noisy']
    ax.imshow(a,cmap='gray',vmin=0,vmax=1,interpolation='nearest'); ax.set_axis_off()
    ax.set_title(Path(r['path']).stem+'\n'+str(r['unique_values'])+' values',fontsize=10)
fig.suptitle('Stored noisy patches | fixed [0, 1] display range')
fig.savefig(OUT/'raw_samples.png',dpi=150); plt.close(fig)
summary['elapsed_seconds']=round(time.time()-started,2)
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
