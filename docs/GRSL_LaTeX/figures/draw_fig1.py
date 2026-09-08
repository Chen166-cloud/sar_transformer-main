"""Editable GRSL architecture; 182 mm two-column vector PDF and SVG.
Run: python draw_fig1.py --output-dir <directory>. Requires Matplotlib.
No experimental images or measurements are read or changed.
"""
from pathlib import Path
import argparse, hashlib, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path as MplPath
from matplotlib import font_manager
W, H = 516 / 72.27 * 72, 262
INK, MUTED, LINE = '#22313B', '#53616A', '#CFD7DC'
GRAY = ('#EFF2F4', '#71818D')
FOURIER = ('#FFF0D9', '#B77321')
GUIDE = ('#F0EAF8', '#78639D')
DOMAIN = ('#E5F2EF', '#397D72')
SKIP = ('#EAF1FA', '#4776A8')
WHITE = ('#FFFFFF', '#81919C')

def make_figure(output_dir, font='Arial', dpi=600):
    font_path = font_manager.findfont(font, fallback_to_default=False)
    plt.rcParams.update({'font.family':font, 'font.size':8.5,
        'svg.fonttype':'none', 'svg.hashsalt':'grsl-fig1-v2', 'pdf.fonttype':42,
        'ps.fonttype':42, 'mathtext.fontset':'custom', 'mathtext.rm':font,
        'mathtext.it':font+':italic', 'mathtext.bf':font+':bold', 'axes.unicode_minus':False})
    fig=plt.figure(figsize=(W/72,H/72),facecolor='white')
    ax=fig.add_axes([0,0,1,1]); ax.set(xlim=(0,W),ylim=(H,0)); ax.axis('off')
    all_text, fitted_text = [], []
    def label(x,y,s,size=8.5,bold=False,ha='center',color=INK,bounds=None):
        t=ax.text(x,y,s,ha=ha,va='center',fontsize=size,fontweight='bold' if bold else 'normal',color=color,linespacing=1.15,zorder=8)
        all_text.append(t)
        if bounds: fitted_text.append((t,bounds,s))
        return t
    def card(x,y,w,h,s='',style=WHITE,size=8.5,bold=False,radius=3):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={radius}',facecolor=style[0],edgecolor=style[1],lw=.7,zorder=3))
        if s: label(x+w/2,y+h/2,s,size,bold,bounds=(x,y,w,h))
    def arrow(points,color=INK,dashed=False,head=True,lw=.85):
        path=MplPath(points,[MplPath.MOVETO]+[MplPath.LINETO]*(len(points)-1))
        ax.add_patch(FancyArrowPatch(path=path,arrowstyle='-|>' if head else '-',mutation_scale=5.8,color=color,lw=lw,linestyle=(0,(3,2)) if dashed else 'solid',joinstyle='round',capstyle='round',zorder=2))
    def point(x,y,color): ax.plot(x,y,'o',color=color,markersize=2.1,zorder=6)
    def panel(x,y,letter,title):
        label(x,y,letter,10,True,ha='left'); label(x+13,y,title,10,True,ha='left')
    panel(8,9,'a','Multi-scale reconstruction')
    card(294,5,8,8,style=GRAY,radius=1); label(307,9,'Inherited',ha='left')
    for x,style in [(354,SKIP),(360,FOURIER),(366,GUIDE),(372,DOMAIN)]: card(x,5,5,8,style=style,radius=.6)
    label(385,9,'Added',ha='left'); label(504,9,r'$H^2\!\times C$',ha='right',color=MUTED)
    card(308,21,188,31,'Guidance generator\n'+r'Shared latent map $G$',GUIDE,9,True)
    arrow([(106,58),(106,36.5),(308,36.5)],GUIDE[1]); point(106,58,GUIDE[1])
    card(8,58,54,33,r'Intensity $Y$'+'\n'+r'$256^2\!\times1$',WHITE,9)
    card(74,58,62,33,'Bounded log\n'+r'$T_\alpha,\ \alpha=10$',DOMAIN,9)
    arrow([(62,74.5),(74,74.5)]); arrow([(136,74.5),(148,74.5)])
    enc_x=[148,222,296,370,444]
    for i,(x,hw,c) in enumerate(zip(enc_x,[128,64,32,16,8],[32,64,128,320,512]),1):
        card(x,58,62,33,style=GRAY)
        label(x+31,67.5,rf'$E_{i}$',10,True); label(x+31,82,rf'${hw}^2\!\times{c}$')
        if i<5: arrow([(x+62,74.5),(enc_x[i],74.5)])
    # Four vertical, same-resolution encoder skips. D0 has no encoder skip.
    for x in enc_x[:4]: arrow([(x+31,91),(x+31,112)],SKIP[1],lw=1.0)
    label(84,101,'Encoder skips',ha='left',color=MUTED); arrow([(475,91),(475,112)])
    dec_x=[74,148,222,296,370]
    for i,(x,hw,c) in enumerate(zip(dec_x,[256,128,64,32,16],[16,32,64,128,320])):
        card(x,112,62,46,style=WHITE)
        label(x+31,121,rf'$D_{i}$',10,True); label(x+31,134,rf'${hw}^2\!\times{c}$')
        card(x+3,142,31,13,'FDR',FOURIER,radius=1.5)
        card(x+37,142,22,13,'GM',GUIDE,radius=1.5)
        if i<4: arrow([(dec_x[i+1],132),(x+62,132)])
    card(444,112,62,46,style=FOURIER)
    label(475,121,'Bottleneck',8.5,True); label(475,134,r'$8^2\!\times512$'); label(475,148.5,'Local + FDR')
    arrow([(444,132),(432,132)])
    card(8,112,54,46,style=WHITE)
    label(35,121,'Log head',9,True)
    card(11,129,48,12,'Projection',GRAY,radius=1.5)
    card(11,143,48,12,'Sigmoid',DOMAIN,radius=1.5)
    arrow([(74,132),(62,132)]); arrow([(35,158),(35,173)])
    label(35,181,r'$\widehat{Z}$',10); label(53,181,'to (c)',ha='left',color=MUTED)
    # The map is shared, not the FDR or per-stage modulation parameters.
    arrow([(496,36.5),(511,36.5),(511,168),(122,168)],GUIDE[1],True,False)
    for x in dec_x:
        arrow([(x+48,168),(x+48,158)],GUIDE[1],True); point(x+48,168,GUIDE[1])
    label(305,181,'Six FDR blocks with independent parameters',color=MUTED)
    ax.plot([8,506],[194,194],color=LINE,lw=.65,zorder=1)
    ax.plot([233,233],[202,256],color=LINE,lw=.65,zorder=1)
    panel(8,205,'b','Decoder stage')
    recipe=[(8,28,'Up',GRAY),(44,54,'Skip fusion',SKIP),(106,32,'FDR',FOURIER),(146,44,'Residual',GRAY),(198,25,'GM',GUIDE)]
    for j,(x,w,s,style) in enumerate(recipe):
        card(x,220,w,21,s,style,radius=2)
        if j<len(recipe)-1: arrow([(x+w,230.5),(recipe[j+1][0],230.5)],lw=.8)
    label(210.5,207,r'$G$',8.5)
    arrow([(210.5,212),(210.5,220)],GUIDE[1],True)
    label(8,253,'Concat + fusion on D1-D4; GM uses resized G.',ha='left',color=MUTED)
    panel(245,205,'c','Intensity reconstruction')
    label(253,235,r'$\widehat{Z}$',10); arrow([(264,235),(278,235)])
    card(278,222,40,26,r'$T_\alpha^{-1}$',DOMAIN,10)
    arrow([(318,235),(354,235)]); label(336,221,r'$\widetilde{X}$',9)
    card(354,223,98,25,'Compensation\n+ clip [0, 1]',DOMAIN)
    label(428,212,r'$Y$',9); arrow([(428,216),(428,223)],DOMAIN[1])
    arrow([(452,235),(473,235)]); label(487,235,r'$\widehat{X}$',10)
    label(245,254,'Fusion after inversion, in the intensity domain',ha='left',color=MUTED)
    fig.canvas.draw(); renderer=fig.canvas.get_renderer(); failures=[]
    for t,(x,y,w,h),s in fitted_text:
        bb=t.get_window_extent(renderer).transformed(ax.transData.inverted())
        if bb.x0<x+.5 or bb.x1>x+w-.5 or min(bb.y0,bb.y1)<y+.2 or max(bb.y0,bb.y1)>y+h-.2: failures.append(s)
    for t in all_text:
        bb=t.get_window_extent(renderer).transformed(ax.transData.inverted())
        if bb.x0<0 or bb.x1>W or min(bb.y0,bb.y1)<0 or max(bb.y0,bb.y1)>H: failures.append('Page bounds: '+t.get_text())
    if failures: raise RuntimeError('Text does not fit: '+repr(failures))
    output_dir.mkdir(parents=True,exist_ok=True); base=output_dir/'fig1_overall_architecture'
    for ext in ('pdf','svg','png'):
        kwargs={'dpi':dpi,'facecolor':'white'}
        if ext=='pdf': kwargs['metadata']={'Title':'SAR despeckling: architecture and reconstruction','Author':'Hongyu Chen, Peng Liu, Yaqiu Jin','Creator':'draw_fig1.py / Matplotlib','CreationDate':None,'ModDate':None}
        if ext=='svg': kwargs['metadata']={'Date':None}
        fig.savefig(base.with_suffix('.'+ext),**kwargs)
    fig.savefig(output_dir/'fig1_actual_size_150dpi.png',dpi=150,facecolor='white')
    manifest={'width_pdf_pt':W,'height_pdf_pt':H,'width_mm':W/72*25.4,'minimum_font_pt':8.5,'font':font,'font_file':font_path,'vector_pdf':True,'editable_svg_text':True,'png_dpi':dpi,'boxed_text_overflows':failures,'matplotlib_version':matplotlib.__version__,'six_independent_fdr_blocks':True,'shared_guidance_map_only':True,'skip_mapping':{'E1':'D1','E2':'D2','E3':'D3','E4':'D4'},'decoder_order':['Upsample','Skip concatenation and fusion (D1-D4)','FDR','Residual convolution','Guidance modulation'],'data':'Architecture only; no experimental image or metric is drawn.','drawing_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output_dir/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    plt.close(fig); print(json.dumps({'output_dir':str(output_dir),'text_overflows':failures}))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--font',default='Arial'); parser.add_argument('--dpi',type=int,default=600)
    args=parser.parse_args(); make_figure(args.output_dir,args.font,args.dpi)
