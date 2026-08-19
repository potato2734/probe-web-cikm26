import math
import os
import json
from PROBE import  PROBE
from typing import List, Sequence
import numpy as np
from collections import defaultdict
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, NullFormatter
from matplotlib import gridspec
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from tqdm import tqdm
import networkx as nx
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib import colors
from collections import Counter
from utils import get_ship, get_decoded
from scipy.stats import ttest_ind
from matplotlib.patches import Patch
from dataclasses import dataclass
from nltk.corpus import wordnet as wn
import warnings
from functools import lru_cache

warnings.filterwarnings(
    "ignore",
    message="No WordNet synset found for pos="
)


@dataclass
class Vector:
    v: np.ndarray
    epop: np.ndarray
    rpop: np.ndarray
    color: str

class VisTool():
    def __init__(self, trn_count_info, tst_count_info, trains, tests, nentity, nrelation, total_degree, rel_prob, data):
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        self.trn_count_info = trn_count_info
        self.tst_count_info = tst_count_info
        self.markers = {'RotatE': 'o', 'ComplEx': 's', 'HousE':'^', 'TuckER':'v', 'pLogicNet':'D',
                        'RNNLogic':'p', 'CompGCN':'*'}
        self.mods = {'RotatE':'#073b4c', 'ComplEx':'#118ab2','HousE':'#06d6a0', 'TuckER':'#FFB715',
                      'pLogicNet':'#F78C6B', 'RNNLogic':'#ef476f'}

        self.trains = trains[:]
        self.tests = tests[:]
        self.nentity = nentity
        self.nrelation = nrelation
        self.total_degree = total_degree
        self.rel_prob = rel_prob
        self.data = data

        self.edges = defaultdict(set)
        for q in self.trains:
            h, r, t = q
            self.edges[h].add((r, t, 'h'))
            self.edges[t].add((r, h, 't'))

    def get_model_colors(self, mets: List[List[PROBE]]):
        model_names = [_mets[0].model for _mets in mets]
        colors = [self.mods[model_name] for model_name in model_names]
        return colors

    def draw_3dhyperPlane(self, mets: List[List[PROBE]], alphas: list, betas: list, gammas: List, mode='ab'):
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111, projection='3d')

        if mode == 'ab':
            target1, target2 = alphas, betas
        elif mode == 'ag':
            target1, target2 = alphas, gammas
        elif mode == 'bg':
            target1, target2 = betas, gammas

        data = defaultdict(list)
        model_name_ls = []
        num_models = len(mets)
        for a_i, alpha in enumerate(target1):
            for b_i, beta in enumerate(target2):

                for _met in mets:
                    values = []
                    if len(model_name_ls) < num_models: model_name_ls.append(_met[0].model)
                    for met in _met:
                        if mode == 'ab': values.append(met.calculate_final_metric(alpha, beta, 0.0))
                        elif mode == 'ag': values.append(met.calculate_final_metric(alpha, 0.0, beta))
                        elif mode == 'bg': values.append(met.calculate_final_metric(1.0, alpha, beta))
                    data[(a_i, b_i)].append(sum(values) / len(values))

        for xy, z_ls in data.items():
            z_ls = np.array(z_ls)
            data[xy] = (z_ls - z_ls.min()) / (z_ls.max() - z_ls.min())

        # Number of z-values per (x, y)
        num_z = len(next(iter(data.values())))

        # Flatten into lists
        x, y, z, color_group = [], [], [], []

        for (xi, yi), z_vals in data.items():
            for idx, zi in enumerate(z_vals):
                x.append(xi)
                y.append(yi)
                z.append(zi)
                color_group.append(idx)

        color_ls = self.get_model_colors(mets)

        # Plotting
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot of all points
        for idx in range(num_z):
            x_i = [x[j] for j in range(len(x)) if color_group[j] == idx]
            y_i = [y[j] for j in range(len(y)) if color_group[j] == idx]
            z_i = [z[j] for j in range(len(z)) if color_group[j] == idx]

            ax.scatter(x_i, y_i, z_i, c=color_ls[idx], marker=self.markers[model_name_ls[idx % len(self.markers)]],
                       s=40, label=model_name_ls[idx])

        # Add faint dotted lines at each (x, y)
        # for (xi, yi), z_vals in data.items():
        #     z_line = sorted(z_vals)
        #     if alphas[xi] == 1 and betas[yi] == 0:
        #         ax.plot([xi] * len(z_line), [yi] * len(z_line), z_line, linestyle='dotted', color='black', alpha=1)
        #     else:
        #         ax.plot([xi] * len(z_line), [yi] * len(z_line), z_line, linestyle='dotted', color='gray', alpha=0.5)



        if mode == 'ab':
            ax.set_xlabel(r"$\alpha$", fontsize=13)
            ax.set_ylabel(r"$\beta$", fontsize=13)
        elif mode == 'ag':
            ax.set_xlabel(r"$\alpha$", fontsize=13)
            ax.set_ylabel(r"$\gamma$", fontsize=13)
        elif mode == 'bg':
            ax.set_xlabel(r"$\beta$", fontsize=13)
            ax.set_ylabel(r"$\gamma$", fontsize=13)
        #ax.set_title(f"Normalized metrics on multiple alphas & betas ({mets[0][0].data})", fontweight='bold')
        # Force the 3D axes to occupy the whole figure
        ax.set_xticks(np.arange(len(target1)))
        ax.set_yticks(np.arange(len(target2)))
        ax.zaxis.set_tick_params(labelsize=13)
        ax.set_xticklabels([f"{a:.2f}" for a in target1], fontsize=13)
        ax.set_yticklabels([f"{b:.1f}" for b in target2], fontsize=13)

        ax.view_init(elev=13, azim=-80)
        os.makedirs(f'../figs/2D_hyper', exist_ok=True)
        plt.savefig(f'../figs/2D_hyper/{mets[0][0].data}_hyper_{mode}.pdf', bbox_inches='tight', pad_inches=0.01)


    def draw__hyperPlane(self, mets: List[List[PROBE]], alphas: list, betas: list, gammas: List, mode='ab',
                        vis_axis=False, vis_cbar=False):
        if mode == 'ab':
            target1, target2 = alphas, betas
        elif mode == 'ag':
            target1, target2 = alphas, gammas
        elif mode == 'bg':
            target1, target2 = betas, gammas
        else:
            raise ValueError(f"Unknown mode: {mode}")

        data = defaultdict(list)
        model_name_ls = []
        num_models = len(mets)
        axis_font = 13

        for a_i, alpha in enumerate(target1):
            for b_i, beta in enumerate(target2):
                for _met in mets:
                    values = []
                    if len(model_name_ls) < num_models:
                        model_name_ls.append(_met[0].model)

                    for met in _met:
                        if mode == 'ab':
                            values.append(met.calculate_final_metric(alpha, beta, 0.0))
                        elif mode == 'ag':
                            values.append(met.calculate_final_metric(alpha, 0.0, beta))
                        elif mode == 'bg':
                            values.append(met.calculate_final_metric(1.0, alpha, beta))

                    data[(a_i, b_i)].append(sum(values) / len(values))

        for xy, z_ls in data.items():
            z_ls = np.array(z_ls, dtype=np.float64)
            denom = (z_ls.max() - z_ls.min())
            data[xy] = (z_ls - z_ls.min()) / (denom + 1e-12)

        n1, n2 = len(target1), len(target2)
        Zs = np.full((num_models, n2, n1), np.nan, dtype=np.float64)
        for (xi, yi), z_vals in data.items():
            for m_idx, z in enumerate(z_vals):
                Zs[m_idx, xi, yi] = z

        X_idx, Y_idx = np.meshgrid(np.arange(n1), np.arange(n2))

        for m_idx in range(num_models):
            if np.isnan(Zs[m_idx]).any():
                missing = np.argwhere(np.isnan(Zs[m_idx]))
                raise ValueError(
                    f"Missing grid points for model {model_name_ls[m_idx]} at indices (target1_idx, target2_idx): "
                    f"{missing.tolist()[:10]}{' ...' if len(missing) > 10 else ''}"
                )

        zmin = float(np.nanmin(Zs))
        zmax = float(np.nanmax(Zs))
        levels = np.linspace(zmin, zmax, 256)  # adjust 12

        def _is_YAGO():
            return self.data == 'YAGO3-10'

        def _is_FB():
            return self.data == 'FB15k237'

        nrow, ncol, mag = 1, 6, 1.3
        fig_row, axes = plt.subplots(
            nrow, ncol,
            figsize=(ncol * mag, nrow * mag),
            constrained_layout=True
        )
        axes = np.array(axes).reshape(-1)

        if num_models == 1:
            axes = [axes]

        last_csf = None
        cmap_scheme = 'RdYlBu_r'
        for m_idx, ax2 in enumerate(axes):
            fig, ax2 = plt.subplots(figsize=(mag, mag), constrained_layout=True)
            if _is_YAGO() and m_idx + 1 == len(axes):
                ax2.set_facecolor("white")
                ax2.set_xticks([])
                ax2.set_yticks([])
            else:
                Z = Zs[m_idx]
                csf = ax2.contourf(
                    Y_idx, X_idx, Z,
                    levels=levels,
                    cmap=cmap_scheme,
                    antialiased=True
                )
                ax2.contour(Y_idx, X_idx, Z,
                            levels=levels,
                            cmap=cmap_scheme,
                            linewidths=0.6)

            x_mid = sum(ax2.get_xlim()) / 2
            y_mid = sum(ax2.get_ylim()) / 2

            if not (m_idx == 5 and _is_YAGO()):
                ax2.axvline(x_mid, color='grey', lw=1, ls='--')
                ax2.axhline(y_mid, color='grey', lw=1, ls='--')

            if m_idx == 5 and _is_YAGO():
                ax2.text(
                    0.5, 0.5,
                    "O.O.M",
                    ha='center',
                    va='center',
                    fontsize=15,
                    transform=plt.gca().transAxes
                )

            if m_idx == 0:
                origin = (0.9, 0.1)

                ax2.annotate(
                    '',
                    xy=(0.7, 0.1), 
                    xytext=origin,
                    arrowprops=dict(arrowstyle='-|>', lw=1, fc='black'),
                    xycoords='axes fraction'
                )

                ax2.annotate(
                    '',
                    xy=(0.9, 0.3),  # up
                    xytext=origin,
                    arrowprops=dict(arrowstyle='-|>', lw=1, fc='black'),
                    xycoords='axes fraction'
                )

                # labels
                ax2.text(0.69, 0.1, r'$\alpha$', transform=ax2.transAxes, ha='right', va='center')
                ax2.text(0.9, 0.31, r'$\beta$', transform=ax2.transAxes, ha='center', va='bottom')

            ax2.tick_params(axis='both', which='both', length=0)
            ax2.set_xticks(np.arange(n1))
            ax2.set_yticks(np.arange(n2))
 
            if vis_axis:
                if m_idx >= 0:
                    ax2.set_yticklabels([f"↑" if v in {target1[len(target1) // 2]} else "" for v in target1], fontsize=axis_font)
                else:
                    ax2.set_yticklabels([])
                ax2.set_xticklabels([f"→" if v in {target2[len(target2) // 2]} else "" for v in target2], fontsize=axis_font)
            else:
                ax2.set_xticks([])
                ax2.set_yticks([])

            ax2.set_xlim(0, n1 - 1)
            ax2.set_ylim(0, n2 - 1)
            ax2.set_aspect("equal", adjustable="box")
            os.makedirs(f'../figs/fig11', exist_ok=True)
            if (_is_YAGO() and m_idx + 1 != len(axes)) or (not _is_YAGO()): last_csf = csf
            if _is_YAGO() and m_idx == 5: save_path = f'../figs/fig11/{self.data}_RNNLogic_contour_ab.pdf'
            else: save_path = f'../figs/fig11/{self.data}_{model_name_ls[m_idx]}_contour_ab.pdf'
            fig.savefig(save_path, bbox_inches='tight', pad_inches=0.01, dpi=300)
            plt.close(fig)


    def draw_rank_cluster_hist(
            self,
            mets: List[List["PROBE"]],
            cuts: Sequence[int] = (1,2,6,21,51,100),
            density: bool = False,
            both: bool = True,
            cut_last: bool = True,
            diff: bool = True,
            format: str = 'pdf',
            xlabel: bool = True
    ):

        models = [row[0].model for row in mets]
        color_ls = self.get_model_colors(mets)
        color_ls.append('grey')
        H, W = len(mets), len(mets[0])
        data = mets[0][0].data

        all_ranks = [[] for _ in range(H)]
        for i, _mets in enumerate(mets):
            for j, met in enumerate(_mets):
                all_ranks[i].append(np.asarray(met.rank, dtype=float))

        cuts = list(cuts)
        if len(cuts) == 0:
            raise ValueError("`cuts` must contain at least one integer.")
        if any(c <= 0 for c in cuts):
            raise ValueError("`cuts` should be positive integers.")
        if any(cuts[i] >= cuts[i + 1] for i in range(len(cuts) - 1)):
            raise ValueError("`cuts` must be strictly increasing.")

        slices = []
        for a, b in zip(cuts[:-1], cuts[1:]):
            slices.append((a, b - 1))
        if not cut_last: slices.append((cuts[-1], np.inf))

        def _lab(lo, hi):
            if lo < hi:
                return f"{int(lo)}~{int(hi)}" if np.isfinite(hi) else f"Others"
            elif lo == hi:
                return f"{int(lo)}"

        x_labels = [_lab(lo, hi) for (lo, hi) in slices]
        nbins = len(slices)

        fig, ax = plt.subplots(figsize=(5,2))
        total_width = 0.7
        bar_width = total_width / H
        centers = np.arange(nbins, dtype=float)  
        offsets = (np.arange(H) - (H - 1) / 2.0) * bar_width
        means = []
        stds = []
        for i in range(H):
            counts_list = []
            target_ranks = all_ranks[i]
            for r in target_ranks:
                counts = np.zeros(nbins, dtype=float)
                if r.size > 0:
                    for k, (lo, hi) in enumerate(slices):
                        if np.isfinite(hi):
                            counts[k] = np.sum((r >= lo) & (r <= hi))
                        else:
                            counts[k] = np.sum(r >= lo)

                if density:
                    s = counts.sum()
                    if s > 0:
                        counts = counts / s
                counts_list.append(counts)

            arr = np.stack(counts_list, axis=0)  
            means.append(arr.mean(axis=0)) 
            stds.append(arr.std(axis=0, ddof=1) if W > 1 else np.zeros_like(means[i]))

            bar_color = self.mods.get(models[i], None)

            if both:
                bar_color = self.mods.get(models[i], None)
                xbar = centers + offsets[i]
                ax.bar(
                    xbar,
                    means[i],
                    width=bar_width,
                    label=models[i],
                    capsize=3,
                    color=bar_color, 
                    linewidth=0.6,
                    zorder=2,
                    error_kw = {
                        "elinewidth": 1.8,  
                        "capthick": 1.8 
                    }
                )

        axis_font = 13
        highest_points = np.max(np.array(means) + np.array(stds), axis=0)
        lowest_points = np.min(np.array(means) - np.array(stds), axis=0)
        if diff:
            try:
                diff = highest_points - lowest_points
                win_sign = []
                for _diff in means[0] - means[1]:
                    _diff = int(_diff)
                    if _diff > 0: win_sign.append(0)
                    elif _diff < 0: win_sign.append(1)
                    else: win_sign.append(2)

                for i in range(len(highest_points)):
                    ax.text(
                        centers[i],
                        highest_points[i] + 100,
                        s=f'+{int(diff[i])}',
                        color=color_ls[win_sign[i]],
                        fontsize=axis_font,
                        fontweight='bold',
                        ha='center'
                    )
            except: pass

        if not cut_last:
            x_pos = sum(centers[len(centers)-2:]) / 2
            plt.axvline(x_pos, ymin=0, ymax=1, color='grey', linestyle='--')

        if not both:
            final_mean = means[0] - means[1]
            bar_color = []
            for i in range(len(means[0])):
                if final_mean[i] > 0: bar_color.append(self.mods.get(models[0], None))
                elif final_mean[i] < 0: bar_color.append(self.mods.get(models[1], None))
                else: bar_color.append('grey')
                final_mean[i] = abs(final_mean[i])
            xbar = centers
            ax.bar(
                xbar,
                final_mean,
                width=bar_width,
                capsize=3,
                color=bar_color,
                edgecolor="None",
                linewidth=0.6,
                zorder=2
            )
   
            for i in range(H):
                bar_color = self.mods.get(models[i], None)
                ax.scatter(
                    centers, means[i],
                    s=18,
                    marker=self.markers[models[i]],
                    color=bar_color,
                    zorder=4,
                    label=models[i],
                )

                ax.plot(
                    centers, means[i],
                    linewidth=1.0,
                    linestyle='-',
                    color=bar_color,
                    zorder=3,
                )

        plt.yscale('log')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis="x", length=0)
        ax.set_xticks(centers)

        if self.data == 'FB15k237': plt.yticks([2000,4000,8000],['2k','4k','8k'])
        else: plt.yticks([1000,2000,4000],['1k','2k','4k'])
        ax.tick_params(axis='y', which='both', left=False, labelleft=True, labelsize=axis_font)
        #ax.set_xticklabels(x_labels, rotation=45, fontsize=10)
        # if cut_last: ax.set_xticklabels(['1st','2nd~5th','6th~20th','21st~50th','51st~100th','101st~200th'], fontsize=15, rotation=24)
        # else: ax.set_xticklabels(['1st','2nd~5th','6th~20th','21st~50th','51st~100th','101st~200th','Others'], fontsize=15)
        xtick_labels = []
        for i, cut in enumerate(cuts):
            if i == len(cuts) - 1: break
            if cuts[i + 1] - cut == 1: xtick_labels.append(f'{cut}')
            else:
                if cuts[-1] == cuts[i + 1]: xtick_labels.append(f'{cut}-{cuts[i + 1]}')
                else: xtick_labels.append(f'{cut}-{cuts[i + 1] - 1}')
        # if cut_last and len(cuts) == 6:
        #     #ax.set_xticklabels(['1st','2nd-5th','6th-20th','21st-50th','51st-100th'], fontsize=axis_font)
        #     ax.set_xticklabels(['1', '2-5', '6-20', '21-50', '51-100'], fontsize=axis_font)
        # elif cut_last and len(cuts) == 7: ax.set_xticklabels(['1', '2-5', '6-20', '21-50', '51-100', '101-200'], fontsize=axis_font)
        # else:
        #     #ax.set_xticklabels(['1st','2nd~5th','6th~20th','21st~50th','51st~100th','101st~200th','Others'], fontsize=axis_font)
        ax.set_xticklabels(xtick_labels,
                           fontsize=axis_font)
        if xlabel: ax.set_xlabel('Rank', fontsize=axis_font, labelpad=0)
        ax.yaxis.grid(True, color="grey", linestyle="-", linewidth=0.5, alpha=0.5, zorder=0)
        ax.set_ylabel("Normalized frequency" if density else "# prediction", fontsize=axis_font, labelpad=0)
        ax.legend(ncol=2, fontsize=axis_font - 1,loc='upper right', borderaxespad=0.0)
        ax.minorticks_off()
        ax.set_ylim(max(0, min(lowest_points) - 200), max(highest_points) + 500)
        plt.tight_layout()
        model_names = '_'.join(models)
        os.makedirs('../figs/fig2_obs1_fig9/', exist_ok=True)
        plt.savefig(f'../figs/fig2_obs1_fig9/{data}_rank_hist_{model_names}.{format}', bbox_inches='tight', pad_inches=0.01)

    def fbdecode(self, id2e):
        assert self.data == 'FB15k237', f'This function can not be used in {self.data}'
        _id2e = {}
        with open(f'../data/{self.data}/FB_decoded.json') as f:
            decoded_entities = json.load(f)

        for id, ename in id2e.items():
            try:
                _id2e[id] = decoded_entities[ename]
            except:
                _id2e[id] = ename
        return _id2e


    class Info:
        def __init__(self, ls):
            self.query = ls[0]
            self.target = ls[1]
            self.pop = float(ls[2])
            self.rank = int(float(ls[3][5:]))
            self.scores = np.array([float(str_score) for str_score in ls[4:]])

    def draw_e_cases(self, args, e2id, r2id, threshold):
        who_win = args.models[0]
        model1, model2 = args.models[0], args.models[1]
        model1_path, model2_path = f'../case_info/{args.data}_{model1}_case.json', \
            f'../case_info/{self.data}_{model2}_case.json'
        ls1, ls2 = get_ship(model1_path), get_ship(model2_path)
        os.makedirs(f'../figs/fig10_top/{self.data}/{model1}', exist_ok=True)
        os.makedirs(f'../figs/fig10_top/{self.data}/{model2}', exist_ok=True)

        if self.data == 'FB15k237':
            decoded = get_decoded(f'../data/FB15k237/FB_decoded.json')
            new_e2id = {}
            for ename, eid in e2id.items():
                try:
                    new_e2id[decoded[ename]] = eid
                except:
                    new_e2id[ename] = eid
            e2id = new_e2id

        def decoder_check(query):
            for el in query:
                if 'm/' in el: return False
            return True

        def win_check(rank1, rank2):
            if who_win == model1:
                return rank1 < rank2
            else:
                return rank1 > rank2

        def norm_win_check(norm1, rank1, norm2, rank2):
            if who_win == model1:
                return norm1[100 - rank1] > norm2[100 - rank2]
            else:
                return norm1[100 - rank1] < norm2[100 - rank2]

        total_pairs = min(len(ls1), len(ls2)) 

        for l1, l2 in tqdm(zip(ls1, ls2),
                           total=total_pairs,
                           desc="Rendering case figs",
                           unit="pair",
                           dynamic_ncols=True):
            mod1_info, mod2_info = self.Info(l1), self.Info(l2)
            assert mod1_info.query == mod2_info.query
            assert mod1_info.target == mod2_info.target
            assert mod1_info.pop == mod2_info.pop

            def normalize(x):
                mn, mx = np.min(x), np.max(x)
                return np.zeros_like(x) if mx == mn else (x - mn) / (mx - mn)

            list1_norm = normalize(mod1_info.scores)[::-1]
            list2_norm = normalize(mod2_info.scores)[::-1]

            who_win = model1 if mod1_info.pop > threshold else model2

            if mod1_info.rank <= 100 and \
                mod2_info.rank <= 100 and \
                decoder_check(mod1_info.query) and \
                win_check(mod1_info.rank, mod2_info.rank) and \
                norm_win_check(list1_norm, mod1_info.rank, list2_norm, mod2_info.rank):

                def sanitize_filename(s):
                    return s.replace('/', '_').replace('%', '_').replace(':', '_').replace(' ', '_')

                fig, ax = plt.subplots(figsize=(2, 2))

                wo_ans_list1_norm, wo_ans_list2_norm = np.delete(list1_norm, 100 - mod1_info.rank), np.delete(
                    list2_norm, 100 - mod2_info.rank)
                vp = ax.violinplot(
                    [wo_ans_list1_norm, wo_ans_list2_norm],
                    positions=[-0.5, 0.5],
                    widths=0.35,
                    vert=True,
                    showmeans=False,
                    showmedians=False,
                    showextrema=False
                )

                for body, color in zip(vp['bodies'], [self.mods[model1], self.mods[model2]]):
                    body.set_facecolor(color)
                    body.set_alpha(0.3)
                    body.set_edgecolor('black')  
                    body.set_linewidth(0.8) 

                ax.scatter([-0.5] * len(wo_ans_list1_norm), wo_ans_list1_norm, s=30, marker='x', color=self.mods[model1],
                           label=model1, alpha=0.5)
                ax.scatter([0.5] * len(wo_ans_list2_norm), wo_ans_list2_norm, s=30, marker='x', color=self.mods[model2],
                           label=model2, alpha=0.5)

                idx1 = 100 - mod1_info.rank
                idx2 = 100 - mod2_info.rank
                ax.scatter(-0.5, list1_norm[idx1], s=150, marker='*', color=self.mods[model1], edgecolor='black',
                           linewidths=0.6, zorder=3)
                ax.scatter(0.5, list2_norm[idx2], s=150, marker='*', color=self.mods[model2], edgecolor='black',
                           linewidths=0.6, zorder=3)

                who_bold = lambda x: 'bold' if x == who_win else 'normal'
                ax.text(-0.3, list1_norm[idx1], f'{mod1_info.rank}', va='center', fontsize=10,
                        fontweight=who_bold(model1))
                ax.text(0.7, list2_norm[idx2], f'{mod2_info.rank}', va='center', fontsize=10,
                        fontweight=who_bold(model2))

                for i in range(6):
                    ax.axhline(i * 0.2, color='grey', linestyle='--', alpha=0.5, linewidth=0.5)

                ax.axvline(0, color='black', linewidth=0.5)

                ax.set_ylim(-0.05, 1.05)

                axis_fontsize = 12
                ax.set_xlim(-1, 1)
                ax.set_xticks([-0.5, 0.5])
                ax.set_xticklabels([f'{model1}', f'{model2}'], fontsize=axis_fontsize)
                ax.set_yticks([])

                ax.grid(True, axis='y', linestyle='--', alpha=0.35)
                new_path = f'../figs/fig10_top/{args.data}/{who_win}'

                plt.title(f'Top {round(mod1_info.pop, 2)}%')
                rank_info = f'{mod1_info.rank} vs {mod2_info.rank}'
                q2ids = f'{e2id[mod1_info.query[0]]}.{r2id[mod1_info.query[1]]}.{e2id[mod1_info.query[2]]}'
                filename = f"{self.data}_top{round(mod1_info.pop, 4)}_{q2ids}_{rank_info}.pdf"
                filename = sanitize_filename(filename)

                fig.savefig(os.path.join(new_path, filename), dpi=300, bbox_inches='tight', pad_inches=0.01)
                plt.close(fig)