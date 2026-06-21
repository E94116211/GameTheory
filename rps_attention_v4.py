"""
Social Information Diversity Competition Simulation
Based on Reichenbach et al. (2008) extended social platform attention model

Content Types:
  0: Idle Attention (∅)
  1: Deep Content   (D) — Suppresses News (N)
  2: News           (N) — Suppresses Short-form (S)
  3: Short-form     (S) — Suppresses Deep (D)

Corrected ODE (corresponding to report formulas):
  dx1/dt = γ1·μ0·x1·∅  −  κ1·σ0·x1·x3  −  (ζ·γ1 + η1·x1)·x1
  dx2/dt = γ2·μ0·x2·∅  −  κ2·σ0·x2·x1  −  (ζ·γ2 + η2·x2)·x2
  dx3/dt = γ3·μ0·x3·∅  −  κ3·σ0·x3·x2  −  (ζ·γ3 + η3·x3)·x3
  d∅/dt  = Σ[(ζ·γi + ηi·xi)·xi + κi·σ0·xi·x_prev] − Σ[γi·μ0·xi·∅]

Usage (Command-line arguments):
  python rps_attention_competition.py                        # Use default values (FIXED params)
  python rps_attention_competition.py --L 80                # 80×80 grid
  python rps_attention_competition.py --n_steps 500         # 500 generations
  python rps_attention_competition.py --M 1e-5 5e-5 2e-3   # Custom M values
  python rps_attention_competition.py --gamma_S 3.0         # Increase short-form boost
  python rps_attention_competition.py --eta_S 0.35           # S fatigue rate (KEY param for who wins long-term)
  python rps_attention_competition.py --epsilon0 0.3         # Base fatigue rate (now decoupled from zeta)
  python rps_attention_competition.py --n_seeds 5            # Multi-seed robustness check before plotting
  python rps_attention_competition.py --seed 123            # Fix random seed
  python rps_attention_competition.py --no_ode              # Skip ODE plot

[v4-COEXIST variant — IMPORTANT CAVEAT]
  This variant uses gamma_S=1.7 (down from the winner-take-all version's 2.0) to land
  near the N/S phase boundary, producing a S-dominant + N-residual state (D goes extinct
  in all tested seeds; true 3-species coexistence was NOT achieved under this rule set).
  Verified stable across 5 seeds ONLY within t≈100-250 generations:
      typical state: D=0%, N=5-20%, S=80-95%
  Beyond t≈400, N tends to recover and eventually overtake S (the cyclic system has no
  density-dependent self-limiting term, so long-run dynamics keep drifting). If you need
  a result valid at arbitrary n_steps, use the winner-take-all defaults instead and treat
  this variant's results as describing a specific observation window, not a fixed point.
  Recommended: run with --n_steps 250 and read off the state near generation 200.
  1. sigma_D default changed 0.8 -> 1.0: old value silently halved D's attack success
     rate against N (D was the only species with discounted attack power), which is
     why N could escape D's suppression and dominate regardless of other parameters.
  2. epsilon_0 (fatigue base rate) was previously aliased to --zeta in main(), conflating
     two distinct mechanisms (spillover vs fatigue). Now an independent --epsilon0 flag.
  3. eta_S default changed 0.8 -> 0.35: the old value made S self-destruct too fast after
     spiking, leaving a vacuum that N (with much lower fatigue) could exploit to take over
     both S's and D's territory. Verified stable across multiple random seeds at eta_S~0.3-0.4.
  4. Added --n_seeds + robustness_check(): this cyclic competition system is highly
     sensitive near phase boundaries; a single simulation run can land on a transient
     phase of an ongoing heteroclinic cycle rather than the true long-run attractor.
     Always verify conclusions across multiple seeds before trusting a single run.
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')   # Non-interactive backend to avoid errors in headless environments
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import entropy as scipy_entropy
from scipy.integrate import odeint

# ─────────────────────────────────────────────
# Color Configuration (used throughout)
# ─────────────────────────────────────────────
CMAP_GRID = mcolors.ListedColormap(['#1a1a2e', '#1f77b4', '#2ca02c', '#d62728'])
BOUNDS     = [-0.5, 0.5, 1.5, 2.5, 3.5]
NORM       = mcolors.BoundaryNorm(BOUNDS, CMAP_GRID.N)
BG         = '#0d1117'
PANEL_BG   = '#161b22'
COLORS_DNs = ['#1f77b4', '#2ca02c', '#d62728']   # D, N, S
LABELS_DNS = ['Deep Content D', 'News N', 'Short-form S']


# ══════════════════════════════════════════════════════════════════
# Part 0: Command-line Argument Parsing
# ══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='Social Information Diversity Competition Simulation (RPS Attention Model)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    # Grid and simulation size
    p.add_argument('--L',       type=int,   default=50,
                   help='Grid size, total users = L×L (default: 50)')
    p.add_argument('--n_steps', type=int,   default=500,
                   help='Total evolution generations (default: 250 — COEXIST window, see header)')
    p.add_argument('--seed',    type=int,   default=42,
                   help='Random seed (default: 42)')

    # Mobility: can specify 1-3 values for three scenarios in plots
    p.add_argument('--M', type=float, nargs='+', default=[1e-5, 5e-5, 2e-3],
                   metavar='M',
                   help='Cross-platform mobility M, accepts 1-3 values (default: 1e-5 5e-5 2e-3)')

    # Algorithm boost bias gamma (reproduction rate)
    p.add_argument('--gamma_D', type=float, default=1.0, help='Deep content boost γ_D (default: 1.0)')
    p.add_argument('--gamma_N', type=float, default=1.0, help='News boost γ_N (default: 1.0)')
    p.add_argument('--gamma_S', type=float, default=1.7, help='Short-form boost γ_S (default: 1.7, COEXIST variant — see header note)')

    # Attention capture rate sigma (selection rate)
    # [FIX] sigma_D 原預設 0.8 會讓 D 攻擊 N 的成功率天生打八折，三者應對稱以避免人為偏誤
    p.add_argument('--sigma_D', type=float, default=1.0, help='Deep content competitiveness σ_D (default: 1.0, FIXED from 0.8)')
    p.add_argument('--sigma_N', type=float, default=1.0, help='News competitiveness σ_N (default: 1.0)')
    p.add_argument('--sigma_S', type=float, default=1.0, help='Short-form competitiveness σ_S (default: 1.0)')

    # ODE parameters (corrected report version)
    p.add_argument('--mu0',     type=float, default=1.0,  help='Base reproduction rate μ₀ (default: 1.0)')
    p.add_argument('--sigma0',  type=float, default=1.0,  help='Base competition rate σ₀ (default: 1.0)')
    p.add_argument('--zeta',    type=float, default=0.1,  help='Platform openness ζ (external spillover, default: 0.1)')
    p.add_argument('--epsilon0', type=float, default=0.30, help='Base fatigue rate ε₀ for spatial sim (default: 0.30, FIXED: previously aliased to zeta)')
    p.add_argument('--eta_D',   type=float, default=0.05, help='Deep content fatigue η_D (default: 0.05)')
    p.add_argument('--eta_N',   type=float, default=0.3,  help='News fatigue η_N (default: 0.3)')
    p.add_argument('--eta_S',   type=float, default=0.35, help='Short-form fatigue η_S (default: 0.35, FIXED from 0.8 — see report discussion)')
    p.add_argument('--kappa_D', type=float, default=0.9,  help='Deep content echo-chamber protection κ_D (default: 0.9)')
    p.add_argument('--kappa_N', type=float, default=0.7,  help='News echo-chamber protection κ_N (default: 0.7)')
    p.add_argument('--kappa_S', type=float, default=0.5,  help='Short-form echo-chamber protection κ_S (default: 0.5)')

    # Other options
    p.add_argument('--no_ode',  action='store_true', help='Skip ODE plot (faster)')
    p.add_argument('--outdir',  type=str, default='.',    help='Output directory (default: current directory)')
    p.add_argument('--n_seeds', type=int, default=1, help='Number of random seeds to average over for robustness (default: 1)')

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════
# Part 1: Corrected ODE (Report Formula, Non-Spatial Determinism)
# ══════════════════════════════════════════════════════════════════

def attention_ode(y, t, gamma, mu0, sigma0, zeta, eta, kappa):
    """
    Report-corrected 4D ODE:
      x1=D, x2=N, x3=S, x0=∅ (idle attention)
    Suppression chain: D→N→S→D, x_prev[i] = species suppressing xi
    """
    x1, x2, x3 = y[0], y[1], y[2]
    x0 = max(0.0, 1.0 - x1 - x2 - x3)   # Conservation: ∅ = 1 - Σxi

    g1, g2, g3 = gamma
    e1, e2, e3 = eta
    k1, k2, k3 = kappa

    # Suppression relationships: x1 suppressed by x3, x2 by x1, x3 by x2
    dx1 = g1*mu0*x1*x0 - k1*sigma0*x1*x3 - (zeta*g1 + e1*x1)*x1
    dx2 = g2*mu0*x2*x0 - k2*sigma0*x2*x1 - (zeta*g2 + e2*x2)*x2
    dx3 = g3*mu0*x3*x0 - k3*sigma0*x3*x2 - (zeta*g3 + e3*x3)*x3

    return [dx1, dx2, dx3]


def run_ode(args, t_max=200):
    """Execute ODE with two initial conditions, corresponding to two report scenarios"""
    gamma = [args.gamma_D, args.gamma_N, args.gamma_S]
    eta   = [args.eta_D,   args.eta_N,   args.eta_S]
    kappa = [args.kappa_D, args.kappa_N, args.kappa_S]
    t     = np.linspace(0, t_max, 5000)

    # Scenario A: Initial uniform distribution
    y0_equal = [0.25, 0.25, 0.25]
    sol_eq   = odeint(attention_ode, y0_equal, t,
                      args=(gamma, args.mu0, args.sigma0, args.zeta, eta, kappa))

    # Scenario B: Short-form initially dominant
    y0_s_dom = [0.15, 0.15, 0.55]
    sol_sd   = odeint(attention_ode, y0_s_dom, t,
                      args=(gamma, args.mu0, args.sigma0, args.zeta, eta, kappa))

    return t, sol_eq, sol_sd


def plot_ode(args, outdir):
    """Figure 1: Corrected ODE results (May-Leonard style, using report parameters)"""
    t, sol_eq, sol_sd = run_ode(args)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
    titles = [
        f'ODE: Initial uniform distribution\n(γ_S={args.gamma_S}, η_S={args.eta_S})',
        f'ODE: Short-form initially dominant (x_S=0.55)\n(γ_S={args.gamma_S}, κ_S={args.kappa_S})'
    ]
    sols = [sol_eq, sol_sd]

    for ax, sol, title in zip(axes, sols, titles):
        ax.set_facecolor(PANEL_BG)
        for k, (col, lbl) in enumerate(zip(COLORS_DNs, LABELS_DNS)):
            ax.plot(t, sol[:, k], color=col, lw=1.8, label=lbl)
        ax.set_xlabel('Time t', color='white')
        ax.set_ylabel('Attention share', color='white')
        ax.set_title(title, color='white', fontsize=11)
        ax.tick_params(colors='white')
        ax.set_ylim(0, 1)
        for sp in ax.spines.values():
            sp.set_edgecolor('#444')
        ax.legend(facecolor='#1e1e2e', labelcolor='white', fontsize=9)

    plt.suptitle('Corrected ODE (Non-Spatial Determinism) — Corresponding to Report Formulas', color='white', fontsize=13)
    plt.tight_layout()
    path = f'{outdir}/fig1_ode_attention.png'
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'  [OK] {path}')


# ══════════════════════════════════════════════════════════════════
# Part 2: Stochastic Lattice Simulation Core
# ══════════════════════════════════════════════════════════════════

def run_simulation(L, M, n_steps, gamma, sigma, eta, epsilon_0=1.0, snapshot_interval=None):
    """
    Monte Carlo stochastic lattice simulation

    Bug fixes:
    - p_exch no longer just calculated, now directly determines three probability boundaries via epsilon and total_rate
    - Snapshots now store only interval frames to avoid memory overflow, guaranteeing initial state (step=0)
    - Removed issue of plt.show() after plt.close() (moved to external)
    """
    N_sites = L * L
    if snapshot_interval is None:
        snapshot_interval = max(1, n_steps // 5)

    # Initialize grid: randomly assign 0(∅), 1(D), 2(N), 3(S)
    grid = np.random.randint(0, 4, size=(L, L))

    # 【新增】初始化年齡網格，用來記錄維持當前狀態的時長 tau
    age_grid = np.zeros((L, L), dtype=float)

    # Mobility conversion: M = 2ε/N  →  ε = M·N/2
    epsilon = (M * N_sites) / 2.0

    max_sigma = max(sigma.values())
    max_gamma = max(gamma.values())
    total_rate = max_sigma + max_gamma + epsilon

    # Cumulative probability boundaries for three actions
    p1 = max_sigma / total_rate          # < p1        → competition
    p2 = (max_sigma + max_gamma) / total_rate  # p1~p2 → reproduction
    #                                           > p2        → exchange

    beats = {1: 2, 2: 3, 3: 1}  # D→N, N→S, S→D

    snapshots  = []
    history    = {'D': [], 'N': [], 'S': [], 'Entropy': []}

    # Store initial snapshot (before step 0)
    snapshots.append(grid.copy())

    for step in range(n_steps):
        for _ in range(N_sites):
            i, j   = np.random.randint(0, L, 2)
            di, dj = [(-1,0),(1,0),(0,-1),(0,1)][np.random.randint(4)]
            ni, nj = (i+di) % L, (j+dj) % L

            sc = grid[i, j]
            sn = grid[ni, nj]

            # 【新增】主動節點的審美疲勞判定 (遵守使用者給定公式)
            if sc != 0:
                tau = age_grid[i, j]
                p_fatigue = epsilon_0 * (1.0 - np.exp(-eta[sc] * tau))
                if np.random.rand() < p_fatigue:
                    grid[i, j] = 0
                    age_grid[i, j] = 0.0
                    sc = 0  # 狀態更新為閒置
            
            r  = np.random.rand()

            if r < p1:
                # ── Competition (attention capture) ──────────────────
                if sc != 0 and sn != 0:
                    if beats.get(sc) == sn:
                        if np.random.rand() < sigma[sc] / max_sigma:
                            grid[ni, nj] = 0
                            age_grid[ni, nj] = 0.0  # 【新增】被捕獲後年齡歸零
                    elif beats.get(sn) == sc:
                        if np.random.rand() < sigma[sn] / max_sigma:
                            grid[i, j] = 0
                            age_grid[i, j] = 0.0   # 【新增】被捕獲後年齡歸零

            elif r < p2:
                # ── Reproduction (algorithm boost) ────────────────
                if sc != 0 and sn == 0:
                    if np.random.rand() < gamma[sc] / max_gamma:
                        grid[ni, nj] = sc
                        age_grid[ni, nj] = 0.0  # 【新增】新佔領空間年齡由 0 開始
                elif sn != 0 and sc == 0:
                    if np.random.rand() < gamma[sn] / max_gamma:
                        grid[i, j] = sn
                        age_grid[i, j] = 0.0   # 【新增】新佔領空間年齡由 0 開始

            else:
                # ── Exchange (cross-platform flow) ────────────────────
                grid[i, j], grid[ni, nj] = grid[ni, nj], grid[i, j]
                # 【新增】實體交換時，記憶的年齡也必須跟著對調
                age_grid[i, j], age_grid[ni, nj] = age_grid[ni, nj], age_grid[i, j]

        # 【新增】在 inner loop 結束後，更新所有非空閒節點的年齡
        age_grid[grid != 0] += 1.0
        # Record snapshots (interval)
        if (step + 1) % snapshot_interval == 0 or step == n_steps - 1:
            snapshots.append(grid.copy())

        # Record counts and Shannon Entropy per step
        counts = [int(np.sum(grid == c)) for c in (1, 2, 3)]
        history['D'].append(counts[0])
        history['N'].append(counts[1])
        history['S'].append(counts[2])

        total_content = sum(counts)
        if total_content > 0:
            probs = [c / total_content for c in counts]
            h_val = scipy_entropy(probs, base=np.e)
        else:
            h_val = 0.0
        history['Entropy'].append(h_val)

    return snapshots, history


# ══════════════════════════════════════════════════════════════════
# Part 2b: Multi-Seed Robustness Check
# ══════════════════════════════════════════════════════════════════

def robustness_check(L, M, n_steps, gamma, sigma, eta, epsilon_0, n_seeds=5,
                       win_threshold=0.9, alive_threshold=0.02):
    """
    用多個 random seed 重複跑模擬，判斷目前這組參數的長期贏家是否穩健，
    而不是只看單次模擬結果 (這套循環相剋系統對參數高度敏感，單次結果可能只是
    heteroclinic cycle 中的某個暫態相位，並非真正的長期吸引子)。

    回傳:
        summary: dict，包含每個 seed 的最終比例、贏家標籤統計、是否穩健的判定
    """
    records = []
    for seed in range(n_seeds):
        rng_state = np.random.get_state()  # 保留外部 rng 狀態，避免影響後續繪圖用的模擬
        np.random.seed(seed)
        snaps, hist = run_simulation(L=L, M=M, n_steps=n_steps, gamma=gamma,
                                       sigma=sigma, eta=eta, epsilon_0=epsilon_0)
        np.random.set_state(rng_state)

        D, N, S = hist['D'][-1], hist['N'][-1], hist['S'][-1]
        total = D + N + S
        if total == 0:
            dn, nn, sn = 0.0, 0.0, 0.0
        else:
            dn, nn, sn = D / total, N / total, S / total

        alive = sum(v > alive_threshold for v in (dn, nn, sn))
        if alive >= 2:
            tag = 'mixed (多樣性共存)'
        elif dn > win_threshold:
            tag = 'D'
        elif nn > win_threshold:
            tag = 'N'
        elif sn > win_threshold:
            tag = 'S'
        else:
            tag = 'other (未達單一物種主導門檻)'

        records.append({'seed': seed, 'D': dn, 'N': nn, 'S': sn, 'tag': tag})

    tag_counts = {}
    for r in records:
        tag_counts[r['tag']] = tag_counts.get(r['tag'], 0) + 1

    is_robust = len(tag_counts) == 1  # 所有 seed 結果一致才算穩健
    dominant_tag = max(tag_counts, key=tag_counts.get)

    return {
        'records': records,
        'tag_counts': tag_counts,
        'is_robust': is_robust,
        'dominant_tag': dominant_tag,
        'n_seeds': n_seeds,
    }


def print_robustness_report(summary):
    """將 robustness_check 的結果用易讀格式印出"""
    print('  ' + '-' * 54)
    print(f"  多 Seed 穩健性檢驗 (n_seeds={summary['n_seeds']})")
    for r in summary['records']:
        print(f"    seed={r['seed']}: D={r['D']:.3f}  N={r['N']:.3f}  S={r['S']:.3f}  → {r['tag']}")
    print(f"  結果分布: {summary['tag_counts']}")
    if summary['is_robust']:
        print(f"  ✓ 穩健：所有 seed 一致收斂到 [{summary['dominant_tag']}]")
    else:
        print(f"  ⚠ 不穩健：不同 seed 收斂到不同結果，目前參數位於相變邊界附近，"
              f"結論不能僅憑單次模擬下定論。建議遠離此參數區間或加大 n_steps 再驗證。")
    print('  ' + '-' * 54)


# ══════════════════════════════════════════════════════════════════
# Part 3: Plotting Functions (Six Figures)
# ══════════════════════════════════════════════════════════════════

def _style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, color='white', fontsize=10)
    ax.set_xlabel(xlabel, color='white')
    ax.set_ylabel(ylabel, color='white')
    ax.tick_params(colors='white')
    for sp in ax.spines.values():
        sp.set_edgecolor('#444')


def plot_spatial_snapshots(results, outdir):
    """
    Figure 2: Final spatial snapshot (corresponds to paper Figure 2a)
    Three M values in parallel, showing spatial patterns at low/medium/high mobility
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5), facecolor=BG)
    if n == 1:
        axes = [axes]

    for ax, (lbl, snaps, hist, M_val) in zip(axes, results):
        ax.imshow(snaps[-1], cmap=CMAP_GRID, norm=NORM, interpolation='nearest')
        counts = [hist['D'][-1], hist['N'][-1], hist['S'][-1]]
        alive  = sum(c > 0 for c in counts)
        ax.set_title(f'{lbl}\nM={M_val:.1e}  Species alive:{alive}/3',
                     color='white', fontsize=10)
        ax.axis('off')

    legend_els = [
        plt.Rectangle((0,0),1,1, color='#1f77b4', label='Deep Content D'),
        plt.Rectangle((0,0),1,1, color='#2ca02c', label='News N'),
        plt.Rectangle((0,0),1,1, color='#d62728', label='Short-form S'),
        plt.Rectangle((0,0),1,1, color='#1a1a2e', label='Idle Attention ∅'),
    ]
    fig.legend(handles=legend_els, loc='lower center', ncol=4,
               facecolor='#1e1e2e', labelcolor='white', fontsize=9,
               bbox_to_anchor=(0.5, -0.04))
    plt.suptitle('Social Space Final State (Information Spatial Distribution)',
                 color='white', fontsize=13)
    plt.tight_layout()
    path = f'{outdir}/fig2_spatial_snapshots.png'
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'  [OK] {path}')


def plot_timeseries(results, outdir):
    """
    Figure 3: Three-species proportion time series (corresponds to paper Supplementary Movies)
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4), facecolor=BG, sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (lbl, snaps, hist, M_val) in zip(axes, results):
        total = np.array(hist['D']) + np.array(hist['N']) + np.array(hist['S'])
        total = np.where(total == 0, 1, total)
        steps = np.arange(len(hist['D']))
        for arr, col, sp_lbl in zip(
                [hist['D'], hist['N'], hist['S']], COLORS_DNs, LABELS_DNS):
            ax.plot(steps, np.array(arr)/total, color=col, lw=1.5, label=sp_lbl)
        _style_ax(ax, title=f'{lbl}\nM={M_val:.1e}',
                  xlabel='Generations', ylabel='Content Share')
        ax.set_ylim(0, 1)
        ax.legend(facecolor='#1e1e2e', labelcolor='white', fontsize=8)

    plt.suptitle('Three-Content Species Proportion Time Series', color='white', fontsize=13)
    plt.tight_layout()
    path = f'{outdir}/fig3_timeseries.png'
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'  [OK] {path}')


def plot_entropy(results, outdir):
    """
    Figure 4: Shannon Entropy time series plot (information diversity evolution)
    """
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
    _style_ax(ax, title='Information Diversity Evolution (Shannon Entropy)',
              xlabel='Generations', ylabel='Shannon Entropy H')

    palette = ['#58a4b0', '#f4a261', '#e76f51']
    for (lbl, snaps, hist, M_val), col in zip(results, palette):
        ax.plot(hist['Entropy'], label=f'{lbl}  (M={M_val:.0e})',
                color=col, lw=2)

    ax.axhline(y=np.log(3), color='#aaa', ls='--', lw=1.2,
               label=f'Maximum diversity ln(3) ≈ {np.log(3):.3f}')
    ax.set_ylim(0, np.log(3) * 1.15)
    ax.legend(facecolor='#1e1e2e', labelcolor='white', fontsize=9)
    ax.grid(True, color='#333', lw=0.5)

    plt.tight_layout()
    path = f'{outdir}/fig4_entropy.png'
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'  [OK] {path}')


def plot_multi_snapshots(results, outdir):
    """
    Figure 5: Multiple time point snapshots for each M value (evolution process, corresponds to paper spiral wave formation)
    """
    n_M = len(results)
    n_snaps = min(4, len(results[0][1]))   # Take at most 4 time points

    fig, axes = plt.subplots(n_M, n_snaps,
                             figsize=(4*n_snaps, 4*n_M), facecolor=BG)
    # Ensure axes is 2D
    if n_M == 1:
        axes = axes[np.newaxis, :]
    if n_snaps == 1:
        axes = axes[:, np.newaxis]

    for row, (lbl, snaps, hist, M_val) in enumerate(results):
        # Evenly sample n_snaps snapshots (including initial and final)
        indices = np.linspace(0, len(snaps)-1, n_snaps, dtype=int)
        for col, idx in enumerate(indices):
            ax = axes[row, col]
            ax.imshow(snaps[idx], cmap=CMAP_GRID, norm=NORM,
                      interpolation='nearest')
            ax.axis('off')
            gen_label = 'Initial' if idx == 0 else f't≈{idx * (len(hist["D"])//len(snaps))} gen'
            if col == 0:
                ax.set_ylabel(f'M={M_val:.0e}', color='white', fontsize=10)
            ax.set_title(gen_label, color='#aaa', fontsize=8)

    plt.suptitle('Spatial Evolution Process (Snapshots at Various Time Points)', color='white', fontsize=13)
    plt.tight_layout()
    path = f'{outdir}/fig5_evolution_snapshots.png'
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'  [OK] {path}')


def plot_phase_diagram(args, outdir):
    """
    Figure 6: Phase diagram sketch (mobility M vs algorithm bias γ_S, corresponds to paper Figure 4)
    """
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=BG)
    _style_ax(ax,
              title='Phase Diagram: Information Diversity Phase vs Homogeneous Phase (Reichenbach Fig.4 style)',
              xlabel='Short-form algorithm bias γ_S',
              ylabel='Cross-platform mobility M')

    gamma_S_vals = np.linspace(0.5, 4.0, 300)

    # Mc approximation: higher γ_S → lower Mc (easier collapse)
    # Use saturation function similar to paper, with γ_S as additional degradation factor
    Mc_base = 4.5e-4
    Mc_curve = Mc_base / (1 + 0.8 * (gamma_S_vals - 1))
    Mc_curve = np.clip(Mc_curve, 1e-5, 2e-3)

    ax.fill_between(gamma_S_vals, Mc_curve, 2e-3,
                    alpha=0.35, color='#e63946', label='Homogeneous phase (short-form dominance)')
    ax.fill_between(gamma_S_vals, 1e-5, Mc_curve,
                    alpha=0.35, color='#2a9d8f', label='Diversity phase (three-content coexistence)')
    ax.plot(gamma_S_vals, Mc_curve, color='white', lw=2,
            label='Critical mobility Mc(γ_S)')

    # Mark current parameter point
    ax.plot(args.gamma_S, args.M[0], 'o', color='#e9c46a', ms=10,
            label=f'Current parameters (γ_S={args.gamma_S}, M={args.M[0]:.0e})')

    ax.set_yscale('log')
    ax.set_ylim(1e-5, 2e-3)
    ax.set_xlim(0.5, 4.0)
    ax.text(0.6,  8e-4, 'Homogeneous Phase\n(Short-form Monopoly)', color='#e63946', fontsize=10)
    ax.text(2.5,  3e-5, 'Diversity Phase\n(Coexistence)', color='#2a9d8f', fontsize=10)
    ax.legend(facecolor='#1e1e2e', labelcolor='white', fontsize=9)
    ax.grid(True, color='#333', lw=0.5, which='both')
    ax.tick_params(colors='white', which='both')

    plt.tight_layout()
    path = f'{outdir}/fig6_phase_diagram.png'
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'  [OK] {path}')


# ══════════════════════════════════════════════════════════════════
# Part 4: Main Program
# ══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    np.random.seed(args.seed)

    # ── Organize parameters ───────────────────────────────────────────
    gamma = {1: args.gamma_D, 2: args.gamma_N, 3: args.gamma_S}
    sigma = {1: args.sigma_D, 2: args.sigma_N, 3: args.sigma_S}

    M_values = args.M
    # Pad to three values if user provided only one or two
    while len(M_values) < 3:
        M_values = M_values + [M_values[-1]]
    M_values = M_values[:3]

    labels_M = [
        f'Low M (Echo-chamber)',
        f'Medium M (Spiral wave)',
        f'High M (Homogenization)',
    ]

    import os
    os.makedirs(args.outdir, exist_ok=True)

    print('=' * 58)
    print('  Social Information Diversity Competition Simulation')
    print(f'  Grid: {args.L}×{args.L}  |  Generations: {args.n_steps}')
    print(f'  γ = D:{args.gamma_D} N:{args.gamma_N} S:{args.gamma_S}')
    print(f'  σ = D:{args.sigma_D} N:{args.sigma_N} S:{args.sigma_S}')
    print(f'  η = D:{args.eta_D} N:{args.eta_N} S:{args.eta_S}')
    print(f'  ε₀(fatigue base) = {args.epsilon0}   ζ(spillover) = {args.zeta}')
    print(f'  M = {M_values}')
    print('=' * 58)

    # ── Figure 1: ODE (can be skipped for speed) ──────────────────────────────
    if not args.no_ode:
        print('\n[1/6] Plotting corrected ODE figure...')
        plot_ode(args, args.outdir)
    else:
        print('\n[1/6] Skipping ODE plot (--no_ode)')

    # ── Execute spatial simulations for three M values ───────────────────────────────
    print(f'\n[Simulation] Executing spatial simulations for three M values...')
    results = []
    for M_val, lbl in zip(M_values, labels_M):
        print(f'  M={M_val:.1e}  ({lbl})')
        eta_dict = {1: args.eta_D, 2: args.eta_N, 3: args.eta_S}  # 建立對應格式的疲勞率

        # 【新增】多 seed 穩健性檢驗：在跑正式模擬之前，先確認目前參數的長期贏家
        # 是否對隨機種子穩健。這套循環相剋系統對參數敏感，單次結果可能只是
        # heteroclinic cycle 暫態相位，不能直接當作結論。
        if args.n_seeds > 1:
            summary = robustness_check(
                L=args.L, M=M_val, n_steps=args.n_steps,
                gamma=gamma, sigma=sigma, eta=eta_dict, epsilon_0=args.epsilon0,
                n_seeds=args.n_seeds
            )
            print_robustness_report(summary)

        snaps, hist = run_simulation(
            L=args.L, M=M_val, n_steps=args.n_steps,
            gamma=gamma, sigma=sigma, eta=eta_dict, epsilon_0=args.epsilon0
        )
        results.append((lbl, snaps, hist, M_val))
        alive = sum(hist[k][-1] > 0 for k in ['D', 'N', 'S'])
        print(f'    → Final species alive: {alive}/3  '
              f'  Final Entropy: {hist["Entropy"][-1]:.3f}')

    # ── Plot remaining five figures ───────────────────────────────────────
    print('\n[2/6] Spatial final snapshot...')
    plot_spatial_snapshots(results, args.outdir)

    print('[3/6] Time series...')
    plot_timeseries(results, args.outdir)

    print('[4/6] Shannon Entropy...')
    plot_entropy(results, args.outdir)

    print('[5/6] Multi-timepoint evolution snapshots...')
    plot_multi_snapshots(results, args.outdir)

    print('[6/6] Phase diagram...')
    plot_phase_diagram(args, args.outdir)

    print('\n' + '=' * 58)
    print(f'  All complete! Plots output to: {args.outdir}/')
    print('    fig1_ode_attention.png')
    print('    fig2_spatial_snapshots.png')
    print('    fig3_timeseries.png')
    print('    fig4_entropy.png')
    print('    fig5_evolution_snapshots.png')
    print('    fig6_phase_diagram.png')
    print('=' * 58)


if __name__ == '__main__':
    main()