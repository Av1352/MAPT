import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data: run2 (1M steps) + run4 (3M steps resumed from checkpoint) ──────────

# Run2: first 1M steps from scratch
run2_steps = [
    1000, 21000, 41000, 61000, 81000, 101000, 121000, 141000, 161000, 181000,
    201000, 221000, 241000, 261000, 281000, 301000, 321000, 341000, 361000, 381000,
    401000, 421000, 441000, 461000, 481000, 501000, 521000, 541000, 561000, 581000,
    601000, 621000, 641000, 661000, 681000, 701000, 721000, 741000, 761000, 781000,
    801000, 821000, 841000, 861000, 881000, 901000, 921000, 941000, 961000, 981000,
]
run2_rewards = [
    -0.24, -4.03, 30.03, 477.36, 499.87, 711.88, 544.58, 457.32, 526.46, 496.12,
    543.49, 614.35, 708.47, 840.25, 959.32, 1004.13, 1118.30, 1092.35, 1039.37, 981.28,
    1078.47, 1051.67, 1197.32, 1239.71, 1240.58, 1258.09, 1393.10, 1318.99, 1373.78, 1346.18,
    1441.39, 1461.13, 1490.46, 1544.00, 1580.72, 1653.26, 1601.21, 1615.92, 1708.02, 1714.16,
    1700.37, 1672.29, 1773.59, 1717.72, 1736.58, 1804.03, 1704.76, 1809.63, 1809.69, 1872.97,
]

# Run4: resumed from checkpoint, internal steps 1k-2981k, offset by 981k for total
run4_internal = [
    1000, 21000, 41000, 61000, 81000, 101000, 121000, 141000, 161000, 181000,
    201000, 221000, 241000, 261000, 281000, 301000, 321000, 341000, 361000, 381000,
    401000, 421000, 441000, 461000, 481000, 501000, 521000, 541000, 561000, 581000,
    601000, 621000, 641000, 661000, 681000, 701000, 721000, 741000, 761000, 781000,
    801000, 821000, 841000, 861000, 881000, 901000, 921000, 941000, 961000, 981000,
    1001000, 1021000, 1041000, 1061000, 1081000, 1101000, 1121000, 1141000, 1161000, 1181000,
    1201000, 1221000, 1241000, 1261000, 1281000, 1301000, 1321000, 1341000, 1361000, 1381000,
    1401000, 1421000, 1441000, 1461000, 1481000, 1501000, 1521000, 1541000, 1561000, 1581000,
    1601000, 1621000, 1641000, 1661000, 1681000, 1701000, 1721000, 1741000, 1761000, 1781000,
    1801000, 1821000, 1841000, 1861000, 1881000, 1901000, 1921000, 1941000, 1961000, 1981000,
    2001000, 2021000, 2041000, 2061000, 2081000, 2101000, 2121000, 2141000, 2161000, 2181000,
    2201000, 2221000, 2241000, 2261000, 2281000, 2301000, 2321000, 2341000, 2361000, 2381000,
    2401000, 2421000, 2441000, 2461000, 2481000, 2501000, 2521000, 2541000, 2561000, 2581000,
    2601000, 2621000, 2641000, 2661000, 2681000, 2701000, 2721000, 2741000, 2761000, 2781000,
    2801000, 2821000, 2841000, 2861000, 2881000, 2901000, 2921000, 2941000, 2961000,
]
run4_steps = [s + 981000 for s in run4_internal]
run4_rewards = [
    1949.80, 1982.58, 1954.97, 1933.75, 2137.08, 2120.51, 1905.67, 1908.98, 2069.35, 2083.12,
    2067.74, 2174.61, 2122.16, 2148.54, 2092.09, 2245.92, 2560.20, 2538.61, 2657.81, 2526.25,
    2256.43, 2167.12, 2272.35, 2069.39, 2198.27, 2296.13, 2358.11, 2515.20, 2377.51, 2494.57,
    2374.48, 2482.43, 2562.18, 2537.47, 2475.33, 2631.20, 2725.35, 2671.36, 2670.47, 2576.89,
    2644.61, 2473.19, 2818.15, 2850.12, 2750.77, 2836.82, 2951.83, 2883.14, 3017.20, 2925.83,
    2848.32, 2759.18, 2805.60, 2795.49, 2887.88, 2883.22, 2854.14, 2973.74, 2885.53, 2778.68,
    2946.67, 3103.65, 3000.86, 2841.61, 2853.11, 2840.83, 2957.05, 2891.17, 2961.74, 3060.18,
    3106.40, 3291.80, 3009.67, 2951.92, 2976.39, 3075.83, 3093.42, 3206.14, 3166.68, 3125.57,
    3037.21, 3012.65, 3064.98, 3127.95, 2976.73, 3096.63, 3171.71, 2962.54, 3050.90, 2870.14,
    2829.99, 2936.28, 3127.66, 2988.49, 3041.67, 2881.06, 2921.39, 2926.05, 3138.62, 3007.68,
    2930.53, 3138.69, 3047.13, 3223.26, 2891.49, 2899.68, 3082.31, 2996.48, 3039.15, 3027.38,
    3158.97, 3040.08, 3078.82, 3066.33, 3186.89, 3106.11, 3027.73, 2771.96, 2998.83, 3158.88,
    3098.07, 3057.53, 3254.63, 3262.48, 3182.17, 3149.53, 3209.52, 3286.44, 3148.25, 2991.65,
    3038.56, 3020.23, 3148.48, 3228.13, 3350.11, 3249.02, 3270.62, 3345.81, 3219.84, 3119.69,
    3274.87, 3189.82, 3051.99, 3253.54, 3214.86, 3298.05, 3272.47, 3291.57, 3183.98,
]

# combine
timesteps = run2_steps + run4_steps
rewards = run2_rewards + run4_rewards

# convert to millions
timesteps_m = [t / 1e6 for t in timesteps]

# smooth with rolling average
window = 10
rewards_smooth = np.convolve(rewards, np.ones(window)/window, mode='valid')
timesteps_smooth = timesteps_m[window-1:]

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

# raw rewards (faded)
ax.plot(timesteps_m, rewards, color='steelblue', alpha=0.25, linewidth=1.0, label='_nolegend_')

# smoothed curve
ax.plot(timesteps_smooth, rewards_smooth, color='steelblue', linewidth=2.5, label='MAPT (Ours, NMR) — Self-generated data')

# paper baseline
paper_mean = 2423
paper_std = 128
ax.axhline(y=paper_mean, color='tomato', linewidth=2.0, linestyle='--', label=f'MAPT Paper Result ({paper_mean} ± {paper_std})')
ax.axhspan(paper_mean - paper_std, paper_mean + paper_std, alpha=0.12, color='tomato')

# peak annotation
peak_idx = rewards.index(max(rewards))
ax.annotate(f'Peak: {max(rewards):.0f}',
            xy=(timesteps_m[peak_idx], rewards[peak_idx]),
            xytext=(timesteps_m[peak_idx] - 0.4, rewards[peak_idx] + 100),
            fontsize=10, color='steelblue',
            arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.5))

# paper exceeded marker
exceed_idx = next(i for i, r in enumerate(rewards) if r >= paper_mean)
ax.axvline(x=timesteps_m[exceed_idx], color='green', linewidth=1.5, linestyle=':', alpha=0.7)
ax.annotate(f'Exceeds paper\n@ {timesteps_m[exceed_idx]:.2f}M steps',
            xy=(timesteps_m[exceed_idx], paper_mean),
            xytext=(timesteps_m[exceed_idx] + 0.1, paper_mean - 400),
            fontsize=9, color='green',
            arrowprops=dict(arrowstyle='->', color='green', lw=1.2))

# formatting
ax.set_xlabel('Environment Steps (Millions)', fontsize=13)
ax.set_ylabel('Eval Episode Reward', fontsize=13)
ax.set_title('MAPT Sanity Check — HalfCheetah 6×1\nMAMuJoCo Preference-Based Policy Learning', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xlim(left=0)
ax.set_ylim(bottom=-100)
ax.tick_params(labelsize=11)

# note
ax.text(0.98, 0.05,
        'Note: Self-generated preference data\n(random + biased policies)\nPaper used expert vs. rookie policies',
        transform=ax.transAxes, fontsize=8.5, color='gray',
        ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='lightgray'))

plt.tight_layout()
plt.savefig('./mapt_learning_curve.pdf', dpi=300, bbox_inches='tight')
plt.savefig('./mapt_learning_curve.png', dpi=300, bbox_inches='tight')
print("Saved!")
