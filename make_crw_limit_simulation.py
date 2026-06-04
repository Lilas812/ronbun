# -*- coding: utf-8 -*-
"""
時間発展対称な相関付きランダムウォーク (CRW) の有限ステップ分布を
ステップ数 n=1000 まで厳密に計算し，スケーリング S_n/sqrt(n) の
分布を極限正規分布 N(0, a/(1-a)) と重ねて比較する図を生成する。

確率ベクトルの時間発展:
    Psi_{n+1}(x) = P Psi_n(x+1) + Q Psi_n(x-1)
    P = [[a, 1-a],[0,0]],  Q = [[0,0],[1-a, a]]   (時間発展対称: a=d, b=c=1-a)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def crw_distribution(a, n, phi=(0.5, 0.5)):
    """時間発展対称 CRW の時刻 n における P(S_n = x) を厳密計算して返す。

    戻り値: (xs, probs)  xs は位置(パリティ n)，probs はその確率。
    """
    b = 1.0 - a
    # 位置 x in [-n, n] を index = x + n で持つ。各位置に2成分 (L, R)。
    size = 2 * n + 1
    psiL = np.zeros(size)
    psiR = np.zeros(size)
    psiL[n] = phi[0]   # 原点 x=0
    psiR[n] = phi[1]

    for _ in range(n):
        newL = np.zeros(size)
        newR = np.zeros(size)
        # Psi_{t+1}(x) = P Psi_t(x+1) + Q Psi_t(x-1)
        # P 行: L成分 = a*L(x+1) + b*R(x+1)
        # Q 行: R成分 = b*L(x-1) + a*R(x-1)
        # x+1 の寄与 -> 左シフト, x-1 の寄与 -> 右シフト
        newL[:-1] += a * psiL[1:] + b * psiR[1:]      # from x+1
        newR[1:]  += b * psiL[:-1] + a * psiR[:-1]     # from x-1
        psiL, psiR = newL, newR

    probs = psiL + psiR
    xs = np.arange(-n, n + 1)
    return xs, probs


def make_figure():
    n = 100
    cases = [
        (2.0 / 3.0, "persistent: $a=2/3,\\ \\sigma^2=a/(1-a)=2$"),
        (1.0 / 3.0, "anti-persistent: $a=1/3,\\ \\sigma^2=a/(1-a)=1/2$"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, (a, title) in zip(axes, cases):
        xs, probs = crw_distribution(a, n)
        # サポートは x のパリティが n のものだけ (間隔 2)
        mask = probs > 0
        xs_s = xs[mask]
        p_s = probs[mask]

        # スケーリング y = x / sqrt(n)。格子間隔 2 -> y 間隔 2/sqrt(n)。
        y = xs_s / np.sqrt(n)
        dy = 2.0 / np.sqrt(n)
        density = p_s / dy   # 確率質量を密度に変換

        sigma2 = a / (1.0 - a)
        yy = np.linspace(-4 * np.sqrt(sigma2), 4 * np.sqrt(sigma2), 800)
        normal = np.exp(-yy ** 2 / (2 * sigma2)) / np.sqrt(2 * np.pi * sigma2)

        ax.plot(y, density, color="#1f77b4", lw=1.2, marker="o", ms=2.5,
                label="$n=100$ exact distribution")
        ax.plot(yy, normal, color="#d62728", lw=1.8, ls="--",
                label="limit $N(0,\\,a/(1-a))$")
        ax.set_title(title)
        ax.set_xlabel("$S_n/\\sqrt{n}$")
        ax.set_ylabel("density")
        ax.set_xlim(-4 * np.sqrt(sigma2), 4 * np.sqrt(sigma2))
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("crw_limit_simulation.png", dpi=150)
    print("saved crw_limit_simulation.png")

    # 念のため正規化を確認
    for a, _ in cases:
        xs, probs = crw_distribution(a, n)
        print("a=%.3f  sum P=%.6f" % (a, probs.sum()))


if __name__ == "__main__":
    make_figure()
