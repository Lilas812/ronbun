# -*- coding: utf-8 -*-
"""
本文 (random_walk_report.tex) が参照する図のうち，
式から一意に決まるものをすべて生成する。

生成する図:
  rw_barchart.png            対称RW (p=q=1/2) の n=1..4 の確率分布
  crw_barchart.pdf           CRW (a=d=2/3) の n=1..4 の確率分布
  normal_limit_distribution.pdf   標準正規分布 N(0,1)
  crw_limit_persistent.pdf   CRW極限 N(0,2)  (a=2/3, 持続性) と N(0,1) 比較
  crw_limit_antipersistent.pdf CRW極限 N(0,1/2) (a=1/3, 反転傾向) と N(0,1) 比較
  rw_transition.pdf          RW の遷移木 (n=0,1,2)
  crw_transition.pdf         CRW の遷移木 (n=0,1,2)
  linechart_renko_compare.pdf 価格系列 -> 平均練行足 への変換例

注: 遷移図のステップ確率・各分布はすべて本文の定義式に一致させてある。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import comb, sqrt, pi, exp


# ---------- 共通: CRW の厳密分布 (時間発展対称 a=d, b=c=1-a) ----------
def crw_distribution(a, n, phi=(0.5, 0.5)):
    b = 1.0 - a
    size = 2 * n + 1
    psiL = np.zeros(size)
    psiR = np.zeros(size)
    psiL[n] = phi[0]
    psiR[n] = phi[1]
    for _ in range(n):
        newL = np.zeros(size)
        newR = np.zeros(size)
        newL[:-1] += a * psiL[1:] + b * psiR[1:]
        newR[1:] += b * psiL[:-1] + a * psiR[:-1]
        psiL, psiR = newL, newR
    return np.arange(-n, n + 1), psiL + psiR


def normal_pdf(x, var):
    return np.exp(-x ** 2 / (2 * var)) / np.sqrt(2 * pi * var)


# ---------- 1. 対称RW バーチャート ----------
def fig_rw_barchart():
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    p = q = 0.5
    for ax, n in zip(axes, [1, 2, 3, 4]):
        xs = list(range(-n, n + 1, 2))
        probs = []
        for x in xs:
            ell = (n - x) // 2
            m = (n + x) // 2
            probs.append(comb(n, ell) * p ** ell * q ** m)
        ax.bar(xs, probs, width=0.6, color="#4C72B0", edgecolor="black", lw=0.5)
        ax.set_title("$n=%d$" % n)
        ax.set_xlabel("$x$")
        ax.set_xticks(xs)
        ax.set_ylim(0, 0.65)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("$P(S_n=x)$")
    fig.tight_layout()
    fig.savefig("rw_barchart.png", dpi=150)
    plt.close(fig)


# ---------- 2. CRW バーチャート ----------
def fig_crw_barchart():
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    a = 2.0 / 3.0
    for ax, n in zip(axes, [1, 2, 3, 4]):
        xs, probs = crw_distribution(a, n)
        mask = probs > 1e-12
        ax.bar(xs[mask], probs[mask], width=0.6, color="#C44E52",
               edgecolor="black", lw=0.5)
        ax.set_title("$n=%d$" % n)
        ax.set_xlabel("$x$")
        ax.set_xticks(xs[mask])
        ax.set_ylim(0, 0.65)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel(r"$P(\tilde{S}_n=x)$")
    fig.tight_layout()
    fig.savefig("crw_barchart.pdf")
    plt.close(fig)


# ---------- 3. 標準正規 ----------
def fig_normal_limit():
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.linspace(-4, 4, 800)
    ax.plot(x, normal_pdf(x, 1.0), color="#C44E52", lw=2,
            label="$N(0,1)$")
    ax.fill_between(x, normal_pdf(x, 1.0), color="#C44E52", alpha=0.15)
    ax.set_xlabel("$x$")
    ax.set_ylabel("density")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("normal_limit_distribution.pdf")
    plt.close(fig)


# ---------- 4,5. CRW 極限 (持続性 / 反転傾向) ----------
def fig_crw_limit(a, fname, label):
    var = a / (1 - a)
    lim = 4 * sqrt(max(var, 1.0))
    x = np.linspace(-lim, lim, 800)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(x, normal_pdf(x, 1.0), color="gray", lw=1.5, ls="--",
            label="ordinary RW: $N(0,1)$")
    ax.plot(x, normal_pdf(x, var), color="#4C72B0", lw=2.2,
            label=label)
    ax.fill_between(x, normal_pdf(x, var), color="#4C72B0", alpha=0.15)
    ax.set_xlabel("$x$")
    ax.set_ylabel("density")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)


# ---------- 6,7. 遷移木 ----------
def _draw_tree(ax, edges, nodes, title):
    # edges: list of ((t0,x0),(t1,x1), label)
    for (t0, x0), (t1, x1), lab in edges:
        ax.annotate("", xy=(t1, x1), xytext=(t0, x0),
                    arrowprops=dict(arrowstyle="-|>", color="#888", lw=1.3))
        ax.text((t0 + t1) / 2, (x0 + x1) / 2 + 0.16, lab,
                fontsize=11, color="#b33", ha="center",
                bbox=dict(boxstyle="round,pad=0.1", fc="white",
                          ec="none", alpha=0.85))
    for (t, x), lab, halign in nodes:
        ax.plot(t, x, "o", ms=9, color="#4C72B0", zorder=3)
        if halign == "below":
            ax.text(t, x - 0.28, lab, color="#1a1a1a", fontsize=10,
                    ha="center", va="top", zorder=4)
        else:
            dx = 0.10 if halign == "left" else -0.10
            ax.text(t + dx, x, lab, color="#1a1a1a", fontsize=10, ha=halign,
                    va="center", zorder=4)
    ax.set_title(title)
    ax.set_xlabel("step $n$")
    ax.set_ylabel("position $x$")
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_xlim(-0.5, 3.1)
    ax.set_ylim(-2.6, 2.6)
    ax.grid(alpha=0.25)


def fig_rw_transition():
    fig, ax = plt.subplots(figsize=(7.5, 5))
    edges = [
        ((0, 0), (1, -1), "$p$"), ((0, 0), (1, 1), "$q$"),
        ((1, -1), (2, -2), "$p$"), ((1, -1), (2, 0), "$q$"),
        ((1, 1), (2, 0), "$p$"), ((1, 1), (2, 2), "$q$"),
    ]
    nodes = [
        ((0, 0), "$P(S_0{=}0){=}1$", "below"),
        ((1, -1), "$p$", "right"), ((1, 1), "$q$", "right"),
        ((2, -2), "$p^2$", "left"), ((2, 0), "$2pq$", "left"),
        ((2, 2), "$q^2$", "left"),
    ]
    _draw_tree(ax, edges, nodes, "ordinary random walk")
    fig.tight_layout()
    fig.savefig("rw_transition.pdf")
    plt.close(fig)


def fig_crw_transition():
    fig, ax = plt.subplots(figsize=(7.5, 5))
    edges = [
        ((0, 0), (1, -1), "$a\\alpha+b\\beta$"),
        ((0, 0), (1, 1), "$c\\alpha+d\\beta$"),
        ((1, -1), (2, -2), "$a$"), ((1, -1), (2, 0), "$c$"),
        ((1, 1), (2, 0), "$b$"), ((1, 1), (2, 2), "$d$"),
    ]
    nodes = [
        ((0, 0), r"$\varphi=(\alpha,\beta)$", "below"),
        ((1, -1), r"L: $a\alpha{+}b\beta$", "right"),
        ((1, 1), r"R: $c\alpha{+}d\beta$", "right"),
        ((2, -2), r"$a(a\alpha{+}b\beta)$", "left"),
        ((2, 0), r"$b(c\alpha{+}d\beta){+}c(a\alpha{+}b\beta)$", "left"),
        ((2, 2), r"$d(c\alpha{+}d\beta)$", "left"),
    ]
    _draw_tree(ax, edges, nodes, "correlated random walk")
    fig.tight_layout()
    fig.savefig("crw_transition.pdf")
    plt.close(fig)


# ---------- 8. ラインチャート -> 練行足 ----------
def renko_signs(prices, B):
    o = prices[0]
    signs = []
    for Pt in prices[1:]:
        while Pt >= o + B:
            signs.append(+1)
            o += B
        while Pt <= o - B:
            signs.append(-1)
            o -= B
    return signs


def fig_linechart_renko():
    rng = np.random.default_rng(7)
    # 合成価格 (XAUUSDを模した水準) : ランダムウォーク + 微小トレンド
    steps = rng.normal(0.0, 1.0, 400)
    prices = 2000.0 + np.cumsum(steps) * 0.8
    B = 2.0
    signs = renko_signs(prices, B)
    x = np.concatenate([[0], np.cumsum(signs)])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(prices, color="#4C72B0", lw=1.0)
    axes[0].set_title("price series (line chart)")
    axes[0].set_xlabel("time $t$")
    axes[0].set_ylabel("price $P_t$")
    axes[0].grid(alpha=0.3)

    axes[1].step(range(len(x)), x, where="post", color="#C44E52", lw=1.2)
    axes[1].set_title("average renko ($B=%.0f$)" % B)
    axes[1].set_xlabel("renko index $n$")
    axes[1].set_ylabel("$x_n$ (box units)")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("linechart_renko_compare.pdf")
    plt.close(fig)


# ---------- 9. 有限ステップ -> 極限 への収束過程 ----------
def fig_crw_convergence():
    cases = [(2.0 / 3.0, "persistent: $a=2/3$ (limit $N(0,2)$)"),
             (1.0 / 3.0, "anti-persistent: $a=1/3$ (limit $N(0,1/2)$)")]
    ns = [2, 5, 20, 100]
    colors = plt.cm.viridis(np.linspace(0.15, 0.8, len(ns)))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, (a, title) in zip(axes, cases):
        var = a / (1 - a)
        for n, c in zip(ns, colors):
            xs, probs = crw_distribution(a, n)
            m = probs > 1e-12
            y = xs[m] / sqrt(n)
            dy = 2 / sqrt(n)
            ax.plot(y, probs[m] / dy, marker="o", ms=3, lw=1.0, color=c,
                    label="$n=%d$" % n)
        lim = 4 * sqrt(var)
        xx = np.linspace(-lim, lim, 800)
        ax.plot(xx, normal_pdf(xx, var), "r--", lw=2.2,
                label="limit $N(0,a/(1-a))$")
        ax.set_title(title)
        ax.set_xlabel("$S_n/\\sqrt{n}$")
        ax.set_ylabel("density")
        ax.set_xlim(-lim, lim)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig("crw_convergence.png", dpi=150)
    plt.close(fig)


def main():
    fig_rw_barchart()
    fig_crw_barchart()
    fig_normal_limit()
    fig_crw_limit(2.0 / 3.0, "crw_limit_persistent.pdf",
                  "CRW limit: $N(0,2)$  ($a=2/3$)")
    fig_crw_limit(1.0 / 3.0, "crw_limit_antipersistent.pdf",
                  "CRW limit: $N(0,1/2)$  ($a=1/3$)")
    fig_rw_transition()
    fig_crw_transition()
    fig_linechart_renko()
    fig_crw_convergence()
    print("all figures generated")


if __name__ == "__main__":
    main()
