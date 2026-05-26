from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties


ROOT = Path(__file__).resolve().parents[1]
FONT = ROOT / "assets/fonts/NotoSansCJKkr-Regular.otf"
PDF = ROOT / "transferable_rdm_project_note.pdf"
MD = ROOT / "transferable_rdm_project_note_current.md"
TEX = ROOT / "transferable_rdm_project_note.tex"


Block = dict[str, Any]


def npz_count(path: str) -> int:
    folder = ROOT / path
    return len(list(folder.glob("*.npz"))) if folder.exists() else 0


def load_selection_plan() -> list[dict[str, Any]]:
    path = ROOT / "qmugs_npz/qm9_pyscf_b631g2dfp_500_100_axis7/selection_plan.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def load_demo_manifest() -> list[dict[str, Any]]:
    path = ROOT / "qmugs_npz/qm9_pyscf_demo_axis7/manifest.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def atom_count_summary(rows: list[dict[str, Any]]) -> str:
    counts: dict[int, int] = {}
    for row in rows:
        n_atoms = int(row["n_atoms"])
        counts[n_atoms] = counts.get(n_atoms, 0) + 1
    return ", ".join(f"{key}: {value}" for key, value in sorted(counts.items()))


def current_status_text() -> str:
    n_npz = npz_count("qmugs_npz/qm9_pyscf_b631g2dfp_500_100_axis7")
    return (
        f"현재 500/100 폴더에는 렌더링 시점 기준 {n_npz}개의 NPZ가 있다. "
        "DFT 변환 계산이 진행 중이면 이 숫자는 계속 증가한다. 같은 변환 명령을 다시 실행하면 "
        "이미 만들어진 NPZ는 건너뛰고 남은 분자부터 이어서 계산한다."
    )


def build_blocks() -> list[Block]:
    plan = load_selection_plan()
    demo = load_demo_manifest()
    plan_summary = (
        f"총 {len(plan)}개, train {sum(row['split'] == 'train' for row in plan)}개, "
        f"validation {sum(row['split'] == 'val' for row in plan)}개, atom count 분포 {{{atom_count_summary(plan)}}}."
        if plan
        else "selection_plan.json이 아직 없으면 같은 변환 명령을 한 번 더 실행하거나 선택 계획 생성 스크립트를 실행하면 된다."
    )
    plan_examples = [
        f"{row['index']:03d} {row['split']:5s} {row['qm9_id']}  {row['formula']}  "
        f"{row['n_atoms']} atoms  {row['electron_count']} e  {row['smiles_gdb']}"
        for row in plan[:12]
    ]
    demo_examples = [
        f"{row['system_id']}: {row.get('formula', '?')}, {row.get('n_atoms', '?')} atoms, "
        f"{row.get('electron_count', '?')} e, SMILES={row.get('smiles_gdb', '')}"
        for row in demo[:12]
    ]

    return [
        {"type": "title", "text": "Transferable 1-RDM / ML-OFDFT 현재 구현 정리"},
        {"type": "para", "text": "작성 기준: 2026-05-20. 이 문서는 현재 폴더에서 실제로 무엇을 계산하고, 어떤 데이터를 넣고, 모델이 어떤 수학적 구조로 gamma(r,r')를 예측하는지 설명한다."},
        {"type": "h1", "text": "1. 큰 그림"},
        {"type": "para", "text": "이 프로젝트의 목표는 3D real-space grid 위의 one-particle reduced density matrix, 즉 1-RDM gamma(r,r')를 직접 학습하는 것이다. 단순히 pair value 하나를 맞추는 regression이 아니라, gamma_theta를 중간 표현으로 두고 density rho, kinetic energy density tau, particle number, occupation spectrum, symmetry를 동시에 제어한다."},
        {"type": "para", "text": "물리적으로는 KS-DFT와 density-only OFDFT 사이의 표현을 만들려는 시도다. KS orbital 전체를 들고 가면 비싸고, density만 쓰면 kinetic/nonlocal 정보가 너무 많이 사라진다. 그래서 gamma(r,r')라는 비국소 표현을 압축된 surrogate로 학습한다."},
        {"type": "h1", "text": "2. 현재 실제 데이터 흐름: QM9 구조 + PySCF DFT"},
        {"type": "para", "text": "처음에는 QMugs를 후보로 보았지만, wavefunction tarball이 매우 커서 10 GB 이하 실험에는 맞지 않았다. 그래서 현재 실제 실행 경로는 QM9의 xyz 구조를 받고, PySCF로 DFT 계산을 직접 수행하여 AO density matrix를 얻는 방식이다."},
        {"type": "bullets", "items": [
            "입력 원본: data/qm9_raw/dsgdb9nsd.xyz.tar.bz2, 약 83 MB.",
            "DFT 코드: PySCF RKS.",
            "DFT 설정: B3LYP / 6-31G(2df,p), charge=0, spin=0, closed-shell, conv_tol=1e-8.",
            "분자 선택: QM9 closed-shell 분자 중 max_atoms=10, 작은 분자부터 600개 선택.",
            "split 계획: 앞 500개 train, 뒤 100개 validation.",
            "grid 설정: axis_points=7, 따라서 N_grid = 7^3 = 343.",
        ]},
        {"type": "para", "text": current_status_text()},
        {"type": "h1", "text": "3. AO density matrix에서 grid gamma_true를 만드는 방법"},
        {"type": "para", "text": "PySCF가 계산하는 DFT wavefunction은 real-space grid 위에 바로 저장되어 있지 않다. 대신 atomic orbital(AO) basis 위의 density matrix P를 얻는다. AO basis function을 chi_mu(r)라고 쓰면, 각 grid point r_g에서 AO 값을 평가하여 행렬 B를 만든다."},
        {"type": "equation", "tex": r"B_{g\mu} = \chi_\mu(\mathbf{r}_g)"},
        {"type": "para", "text": "closed-shell RKS에서 density matrix는 개념적으로 occupied molecular orbital coefficient C와 occupation에서 온다. PySCF에서는 mf.make_rdm1()이 이 P를 만들어 준다."},
        {"type": "equation", "tex": r"P_{\mu\nu} \simeq 2\sum_{i\in \mathrm{occ}} C_{\mu i} C_{\nu i}"},
        {"type": "para", "text": "그러면 grid 위의 1-RDM target은 AO basis를 real-space grid에 평가한 뒤 양쪽에 붙인 것이다."},
        {"type": "equation", "tex": r"\gamma_{\mathrm{true}}(\mathbf{r}_g,\mathbf{r}_h)=\sum_{\mu,\nu}\chi_\mu(\mathbf{r}_g)P_{\mu\nu}\chi_\nu(\mathbf{r}_h)"},
        {"type": "equation", "tex": r"\Gamma^{\mathrm{true}}_{gh}=(B P B^{T})_{gh}"},
        {"type": "para", "text": "여기서 Gamma_true는 코드의 gamma_matrix다. 대각 원소는 density다."},
        {"type": "equation", "tex": r"\rho_{\mathrm{true}}(\mathbf{r}_g)=\gamma_{\mathrm{true}}(\mathbf{r}_g,\mathbf{r}_g)"},
        {"type": "para", "text": "변환기는 grid 적분 trace가 전자수 N과 맞도록 gamma_matrix를 한 번 스케일한다. 거친 7^3 grid에서 trace error를 줄이기 위한 실용적 보정이다."},
        {"type": "equation", "tex": r"\sum_g \gamma_{\mathrm{true}}(\mathbf{r}_g,\mathbf{r}_g)\,\Delta V \approx N"},
        {"type": "h1", "text": "4. 저장 파일과 사람이 보는 인덱스"},
        {"type": "para", "text": "학습은 NPZ를 읽지만, NPZ는 사람이 보기 어렵다. 그래서 데이터 폴더에는 학습용 바이너리와 사람이 보는 인덱스를 함께 둔다."},
        {"type": "code", "text": "qmugs_npz/qm9_pyscf_b631g2dfp_500_100_axis7/\n  *.npz                 # 학습용: points, gamma_matrix, features\n  selection_plan.json   # 600개 선택 계획과 train/val split\n  manifest.json         # 실제 변환 완료된 계산 기록\n  molecule_index.csv    # 사람이 보는 표 형식 인덱스\n  molecule_index.md     # 사람이 보는 Markdown 인덱스\n  xyz/*.xyz             # 분자 구조 확인용"},
        {"type": "para", "text": "중요한 점은 NPZ만으로도 학습은 가능하지만, 연구 데이터셋으로 관리하려면 CSV/Markdown/XYZ가 같이 있어야 한다는 것이다. 현재 변환 스크립트는 재실행 시 이미 만들어진 NPZ를 건너뛰면서 인덱스와 XYZ도 채울 수 있게 되어 있다."},
        {"type": "h1", "text": "5. 500/100 분자 선택 계획"},
        {"type": "para", "text": plan_summary},
        {"type": "bullets", "items": plan_examples or ["선택 계획 파일이 아직 없어서 예시를 표시하지 못했다."]},
        {"type": "para", "text": "12개 smoke demo에 들어간 분자 예시는 다음과 같다."},
        {"type": "bullets", "items": demo_examples or ["smoke demo manifest가 아직 없다."]},
        {"type": "h1", "text": "6. NPZ schema와 차원"},
        {"type": "para", "text": "axis_points=7인 한 분자 system은 다음 배열을 가진다. 여기서 343은 7^3이고, full pair 수는 343^2 = 117,649개다."},
        {"type": "code", "text": "points           : (343, 3)\ngamma_matrix     : (343, 343)\nlocal_features   : (343, 15)\nglobal_context   : (11,)\npotential        : (343, 1)\ngrad_potential   : (343, 3)\nelectron_count   : scalar\noccupancies      : (n_mo,)\norbital_energies : (n_mo,)"},
        {"type": "para", "text": "학습 target의 중심은 gamma_matrix다. rho, tau, trace, occupation penalty는 이 gamma_matrix와 모델 예측으로부터 다시 계산된다."},
        {"type": "h1", "text": "7. local point feature xi(r): 15차원"},
        {"type": "para", "text": "xi(r)는 grid point 하나의 local environment를 설명한다. 현재 QM9/PySCF 변환기에서는 다음 15개 feature를 쓴다."},
        {"type": "code", "text": "1-3   normalized coordinate r/R                    3\n4     softened nuclear potential V_nuc(r)          1\n5-7   gradient of V_nuc(r)                         3\n8     radial distance |r|/R                         1\n9-13  element-wise Gaussian atom density H,C,N,O,F  5\n14    nearest atom nuclear charge / 10              1\n15    electron count / 30                           1"},
        {"type": "h1", "text": "8. global context c: 11차원"},
        {"type": "para", "text": "c는 분자 전체를 요약한다. 같은 local coordinate라도 어떤 분자 안에 있느냐에 따라 gamma의 의미가 달라지므로, point/pair model에 global context를 함께 넣는다."},
        {"type": "code", "text": "electron_count / 30\natom_count / 30\nheavy_atom_count / 10\nmean(Z) / 10\nstd(Z) / 10\nmolecular_radius / 10\ncounts(H,C,N,O,F) / 10      # 5개"},
        {"type": "h1", "text": "9. pair feature eta(r,r'): 18차원"},
        {"type": "para", "text": "eta(r,r')는 두 grid point 사이의 관계를 설명한다. gamma는 두 점 함수이므로 point feature만으로는 부족하고, separation과 local field mismatch를 직접 알려주는 pair descriptor가 필요하다."},
        {"type": "code", "text": "midpoint (r+r')/2                              3\nabsolute separation |r-r'| components           3\nsquared separation components                   3\nseparation norm |r-r'|                          1\nsquared norm |r-r'|^2                           1\naverage potential [V(r)+V(r')]/2                1\naverage gradient [grad V(r)+grad V(r')]/2       3\nabsolute gradient difference                    3"},
        {"type": "para", "text": "이 feature들은 대부분 r과 r'를 바꾸어도 변하지 않는 대칭량이다. 이렇게 하면 모델이 gamma(r,r') = gamma(r',r)라는 Hermitian/symmetric 구조를 배우기 쉽다."},
        {"type": "h1", "text": "10. 전체 모델은 세 개의 submodel로 구성된다"},
        {"type": "para", "text": "현재 ModelBundle은 point_model, pair_model, context_model 세 개로 구성된다. 세 모델은 모두 Random Fourier Features(RFF)로 입력을 확장한 뒤 SiLU activation을 쓰는 MLP다."},
        {"type": "equation", "tex": r"\mathrm{RFF}(x)=\left[x,\sin(2\pi x\Omega),\cos(2\pi x\Omega)\right]"},
        {"type": "para", "text": "SiLU activation은 x sigmoid(x) 형태의 smooth activation이다. tau처럼 derivative-sensitive한 물리량을 맞출 때 ReLU보다 부드러운 함수가 유리하다."},
        {"type": "h2", "text": "10.1 point model"},
        {"type": "para", "text": "point model은 한 점의 local feature xi(r)와 global context c를 받아 density logit과 residual mode amplitude를 출력한다. r과 r'에 같은 point model을 공유 적용한다."},
        {"type": "equation", "tex": r"\mathrm{point}_\theta([\xi(\mathbf{r}),c])=\left[\ell_\rho(\mathbf{r};c),a_1(\mathbf{r};c),\ldots,a_k(\mathbf{r};c)\right]"},
        {"type": "equation", "tex": r"\rho_\theta(\mathbf{r};c)=\mathrm{softplus}(\ell_\rho(\mathbf{r};c))+\varepsilon"},
        {"type": "code", "text": "입력 차원: 15 + 11 = 26\n구조: RFF -> Dense(W, SiLU) x 3 -> Dense(1+k)\n출력: density logit 1개 + residual mode amplitude k개"},
        {"type": "h2", "text": "10.2 context model"},
        {"type": "para", "text": "context model은 분자 전체 descriptor c만 보고 system-specific residual mode weight를 만든다. 같은 local mode라도 분자 종류에 따라 중요도가 달라질 수 있기 때문이다."},
        {"type": "equation", "tex": r"\mathrm{context}_\theta(c)=q(c)\in R^k"},
        {"type": "equation", "tex": r"w(c)=\left[1,w_1(c),\ldots,w_k(c)\right],\qquad w_i(c)>0"},
        {"type": "code", "text": "입력 차원: 11\n구조: RFF -> Dense(max(64,W/2), SiLU) x 2 -> Dense(k)\n후처리: softplus + cumulative descending ordering + anchor mode"},
        {"type": "h2", "text": "10.3 pair model"},
        {"type": "para", "text": "pair model은 eta(r,r')와 c를 받아 Gaussian baseline의 폭 alpha_theta와 residual correction을 얼마나 섞을지 정하는 scalar gate m_theta를 출력한다."},
        {"type": "equation", "tex": r"\mathrm{pair}_\theta([\eta(\mathbf{r},\mathbf{r}'),c])=\left[\beta(\mathbf{r},\mathbf{r}';c),z(\mathbf{r},\mathbf{r}';c)\right]"},
        {"type": "equation", "tex": r"\alpha_\theta=\mathrm{softplus}(\beta)+\varepsilon,\qquad m_\theta=\sigma(z)"},
        {"type": "code", "text": "입력 차원: 18 + 11 = 29\n구조: RFF -> Dense(W, SiLU) x 2 -> Dense(2)\n출력: beta 1개, gate logit z 1개"},
        {"type": "h1", "text": "11. g_theta의 정확한 정의"},
        {"type": "para", "text": "이전 문서에서는 g_theta라는 이름이 빠져 있거나 gate와 혼동될 수 있었다. 여기서는 명확히 분리한다. g_theta는 scalar gate가 아니라, 각 grid point를 normalized latent vector로 보내는 embedding이다."},
        {"type": "para", "text": "point model의 residual mode amplitude를 a_theta(r;c)라고 하고, constant anchor channel을 붙인 feature를 phi_theta라고 하자."},
        {"type": "equation", "tex": r"a_\theta(\mathbf{r};c)=\left[a_1(\mathbf{r};c),\ldots,a_k(\mathbf{r};c)\right]"},
        {"type": "equation", "tex": r"\phi_\theta(\mathbf{r};c)=\left[1,a_1(\mathbf{r};c),\ldots,a_k(\mathbf{r};c)\right]"},
        {"type": "para", "text": "context model이 만든 positive mode weight w(c)를 element-wise로 곱하고 L2 normalization하면 g_theta가 된다."},
        {"type": "equation", "tex": r"g_\theta(\mathbf{r};c)=\frac{\sqrt{w(c)}\odot\phi_\theta(\mathbf{r};c)}{\left\|\sqrt{w(c)}\odot\phi_\theta(\mathbf{r};c)\right\|_2}"},
        {"type": "para", "text": "따라서 residual kernel은 두 점 embedding의 내적이다."},
        {"type": "equation", "tex": r"K_{\mathrm{res},\theta}(\mathbf{r},\mathbf{r}';c)=g_\theta(\mathbf{r};c)\cdot g_\theta(\mathbf{r}';c)"},
        {"type": "para", "text": "코드 변수로는 unit_r, unit_rp가 g_theta에 해당하고, residual_kernel이 위 내적에 해당한다. 반면 코드 변수 gate는 scalar m_theta다. 이 둘은 서로 다른 개념이다."},
        {"type": "h1", "text": "12. K_theta와 gamma_theta의 의미"},
        {"type": "para", "text": "gamma_theta는 density amplitude와 normalized kernel의 곱으로 쓴다. K_theta는 density scale을 제거한 뒤 남는 off-diagonal coherence를 표현한다."},
        {"type": "equation", "tex": r"\gamma_\theta(\mathbf{r},\mathbf{r}';c)=\sqrt{\rho_\theta(\mathbf{r};c)\rho_\theta(\mathbf{r}';c)}\,K_\theta(\mathbf{r},\mathbf{r}';c)"},
        {"type": "para", "text": "baseline kernel은 pair model이 예측한 alpha_theta를 사용한 Gaussian decay다."},
        {"type": "equation", "tex": r"K_{\mathrm{base},\theta}(\mathbf{r},\mathbf{r}';c)=\exp\left[-\alpha_\theta(\mathbf{r},\mathbf{r}';c)\|\mathbf{r}-\mathbf{r}'\|^2\right]"},
        {"type": "para", "text": "최종 kernel은 baseline과 residual을 scalar gate m_theta로 섞는다."},
        {"type": "equation", "tex": r"K_\theta=K_{\mathrm{base},\theta}\left[(1-m_\theta)+m_\theta K_{\mathrm{res},\theta}\right]"},
        {"type": "para", "text": "해석은 다음과 같다. K_base는 가까운 pair에서 기본적인 decay를 주는 물리적 bias다. K_res는 orbital-like nonlocal coherence correction이다. m_theta는 이 pair에서 residual correction을 얼마나 믿을지 정하는 adaptive switch다."},
        {"type": "para", "text": "대각에서는 K_theta(r,r) ~= 1이 되어야 gamma_theta(r,r) = rho_theta(r)가 된다. 그래서 loss에 kernel diagonal normalization을 넣는다."},
        {"type": "h1", "text": "13. 학습 loss와 물리량"},
        {"type": "para", "text": "전체 objective는 여러 물리 조건을 동시에 반영한다."},
        {"type": "equation", "tex": r"L=\lambda_\gamma L_\gamma+\lambda_\rho L_\rho+\lambda_K L_K+\lambda_\partial L_\partial+\lambda_\tau L_\tau+\lambda_{\mathrm{trace}}L_{\mathrm{trace}}+\lambda_{\mathrm{occ}}L_{\mathrm{occ}}+\lambda_{\mathrm{mode}}L_{\mathrm{mode}}"},
        {"type": "bullets", "items": [
            "L_gamma: sampled pair gamma(r,r') weighted MSE.",
            "L_rho: diagonal density gamma_theta(r,r)와 rho_true(r)의 MSE.",
            "L_K: K_theta(r,r)=1 normalization.",
            "L_partial: near-diagonal mixed derivative finite-difference loss.",
            "L_tau: kinetic energy density tau(r) loss.",
            "L_trace: integral rho(r) dr = electron count.",
            "L_occ: coarse 1-RDM eigenvalue가 occupation 범위를 벗어나지 않도록 하는 penalty.",
            "L_mode: residual mode weight regularization.",
        ]},
        {"type": "para", "text": "kinetic energy density는 gamma의 대각 근처 곡률에 민감하다. 그래서 단순 pair MSE와 별개로 mixed derivative stencil을 직접 넣는다."},
        {"type": "equation", "tex": r"\tau(\mathbf{r})=\frac{1}{2}\sum_{\alpha=x,y,z}\left.\partial_{r_\alpha}\partial_{r'_\alpha}\gamma(\mathbf{r},\mathbf{r}')\right|_{\mathbf{r}'=\mathbf{r}}"},
        {"type": "para", "text": "molecular closed-shell DFT density matrix의 natural occupation 범위는 0에서 2다. 그래서 QM9/PySCF 학습에서는 RDM_OCC_MAX=2.0을 쓰는 것이 맞다."},
        {"type": "h1", "text": "14. 학습 batch는 어떻게 만들어지는가"},
        {"type": "para", "text": "train loop는 먼저 train system 중 하나를 균등하게 고른다. 그 system에서 pair index를 뽑는다. 초반에는 diagonal/near-diagonal pair를 많이 보고, 후반에는 far off-diagonal pair 비중을 늘리는 curriculum sampling을 사용한다."},
        {"type": "bullets", "items": [
            "diag pair: r = r'. density와 trace에 중요하다.",
            "near pair: kinetic energy density tau에 중요하다.",
            "mid/far pair: nonlocal coherence와 off-diagonal structure에 중요하다.",
            "pair weight: diagonal과 near-diagonal pair에 더 큰 weight를 준다.",
        ]},
        {"type": "h1", "text": "15. 학습이 끝나면 얻는 output"},
        {"type": "para", "text": "학습 스크립트는 결과 그림, 요약 JSON, 세 submodel의 weight를 저장한다."},
        {"type": "code", "text": "transferable_outputs/<run_name>.png\ntransferable_outputs/<run_name>_summary.json\ntransferable_outputs/<run_name>_point.weights.h5\ntransferable_outputs/<run_name>_pair.weights.h5\ntransferable_outputs/<run_name>_context.weights.h5"},
        {"type": "para", "text": "PNG에는 training curve, held-out gamma parity, density slice, tau slice, natural occupation spectrum, metric summary가 들어간다."},
        {"type": "h1", "text": "16. 현재 실행 명령"},
        {"type": "code", "text": "/home/hbji/miniconda3/envs/polymer-gp/bin/python scripts/build_qm9_pyscf_npz.py \\\n  --num-systems 600 \\\n  --axis-points 7 \\\n  --max-atoms 10 \\\n  --basis '6-31g(2df,p)' \\\n  --xc b3lyp \\\n  --selection smallest \\\n  --output-dir qmugs_npz/qm9_pyscf_b631g2dfp_500_100_axis7"},
        {"type": "code", "text": "env MPLBACKEND=Agg RDM_OCC_MAX=2.0 \\\n/home/hbji/miniconda3/envs/polymer-gp/bin/python train_transferable_1rdm.py \\\n  --dataset-mode npz \\\n  --npz-glob 'qmugs_npz/qm9_pyscf_b631g2dfp_500_100_axis7/*.npz' \\\n  --axis-points 7 \\\n  --train-system-count 500 \\\n  --val-system-count 100 \\\n  --run-name qm9_b631g2dfp_500_100_axis7"},
        {"type": "h1", "text": "17. 현재 한계와 다음 단계"},
        {"type": "bullets", "items": [
            "axis_points=7은 매우 거친 grid다. 모델/데이터 흐름 검증에는 좋지만 고정밀 물리량에는 부족하다.",
            "full gamma_matrix 저장 비용은 axis_points^6로 증가한다. axis_points=9 이상부터는 sampled pair 저장 또는 on-the-fly AO evaluation이 필요하다.",
            "600개 생성은 디스크보다 SCF 계산 시간이 병목이다.",
            "현재는 QM9 구조를 PySCF로 직접 계산한 DFT density matrix를 사용한다. QMugs wavefunction tarball을 직접 파싱한 것은 아니다.",
            "local feature는 핵전하 potential 기반의 단순 descriptor다. 더 정교한 chemical descriptor를 추가할 수 있다.",
        ]},
        {"type": "para", "text": "요약하면, 현재 폴더는 QM9/PySCF로 계산한 실제 DFT AO density matrix를 3D grid 1-RDM target으로 바꾸고, 이를 point/pair/context 구조의 transferable neural surrogate로 학습하는 연구 prototype이다."},
    ]


def blocks_to_markdown(blocks: list[Block]) -> str:
    lines: list[str] = []
    for block in blocks:
        kind = block["type"]
        if kind == "title":
            lines.append(f"# {block['text']}")
        elif kind == "h1":
            lines.append(f"## {block['text']}")
        elif kind == "h2":
            lines.append(f"### {block['text']}")
        elif kind == "para":
            lines.append(block["text"])
        elif kind == "bullets":
            lines.extend(f"- {item}" for item in block["items"])
        elif kind == "code":
            lines.append("```text")
            lines.append(block["text"])
            lines.append("```")
        elif kind == "equation":
            lines.append("$$")
            lines.append(block["tex"])
            lines.append("$$")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def blocks_to_tex(blocks: list[Block]) -> str:
    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage{fontspec}",
        r"\usepackage{kotex}",
        r"\usepackage[a4paper,margin=1in]{geometry}",
        r"\usepackage{amsmath,amssymb,bm,mathtools}",
        r"\usepackage{enumitem}",
        r"\usepackage{setspace}",
        r"\usepackage{hyperref}",
        r"\usepackage{xcolor}",
        r"\usepackage{listings}",
        r"\lstset{basicstyle=\ttfamily\small,breaklines=true,frame=single,columns=fullflexible}",
        r"\setstretch{1.14}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{0.45em}",
        r"\title{Transferable 1-RDM / ML-OFDFT 현재 구현 정리}",
        r"\author{}",
        r"\date{\today}",
        r"\begin{document}",
        r"\maketitle",
    ]
    first_title = True
    for block in blocks:
        kind = block["type"]
        if kind == "title":
            if first_title:
                first_title = False
            else:
                lines.append(r"\section*{" + latex_escape(block["text"]) + "}")
        elif kind == "h1":
            lines.append(r"\section*{" + latex_escape(block["text"]) + "}")
        elif kind == "h2":
            lines.append(r"\subsection*{" + latex_escape(block["text"]) + "}")
        elif kind == "para":
            lines.append(latex_escape(block["text"]))
        elif kind == "bullets":
            lines.append(r"\begin{itemize}[leftmargin=1.5em]")
            for item in block["items"]:
                lines.append(r"\item " + latex_escape(item))
            lines.append(r"\end{itemize}")
        elif kind == "code":
            lines.append(r"\begin{lstlisting}")
            lines.append(block["text"])
            lines.append(r"\end{lstlisting}")
        elif kind == "equation":
            lines.append(r"\[")
            lines.append(block["tex"])
            lines.append(r"\]")
    lines.append(r"\end{document}")
    return "\n".join(lines) + "\n"


def wrap_text(text: str, width: int) -> list[str]:
    return textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False) or [""]


def render_pdf(blocks: list[Block]) -> None:
    font = FontProperties(fname=str(FONT), size=9.2)
    title_font = FontProperties(fname=str(FONT), size=15)
    h1_font = FontProperties(fname=str(FONT), size=11.8)
    h2_font = FontProperties(fname=str(FONT), size=10.2)
    code_font = FontProperties(fname=str(FONT), size=7.8)

    page_w, page_h = 8.27, 11.69
    left, right, top, bottom = 0.68, 0.55, 0.62, 0.55
    line_h = 0.18
    code_h = 0.155

    pdf = PdfPages(PDF)
    fig = None
    ax = None
    y = 0.0
    page_no = 0

    def new_page() -> None:
        nonlocal fig, ax, y, page_no
        if fig is not None:
            ax.text(page_w / 2, 0.28, str(page_no), fontproperties=font, ha="center", color="#666666")
            pdf.savefig(fig)
            plt.close(fig)
        page_no += 1
        fig = plt.figure(figsize=(page_w, page_h))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, page_w)
        ax.set_ylim(0, page_h)
        ax.axis("off")
        y = page_h - top

    def ensure(height: float) -> None:
        if y - height < bottom:
            new_page()

    def put(text: str, x: float, prop: FontProperties, height: float, color: str = "#111111") -> None:
        nonlocal y
        assert ax is not None
        ax.text(x, y, text, fontproperties=prop, va="top", color=color)
        y -= height

    new_page()
    for block in blocks:
        kind = block["type"]
        if kind == "title":
            ensure(0.7)
            put(block["text"], left, title_font, 0.45, "#1f4e79")
            y -= 0.05
        elif kind == "h1":
            ensure(0.55)
            y -= 0.05
            put(block["text"], left, h1_font, 0.31, "#1f4e79")
        elif kind == "h2":
            ensure(0.42)
            put(block["text"], left, h2_font, 0.25, "#376092")
        elif kind == "para":
            lines = wrap_text(block["text"], 78)
            ensure(len(lines) * line_h + 0.05)
            for line in lines:
                put(line, left, font, line_h)
            y -= 0.04
        elif kind == "bullets":
            for item in block["items"]:
                lines = wrap_text(item, 74)
                ensure(len(lines) * line_h + 0.04)
                put("• " + lines[0], left + 0.12, font, line_h)
                for cont in lines[1:]:
                    put("  " + cont, left + 0.20, font, line_h)
            y -= 0.04
        elif kind == "code":
            lines = block["text"].splitlines()
            ensure(len(lines) * code_h + 0.16)
            assert ax is not None
            box_top = y + 0.04
            box_bottom = y - len(lines) * code_h - 0.08
            ax.add_patch(
                plt.Rectangle(
                    (left - 0.05, box_bottom),
                    page_w - left - right + 0.04,
                    box_top - box_bottom,
                    facecolor="#f6f8fa",
                    edgecolor="#d0d7de",
                    linewidth=0.7,
                )
            )
            for line in lines:
                put(line, left, code_font, code_h, "#111111")
            y -= 0.10
        elif kind == "equation":
            ensure(0.52)
            assert ax is not None
            ax.text(
                page_w / 2,
                y,
                "$" + block["tex"] + "$",
                ha="center",
                va="top",
                fontsize=11,
                color="#111111",
            )
            y -= 0.46
    if fig is not None:
        assert ax is not None
        ax.text(page_w / 2, 0.28, str(page_no), fontproperties=font, ha="center", color="#666666")
        pdf.savefig(fig)
        plt.close(fig)
    pdf.close()


def main() -> None:
    blocks = build_blocks()
    MD.write_text(blocks_to_markdown(blocks), encoding="utf-8")
    TEX.write_text(blocks_to_tex(blocks), encoding="utf-8")
    render_pdf(blocks)
    print(f"Wrote {MD}")
    print(f"Wrote {TEX}")
    print(f"Wrote {PDF}")


if __name__ == "__main__":
    main()
