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
    n_npz = npz_count("qmugs_npz/qm9_pyscf_ldavwn_b631gd_atoms10_400_50_50_spacing1p5_kp")
    return (
        f"현재 LDA/VWN 400/50/50 폴더에는 렌더링 시점 기준 {n_npz}개의 NPZ가 있다. "
        "pipeline을 다시 실행하면 같은 NPZ_DIR 안의 정상 NPZ는 건너뛰고, 누락되었거나 "
        "필수 key가 빠진 손상 NPZ만 다시 만든다."
    )


def build_blocks() -> list[Block]:
    return [
        {"type": "title", "text": "Transferable 1-RDM surrogate: current implementation note"},
        {"type": "para", "text": "작성 기준: 2026-05-26. 이 문서는 현재 코드가 실제로 어떤 데이터를 만들고, 어떤 모델로 gamma(r,r')를 예측하며, 어떤 loss와 실행 옵션을 사용하는지 정리한다."},
        {"type": "h1", "text": "1. 목표와 현재 방향"},
        {"type": "para", "text": "목표는 분자마다 Cartesian real-space grid 위의 spin-summed closed-shell 1-RDM gamma(r,r')를 직접 예측하는 transferable surrogate를 만드는 것이다. density rho만 맞추는 모델이 아니라, gamma를 중심 표현으로 두고 density, kinetic potential, kinetic energy, trace consistency, off-diagonal coherence를 함께 점검한다."},
        {"type": "equation", "tex": r"\gamma_\theta(\mathbf r,\mathbf r')=\sqrt{\rho_\theta(\mathbf r)\rho_\theta(\mathbf r')}\,K_\theta(\mathbf r,\mathbf r')"},
        {"type": "para", "text": "현재의 실험 방향은 gamma 자체를 강하게 맞추되, 후반부에 kinetic-potential(KP) loss와 scalar kinetic-energy T loss를 schedule로 천천히 켜는 것이다. 이는 초반부터 어려운 auxiliary target을 강제하면 gamma/rho fit이 흔들릴 수 있기 때문이다."},
        {"type": "h1", "text": "2. 데이터 생성 pipeline"},
        {"type": "para", "text": "현재 실제 데이터 경로는 QM9 raw xyz tar를 읽고, PySCF RKS 계산을 수행한 뒤, AO density matrix를 grid 1-RDM target으로 변환하여 NPZ로 저장하는 방식이다."},
        {"type": "bullets", "items": [
            "QM9 raw tar: data/qm9_raw/dsgdb9nsd.xyz.tar.bz2.",
            "자동 다운로드 URL: https://ndownloader.figshare.com/files/3195389.",
            "DFT backend: PySCF RKS.",
            "현재 weekend dataset 설정: xc=lda,vwn, basis=6-31g(d), max_atoms=10.",
            "grid: target spacing 1.5 bohr, max_axis_points=21, 분자 크기에 따라 axis_points는 대략 9..21.",
            "split: 400 train / 50 validation / 50 test.",
            "출력 폴더: qmugs_npz/qm9_pyscf_ldavwn_b631gd_atoms10_400_50_50_spacing1p5_kp.",
        ]},
        {"type": "para", "text": current_status_text()},
        {"type": "para", "text": "서버 재현성을 위해 pipeline은 Python 경로를 고정하지 않는다. PYTHON을 따로 지정하지 않으면 현재 활성화된 conda environment의 python을 사용한다. PySCF가 없으면 데이터 생성에서 실패하고, TensorFlow가 없으면 학습에서 실패한다."},
        {"type": "h1", "text": "3. AO density matrix에서 grid gamma target 생성"},
        {"type": "para", "text": "PySCF는 AO basis 위의 density matrix P를 만든다. grid point r_g에서 AO basis function chi_mu를 평가하여 B 행렬을 구성한다."},
        {"type": "equation", "tex": r"B_{g\mu} = \chi_\mu(\mathbf{r}_g)"},
        {"type": "para", "text": "closed-shell RKS에서는 occupation이 2인 occupied orbital에서 density matrix가 만들어진다."},
        {"type": "equation", "tex": r"P_{\mu\nu} \simeq 2\sum_{i\in \mathrm{occ}} C_{\mu i} C_{\nu i}"},
        {"type": "para", "text": "grid 위 target gamma는 AO density matrix를 양쪽에서 grid AO 값으로 감싼 것이다."},
        {"type": "equation", "tex": r"\gamma_{\mathrm{true}}(\mathbf{r}_g,\mathbf{r}_h)=\sum_{\mu,\nu}\chi_\mu(\mathbf{r}_g)P_{\mu\nu}\chi_\nu(\mathbf{r}_h)"},
        {"type": "equation", "tex": r"\Gamma^{\mathrm{true}}_{gh}=(B P B^{T})_{gh}"},
        {"type": "para", "text": "대각 성분은 density target이다."},
        {"type": "equation", "tex": r"\rho_{\mathrm{true}}(\mathbf{r}_g)=\gamma_{\mathrm{true}}(\mathbf{r}_g,\mathbf{r}_g)"},
        {"type": "para", "text": "grid trace가 전자수와 맞도록 gamma_matrix, tau, derivative target을 같은 factor로 normalize한다. 이 값은 NPZ의 gamma_trace_scale에 저장된다."},
        {"type": "equation", "tex": r"\sum_g \gamma_{\mathrm{true}}(\mathbf{r}_g,\mathbf{r}_g)\,\Delta V \approx N"},
        {"type": "h1", "text": "4. NPZ schema"},
        {"type": "para", "text": "새 NPZ에는 학습 target과 진단 target이 함께 들어간다. builder는 기존 NPZ가 있어도 필수 key가 빠져 있으면 손상 파일로 판단하고 삭제 후 다시 만든다."},
        {"type": "code", "text": "qmugs_npz/qm9_pyscf_b631g2dfp_500_100_axis7/\n  *.npz                 # 학습용: points, gamma_matrix, features\n  selection_plan.json   # 600개 선택 계획과 train/val split\n  manifest.json         # 실제 변환 완료된 계산 기록\n  molecule_index.csv    # 사람이 보는 표 형식 인덱스\n  molecule_index.md     # 사람이 보는 Markdown 인덱스\n  xyz/*.xyz             # 분자 구조 확인용"},
        {"type": "code", "text": "points                        : (n_grid, 3)\ngamma_matrix                  : (n_grid, n_grid)\nderivative_true_ao             : (n_interior, 3)\ntau_true_ao                    : (n_interior, 1)\nlocal_features                 : (n_grid, 15)\nglobal_context                 : (11,)\npotential, grad_potential      : nuclear potential descriptors\nhartree_potential              : (n_grid, 1)\nxc_potential_local             : (n_grid, 1)\nks_potential                   : (n_grid, 1)\nkinetic_potential              : mu - v_s(r)\nkinetic_potential_centered     : rho-weighted centered KP\nelectron_count                 : scalar\nkinetic_energy_hartree         : scalar T_s reference\ngamma_trace_scale              : scalar"},
        {"type": "h1", "text": "5. 입력 feature 차원"},
        {"type": "h2", "text": "5.1 point/local feature xi(r): 15차원"},
        {"type": "code", "text": "1-3   normalized coordinate r/R                    3\n4     softened nuclear potential V_nuc(r)          1\n5-7   gradient of V_nuc(r)                         3\n8     radial distance |r|/R                         1\n9-13  element-wise Gaussian atom density H,C,N,O,F  5\n14    nearest atom nuclear charge / 10              1\n15    electron count / 30                           1"},
        {"type": "h2", "text": "5.2 global context c: 11차원"},
        {"type": "code", "text": "electron_count / 30\natom_count / 30\nheavy_atom_count / 10\nmean(Z) / 10\nstd(Z) / 10\nmolecular_radius / 10\ncounts(H,C,N,O,F) / 10      # 5개"},
        {"type": "h2", "text": "5.3 pair feature eta(r,r'): 18차원"},
        {"type": "code", "text": "midpoint (r+r')/2                              3\nabsolute separation |r-r'| components           3\nsquared separation components                   3\nseparation norm |r-r'|                          1\nsquared norm |r-r'|^2                           1\naverage potential [V(r)+V(r')]/2                1\naverage gradient [grad V(r)+grad V(r')]/2       3\nabsolute gradient difference                    3"},
        {"type": "para", "text": "따라서 raw 입력 차원은 point model 26, pair model 29, context model 11이다. RFF를 사용하면 내부적으로 x, sin(2*pi*x*Omega), cos(2*pi*x*Omega)가 concatenate되어 MLP에 들어간다."},
        {"type": "h1", "text": "6. 모델 구조"},
        {"type": "para", "text": "ModelBundle은 point_model, pair_model, context_model 세 개의 작은 MLP로 구성된다. 모두 RFF-enhanced MLP이며 activation은 SiLU다."},
        {"type": "equation", "tex": r"\mathrm{RFF}(x)=\left[x,\sin(2\pi x\Omega),\cos(2\pi x\Omega)\right]"},
        {"type": "h2", "text": "6.1 point model"},
        {"type": "code", "text": "input  : local_features(r) 15 + global_context 11 = 26\noutput : [rho logit, kinetic-potential head, latent amplitudes]\nsize   : RFF -> Dense(width, SiLU) x 3 -> Dense(2 + learned_rank)"},
        {"type": "equation", "tex": r"\rho_\theta(\mathbf r)=\mathrm{softplus}(\ell_\rho(\mathbf r))+\epsilon"},
        {"type": "para", "text": "density normalization을 켜면 rho_theta 전체 grid 적분이 electron_count와 같도록 rescale한다."},
        {"type": "h2", "text": "6.2 context model"},
        {"type": "code", "text": "input  : global_context 11\noutput : molecule-specific latent mode weights\nsize   : RFF -> Dense(max(64,width/2), SiLU) x 2 -> Dense(learned_rank)"},
        {"type": "h2", "text": "6.3 pair model"},
        {"type": "code", "text": "input  : pair_features(r,r') 18 + global_context 11 = 29\noutput : [baseline width beta, residual gate logit]\nsize   : RFF -> Dense(width, SiLU) x 2 -> Dense(2)"},
        {"type": "h1", "text": "7. Kernel construction"},
        {"type": "para", "text": "point model의 latent amplitude에 constant anchor channel을 붙이고, context model의 positive mode weight를 곱한 뒤 normalize하여 g_theta를 만든다."},
        {"type": "equation", "tex": r"\phi_\theta(\mathbf r)=\left[1,a_1(\mathbf r),\ldots,a_k(\mathbf r)\right]"},
        {"type": "equation", "tex": r"g_\theta(\mathbf r)=\frac{\sqrt{w}\odot\phi_\theta(\mathbf r)}{\|\sqrt{w}\odot\phi_\theta(\mathbf r)\|_2}"},
        {"type": "equation", "tex": r"K_{\mathrm{res}}(\mathbf r,\mathbf r')=g_\theta(\mathbf r)\cdot g_\theta(\mathbf r')"},
        {"type": "para", "text": "pair model은 baseline Gaussian decay와 residual correction gate를 만든다."},
        {"type": "equation", "tex": r"K_{\mathrm{base}}=\exp[-\alpha_\theta(\mathbf r,\mathbf r')\|\mathbf r-\mathbf r'\|^2]"},
        {"type": "equation", "tex": r"K_\theta=K_{\mathrm{base}}\left[(1-m_\theta)+m_\theta K_{\mathrm{res}}\right]"},
        {"type": "h1", "text": "8. Loss: 지원되는 전체 항"},
        {"type": "para", "text": "training.py는 다음 loss들을 지원한다. preset에 따라 모두 쓰거나 일부만 쓴다."},
        {"type": "equation", "tex": r"L=\sum_i \lambda_i(epoch)L_i"},
        {"type": "bullets", "items": [
            "gamma: sampled pair gamma(r,r') weighted MSE.",
            "rho: diagonal density gamma_theta(r,r) vs rho_true(r).",
            "kernel: diagonal kernel K_theta(r,r) -> 1.",
            "deriv: near-diagonal mixed derivative components.",
            "tau: tau(r) kinetic energy density.",
            "trace: integral rho(r) dr = electron_count.",
            "occ: coarse spectral occupation penalty, with OCC_MAX=2.0 for closed-shell spin-summed 1-RDM.",
            "mode: weak latent mode regularization.",
            "kinetic: scalar T_s loss from integral tau_pred.",
            "kp: centered kinetic potential loss from point-model KP head.",
        ]},
        {"type": "h1", "text": "9. 현재 active loss와 schedule"},
        {"type": "para", "text": "현재 weekend pipeline은 custom preset을 사용한다. gamma/rho/kernel/KP/kinetic을 켜고 trace/mode/deriv/tau/occ는 끈다. tau와 deriv는 evaluation metric으로 계속 기록하지만 training objective에는 직접 넣지 않는다."},
        {"type": "code", "text": "RDM_LOSS_PRESET=custom\nRDM_USE_GAMMA_LOSS=1     RDM_LAMBDA_GAMMA=8.0\nRDM_USE_RHO_LOSS=1       RDM_LAMBDA_RHO=2.0\nRDM_USE_KERNEL_LOSS=1    RDM_LAMBDA_KERNEL=1.0\nRDM_USE_KP_LOSS=1        RDM_LAMBDA_KP=0.75\nRDM_USE_KINETIC_LOSS=1   RDM_LAMBDA_KINETIC=0.20\nRDM_USE_TRACE_LOSS=0\nRDM_USE_MODE_LOSS=0\nRDM_USE_DERIV_LOSS=0\nRDM_USE_TAU_LOSS=0\nRDM_USE_OCC_LOSS=0"},
        {"type": "para", "text": "KP와 kinetic loss는 epoch-dependent ramp를 사용한다."},
        {"type": "equation", "tex": r"\lambda_{\mathrm{KP}}(e)=\lambda_{\mathrm{KP}}^{\max}\min\left(1,\frac{e-e_0+1}{R}\right)"},
        {"type": "code", "text": "KP schedule : start epoch 100, ramp 60 epochs\nT schedule  : start epoch 240, ramp 80 epochs"},
        {"type": "para", "text": "schedule stage가 바뀌면 early stopping과 LR scheduler 기준점을 reset한다. loss landscape가 바뀐 뒤 이전 best validation 기준으로 바로 early stop되는 것을 막기 위한 처리다."},
        {"type": "h1", "text": "10. Kinetic quantities"},
        {"type": "para", "text": "tau는 gamma의 mixed derivative에서 정의한다."},
        {"type": "equation", "tex": r"\tau(\mathbf{r})=\frac{1}{2}\sum_{\alpha=x,y,z}\left.\partial_{r_\alpha}\partial_{r'_\alpha}\gamma(\mathbf{r},\mathbf{r}')\right|_{\mathbf{r}'=\mathbf{r}}"},
        {"type": "para", "text": "reference tau는 가능하면 AO gradient에서 직접 계산한 tau_true_ao를 사용한다. predicted tau는 model gamma에 finite-difference stencil을 적용한다. RDM_TAU_STENCIL=richardson이면 h와 2h stencil을 결합해 O(h^2) error를 줄인다."},
        {"type": "equation", "tex": r"T_s[\gamma]\approx\int \tau_\theta(\mathbf r)d\mathbf r\approx \sum_g \tau_\theta(\mathbf r_g)\Delta V"},
        {"type": "para", "text": "KP head는 point model output 중 하나다. loss에서는 reference와 prediction 모두 rho-weighted 평균 shift를 제거한 centered kinetic potential을 비교한다. 이유는 potential이 additive constant gauge를 갖기 때문이다."},
        {"type": "equation", "tex": r"\tilde v_T(\mathbf r)=v_T(\mathbf r)-\frac{\int \rho(\mathbf r)v_T(\mathbf r)d\mathbf r}{\int \rho(\mathbf r)d\mathbf r}"},
        {"type": "h1", "text": "11. Pair sampling과 weights"},
        {"type": "para", "text": "학습 step마다 train molecule 하나를 균등하게 고르고, 그 molecule에서 pair batch를 뽑는다. batch_size=1024는 1024개의 (r,r') pair sample을 의미한다."},
        {"type": "bullets", "items": [
            "diag pair: r = r'. density와 trace에 중요하다.",
            "near pair: kinetic energy density tau에 중요하다.",
            "mid/far pair: nonlocal coherence와 off-diagonal structure에 중요하다.",
            "pair weight base: diag=20, near=8, mid=4, far=1.",
            "batch 내부 평균 weight가 1이 되도록 normalize한다.",
        ]},
        {"type": "h1", "text": "12. Evaluation and saved artifacts"},
        {"type": "para", "text": "evaluation은 validation/test molecules에서 sampled gamma, full diagonal density, tau stencil, scalar T, centered KP, trace, symmetry, near/far MAE를 계산한다. 대표 molecule에는 그림용 배열도 저장한다."},
        {"type": "code", "text": "result/<run>.png\nresult/<run>_summary.json\nresult/<run>_history.csv\nresult/<run>_split_metrics.csv\nresult/<run>_per_system_metrics.csv\nresult/<run>_point.weights.h5\nresult/<run>_pair.weights.h5\nresult/<run>_context.weights.h5"},
        {"type": "para", "text": "plotting은 같은 물리량의 true/pred colorbar scale을 공유하도록 수정되어, density/tau/KP 비교가 더 해석 가능하다."},
        {"type": "h1", "text": "13. CPU/GPU 실행"},
        {"type": "para", "text": "PySCF DFT와 NPZ 생성은 CPU 작업이다. DEVICE=gpu를 줘도 데이터 생성은 GPU로 가지 않는다. GPU는 TensorFlow training 단계에서만 사용된다."},
        {"type": "code", "text": "CPU: DEVICE=cpu bash scripts/run_qm9_ldavwn_weekend_500.sh\nGPU: DEVICE=gpu GPU_IDS=0 GPU_MEMORY_GROWTH=1 bash scripts/run_qm9_ldavwn_weekend_500.sh"},
        {"type": "para", "text": "GPU_IDS는 내부에서 CUDA_VISIBLE_DEVICES로 전달된다. 실행 로그의 TensorFlow runtime block에서 visible GPUs가 none이면 TensorFlow/CUDA 환경이 GPU를 못 보는 상태다."},
        {"type": "h1", "text": "14. Output rotation and resume behavior"},
        {"type": "para", "text": "고정 output directory를 사용할 때 기존 결과는 자동으로 회전시킬 수 있다."},
        {"type": "code", "text": "OUTPUT_DIR=result\nTIMESTAMP_OUTPUT=0\nROTATE_OUTPUT_DIR=1\nOUTPUT_ROTATION_DEPTH=2"},
        {"type": "para", "text": "이 설정이면 old_old_result는 삭제되고, old_result는 old_old_result로, result는 old_result로 이동한 뒤 새 result가 만들어진다. NPZ dataset은 별도 폴더이므로 이 rotation의 영향을 받지 않는다."},
        {"type": "h1", "text": "15. Current one-command pipeline"},
        {"type": "code", "text": "DOWNLOAD_QM9=1 \\\nQM9_TAR_URL=https://ndownloader.figshare.com/files/3195389 \\\nOMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \\\nDEVICE=cpu \\\nNPZ_DIR=qmugs_npz/qm9_pyscf_ldavwn_b631gd_atoms10_400_50_50_spacing1p5_kp \\\nOUTPUT_DIR=result TIMESTAMP_OUTPUT=0 ROTATE_OUTPUT_DIR=1 OUTPUT_ROTATION_DEPTH=2 \\\nNUM_SYSTEMS=500 TRAIN_SYSTEM_COUNT=400 VAL_SYSTEM_COUNT=50 TEST_SYSTEM_COUNT=50 \\\nMAX_ATOMS=10 BASIS='6-31g(d)' XC='lda,vwn' \\\nGRID_SPACING_BOHR=1.5 MAX_AXIS_POINTS=21 \\\nEPOCHS=700 BATCH_SIZE=1024 STEPS_PER_EPOCH=80 GAMMA_CACHE_GB=1.0 \\\nbash scripts/run_qm9_ldavwn_weekend_500.sh 2>&1 | tee run.log"},
        {"type": "h1", "text": "16. Practical diagnostics"},
        {"type": "bullets", "items": [
            "tarfile.ReadError 또는 empty tar: QM9 tar가 0 byte 또는 깨진 것. ndownloader.figshare.com URL을 사용한다.",
            "No module named pyscf: 데이터 생성 환경에 PySCF가 없다. python -m pip install --prefer-binary pyscf.",
            "No fixed python path: script는 현재 conda env의 python을 자동으로 사용한다. 필요하면 PYTHON=$(which python) 지정.",
            "NPZ KeyError: 기존 NPZ가 구버전/손상 파일일 수 있다. 현재 builder는 필수 key 검사 후 자동 repair한다.",
            "GPU가 안 잡힘: DEVICE=gpu만으로 충분하지 않고, TensorFlow가 CUDA driver/runtime을 볼 수 있어야 한다.",
        ]},
        {"type": "h1", "text": "17. Current limits"},
        {"type": "bullets", "items": [
            "PySCF 데이터 생성은 CPU 병목이다. GPU는 학습 가속에만 도움된다.",
            "gamma_matrix 저장 비용은 n_grid^2이고, n_grid 자체가 axis_points^3이므로 큰 grid에서는 디스크/메모리 비용이 빠르게 커진다.",
            "KP reference는 LDA/VWN local KS potential 기반이다. hybrid/nonlocal functional에서는 같은 해석을 그대로 쓰기 어렵다.",
            "현재 KP head는 point model output으로 직접 예측하며, gamma에서 variational derivative를 직접 계산하는 구조는 아직 아니다.",
            "더 큰 데이터/더 좋은 split에서 generalization을 확인해야 한다.",
        ]},
        {"type": "para", "text": "요약하면, 현재 코드는 QM9 raw 구조에서 PySCF로 DFT 1-RDM target을 만들고, point/pair/context neural surrogate로 gamma(r,r')를 예측하며, gamma 중심 loss에 KP/T auxiliary loss를 staged schedule로 추가하는 연구 prototype이다."},
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
