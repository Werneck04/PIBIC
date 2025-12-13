import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd, norm as l2norm
import warnings

#  1. Parâmetros da Simulação

M = 16         # Número de APs
K = 3          # Número de UEs
L = 1          # Número de alvos
N_rx = 8       # Antenas por AP
N_tx = 4       # Antenas por UE

# Geometria
AREA_SIZE = 100  # Área de 100x100 m^2
H_AP = 11.5    # Altura do AP [m]
H_UE = 1.5     # Altura do UE [m]
H_TGT = 0.0    # Altura do alvo [m]
AP_GRID_N = int(np.sqrt(M)) # APs em uma grade 4x4
AP_SPACING = AREA_SIZE / (AP_GRID_N - 1) if AP_GRID_N > 1 else 0
M_SENSING = 4  # APs mais próximos usados para sensoriamento

# Parâmetros de RF e Potência
F_C = 1.9e9      # Frequência da portadora (1.9 GHz)
BW = 20e6        # Largura de banda (20 MHz)
P_TOTAL_DBM = 30 # Potência total do UE [dBm]
P_TOTAL_W = 10**((P_TOTAL_DBM - 30) / 10) # Potência total em Watts
NOISE_FIGURE_DB = 5 # Figura de ruído [dB]
TEMP_K = 298       # Temperatura [K]
BOLTZMANN_K = 1.380649e-23

# Calibração de RCS (sensoriamento)
RCS_GAIN_DB = 92.0
RCS_GAIN_LINEAR = 10**(RCS_GAIN_DB / 10)
SIGMA_RCS_2 = 1.0

# Cálculo da Potência de Ruído
NOISE_FIGURE = 10**(NOISE_FIGURE_DB / 10)
NOISE_POWER_W = BOLTZMANN_K * TEMP_K * BW * NOISE_FIGURE
SIGMA_N2 = NOISE_POWER_W

# Parâmetros do Canal e Simulação
SHADOWING_STD_DB = 12.0 # Desvio padrão do sombreamento [dB]
N_REALIZATIONS = 1000  # 1000 realizações de Monte Carlo

# Distribuições de potência
POWER_SPLITS = {
    "90/10": (0.9, 0.1),
    "50/50": (0.5, 0.5),
    "10/90": (0.1, 0.9)
}

# 2. Funções Auxiliares (Modelos Físicos)

def calculate_3d_distance(pos1, pos2):
    return np.sqrt(np.sum((np.array(pos1) - np.array(pos2))**2))

def get_angles(tx_pos, rx_pos):
    tx_pos, rx_pos = np.array(tx_pos), np.array(rx_pos)
    delta = rx_pos - tx_pos
    dist_xy = np.sqrt(delta[0]**2 + delta[1]**2)
    theta = 0.0 if dist_xy == 0 else np.arctan2(delta[1], delta[0])
    dist_3d = l2norm(delta)
    if dist_3d == 0 or np.isclose(dist_3d, 0):
        phi = 0.0
    else:
        asin_arg = np.clip(delta[2] / dist_3d, -1.0, 1.0)
        phi = np.arcsin(asin_arg)
    return theta, phi

def get_steering_vector(theta, phi, N_antennas):
    n = np.arange(N_antennas)
    spatial_freq = np.sin(phi) * np.cos(theta)
    vec = (1 / np.sqrt(N_antennas)) * np.exp(-1j * np.pi * n * spatial_freq)
    return vec.reshape(N_antennas, 1)

def get_path_loss_shadowing(dist_3d, h_1, h_2):
    dist_2d = np.sqrt(max(0, dist_3d**2 - (h_1 - h_2)**2))
    f_ghz = F_C / 1e9
    if dist_2d < 10:
        dist_2d = 10
    pl_db = 28.0 + 22 * np.log10(dist_2d) + 20 * np.log10(f_ghz)
    shadowing = np.random.normal(0, SHADOWING_STD_DB)
    pl_total_db = pl_db + shadowing
    beta = 10**(-pl_total_db / 10)
    return beta

#  3. Funções de Beamforming (Sensoriamento)

def get_svd_beamforming(H_effective):
    if H_effective is None or H_effective.shape[0] == 0:
        w_bf = np.zeros((N_tx, 1), dtype=complex)
        w_bf[0] = 1.0
        return w_bf
    try:
        U, S, Vh = svd(H_effective)
        w_bf = Vh[0, :].conj().reshape(N_tx, 1)
        return w_bf / l2norm(w_bf)
    except np.linalg.LinAlgError:
        w_bf = np.zeros((N_tx, 1), dtype=complex)
        w_bf[0] = 1.0
        return w_bf

def get_angular_beamforming_sensing(ue_pos, sensing_cluster_center):
    aod_theta, aod_phi = get_angles(ue_pos, sensing_cluster_center)
    w_bf = get_steering_vector(aod_theta, aod_phi, N_tx)
    return w_bf / l2norm(w_bf)

# 4. Função de Cálculo de SINR
def calculate_sinr(k_user, all_H, all_w_signal, rho_signal_vec, all_w_interf=None, rho_interf_vec=None):
    H_k = all_H[k_user]
    w_sig = all_w_signal[k_user]
    rho_sig = rho_signal_vec[k_user]
    if H_k is None or w_sig is None:
        return 1e-20
    h_desired = H_k @ w_sig
    signal_power = np.abs(l2norm(h_desired))**2 * rho_sig
    if signal_power <= 0:
        return 1e-20
    interference_power = 0.0
    w_int_list = all_w_interf if all_w_interf is not None else all_w_signal
    rho_int_vec = rho_interf_vec if rho_interf_vec is not None else rho_signal_vec
    for j_user in range(K):
        if j_user != k_user and w_int_list[j_user] is not None:
            H_j = all_H[j_user]
            if H_j is not None:
                h_int = H_k @ w_int_list[j_user]
                rho_j = rho_int_vec[j_user]
                interference_power += np.abs(l2norm(h_int))**2 * rho_j
    noise_power = (l2norm(w_sig)**2) * SIGMA_N2
    denominator = interference_power + noise_power
    if denominator <= 0:
        return 1e-20
    return signal_power / denominator

#  5. Loop Principal da Simulação

print(f"Iniciando a simulação de Monte Carlo (Sensoriamento)...")
print(f"Parâmetros: M={M}, K={K}, N_rx={N_rx}, N_tx={N_tx}, Área={AREA_SIZE}m, Realizações={N_REALIZATIONS}")
print(f"Sensoriamento: Canal LoS via alvo (RCS gain={RCS_GAIN_DB} dB).")
print(f"Alocações de potência: {list(POWER_SPLITS.keys())}")

warnings.filterwarnings('ignore', category=RuntimeWarning)

#  PASSO 0: Gerar posições dos APs (fixas)
ap_coords = []
for i in range(AP_GRID_N):
    for j in range(AP_GRID_N):
        ap_coords.append([i * AP_SPACING, j * AP_SPACING, H_AP])
ap_positions = np.array(ap_coords)

# Dicionário para armazenar resultados de cada alocação
results = {}

# PASSO 1: Loop de Monte Carlo
for n in range(N_REALIZATIONS):
    if (n + 1) % 100 == 0:
        print(f"  Realização {n+1}/{N_REALIZATIONS}")

    # Geração de cenário para esta realização
    ue_positions = np.zeros((K, 3))
    ue_positions[:, 0] = np.random.rand(K) * AREA_SIZE
    ue_positions[:, 1] = np.random.rand(K) * AREA_SIZE
    ue_positions[:, 2] = H_UE

    target_pos = np.zeros(3)
    target_pos[0] = np.random.rand() * AREA_SIZE
    target_pos[1] = np.random.rand() * AREA_SIZE
    target_pos[2] = H_TGT

    # Cluster de sensoriamento: APs mais próximos do alvo
    ap_dist_to_target = [calculate_3d_distance(ap, target_pos) for ap in ap_positions]
    sensing_ap_indices = np.argsort(ap_dist_to_target)[:M_SENSING]
    sensing_ap_positions = ap_positions[sensing_ap_indices]
    sensing_cluster_center = np.mean(sensing_ap_positions, axis=0) if M_SENSING > 0 else target_pos

    # Geração dos canais de sensoriamento (UE → alvo → APs)
    H_s_to_sens = {}
    for k in range(K):
        ue_pos = ue_positions[k]
        H_k_s_to_sens_stack = []

        alpha_l = (np.random.normal(0, np.sqrt(SIGMA_RCS_2 / 2)) + 
                   1j * np.random.normal(0, np.sqrt(SIGMA_RCS_2 / 2)))

        # Sensoriamento: UE -> alvo -> APs de sensoriamento (LoS)
        dist_ue_tgt = calculate_3d_distance(ue_pos, target_pos)
        beta_1 = get_path_loss_shadowing(dist_ue_tgt, H_UE, H_TGT)
        aod_theta, aod_phi = get_angles(ue_pos, target_pos)
        a_tx = get_steering_vector(aod_theta, aod_phi, N_tx)

        for m_idx in sensing_ap_indices:
            ap_pos = ap_positions[m_idx]
            dist_tgt_ap = calculate_3d_distance(target_pos, ap_pos)
            beta_2 = get_path_loss_shadowing(dist_tgt_ap, H_TGT, H_AP)
            beta_total = beta_1 * beta_2 * RCS_GAIN_LINEAR
            aoa_theta, aoa_phi = get_angles(target_pos, ap_pos)
            a_rx = get_steering_vector(aoa_theta, aoa_phi, N_rx)
            H_km_s = alpha_l * np.sqrt(beta_total) * (a_rx @ a_tx.conj().T)
            H_k_s_to_sens_stack.append(H_km_s)

        H_s_to_sens[k] = np.vstack(H_k_s_to_sens_stack) if H_k_s_to_sens_stack else None

    # Calcular beamformers para esta realização
    all_w_s_svd = [None]*K
    all_w_s_ang = [None]*K
    for k in range(K):
        ue_pos = ue_positions[k]
        all_w_s_svd[k] = get_svd_beamforming(H_s_to_sens[k])
        all_w_s_ang[k] = get_angular_beamforming_sensing(ue_pos, sensing_cluster_center)

    # Para cada alocação de potência, calcular SINR de sensoriamento
    for split_name, (p_c, p_s) in POWER_SPLITS.items():
        if split_name not in results:
            results[split_name] = {"svd_s": [], "ang_s": []}

        rho_s = P_TOTAL_W * p_s  # Potência de sinal sensoriamento
        rho_s_vec = np.full(K, rho_s)
        # Interferência sensoriamento escala com p_c (comunicação) para afetar sensing
        rho_int_sens = np.full(K, P_TOTAL_W * max(1e-6, p_c))

        for k in range(K):
            # Sensoriamento com SVD BF
            sinr_s_svd = calculate_sinr(k, H_s_to_sens, all_w_s_svd, rho_s_vec, 
                                       all_w_interf=all_w_s_svd, rho_interf_vec=rho_int_sens)
            results[split_name]["svd_s"].append(10 * np.log10(sinr_s_svd) if sinr_s_svd > 0 else -200.0)

            # Sensoriamento com Angular BF
            sinr_s_ang = calculate_sinr(k, H_s_to_sens, all_w_s_ang, rho_s_vec, 
                                       all_w_interf=all_w_s_ang, rho_interf_vec=rho_int_sens)
            results[split_name]["ang_s"].append(10 * np.log10(sinr_s_ang) if sinr_s_ang > 0 else -200.0)

print("Simulação concluída.")
warnings.filterwarnings('default', category=RuntimeWarning)

# Impressão de estatísticas (antes do gráfico)
print("\n=== Estatísticas de SINR (Sensoriamento) por Alocação de Potência ===")
for split_name, data in results.items():
    print(f"\n{split_name}:")
    print(f"  SVD BF - Média: {np.mean(data['svd_s']):.2f} dB, Mediana: {np.median(data['svd_s']):.2f} dB")
    print(f"  Ang. BF - Média: {np.mean(data['ang_s']):.2f} dB, Mediana: {np.median(data['ang_s']):.2f} dB")

#  6. Geração dos Gráficos

def plot_ecdf(ax, data, label, linestyle='-', color=None):
    sorted_data = np.sort(data)
    yvals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax.plot(sorted_data, yvals, label=label, linestyle=linestyle, color=color)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
titles = [
    "(a) 10%-sensoriamento",
    "(b) 50%-sensoriamento",
    "(c) 90%-sensoriamento"
]

plot_keys = list(POWER_SPLITS.keys())

for i in range(3):
    ax = axes[i]
    split_name = plot_keys[i]
    title = titles[i]
    res = results[split_name]

    color_svd_s = '#3A9D3A' # Verde escuro
    color_ang_s = '#7AFB7A' # Verde claro

    plot_ecdf(ax, res["svd_s"], "SVD BF (sens)", linestyle='-', color=color_svd_s)
    plot_ecdf(ax, res["ang_s"], "Ang. BF (sens)", linestyle='--', color=color_ang_s)

    ax.set_title(title)
    ax.set_xlabel("SINR [dB]")
    ax.set_ylabel("eCDF")
    ax.set_xlim([-40, 80])
    ax.set_ylim([0, 1])
    ax.grid(True)
    ax.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
script_dir = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(script_dir, "figura_sensoriamento_multiplas_alocacoes.png")
plt.savefig(out_path)
plt.show()
print(f"Gráfico salvo em: {out_path}")
plt.close()
