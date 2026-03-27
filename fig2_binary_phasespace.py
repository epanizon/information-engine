import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm

# ==========================================
# 1. Formatting
# ==========================================
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'lines.linewidth': 1.0,
    'figure.titlesize': 10
})

# ==========================================
# 2. Physical Parameters & Analytical Limits
# ==========================================
kBT = 1.0           
kappa = 1.0         
gamma = 1.0         
dt = 0.01           

C_max = 0.5 * kBT 
v_max = np.sqrt(kappa * kBT / gamma**2) 

v_vals = np.linspace(0, v_max * 1.05, 300)
C_vals = np.linspace(0, C_max * 1.05, 300)

V_grid, C_grid = np.meshgrid(v_vals, C_vals)

# Analytical Envelope C_env(v)
v_safe = np.clip(v_vals, 1e-10, None)
v_ratio_sq = (v_safe / v_max)**2
C_env_analytical = C_max * (1 - v_ratio_sq + v_ratio_sq * np.log(v_ratio_sq))
C_env_analytical[0] = C_max # Patch the limit at v=0

# ==========================================
# 3. Compute Measurement Frequency and Max Power
# ==========================================
N_array = np.arange(1, 5000)
t_p_array = N_array * dt
alpha_t = np.exp(-kappa * t_p_array / gamma)

Freq_1D = np.zeros(len(C_vals))
P_max_1D = np.zeros(len(C_vals))

for i, C in enumerate(C_vals):
    if C >= C_max:
        Freq_1D[i] = 0.0
        P_max_1D[i] = 0.0
    else:
        P_fluct_rates = (C_max * (1 - alpha_t**2) - C) / t_p_array
        best_idx = np.argmax(P_fluct_rates)
        
        if P_fluct_rates[best_idx] > 0:
            P_max_1D[i] = P_fluct_rates[best_idx]
            Freq_1D[i] = 1.0 / t_p_array[best_idx]

# ==========================================
# 4. Profitability Constraint
# ==========================================
P_drag_grid = gamma * V_grid**2
P_max_2D = P_max_1D[:, np.newaxis]

Freq_2D_full = np.broadcast_to(Freq_1D[:, np.newaxis], V_grid.shape)

# Masking
active_engine_mask = (P_max_2D >= P_drag_grid) & (Freq_2D_full > 0)
Freq_map_masked = np.ma.masked_where(~active_engine_mask, Freq_2D_full)

# Ratio
epsilon = 1e-10 
P_drag_safe = gamma * (V_grid + epsilon)**2
P_ratio = P_max_2D / P_drag_safe

# ==========================================
# 5. Plotting 
# ==========================================
fig, ax = plt.subplots(2, 1, figsize=(3.375, 4.5), sharex=True)

# --- TOP PLOT: Frequency ---
ax[0].set_facecolor('black')
vmin_safe = Freq_map_masked.min() if Freq_map_masked.count() > 0 else 1e-3
vmax_safe = Freq_map_masked.max() if Freq_map_masked.count() > 0 else 1.0

mesh0 = ax[0].pcolormesh(V_grid, C_grid, Freq_map_masked, shading='auto', cmap='magma', 
                         norm=LogNorm(vmin=vmin_safe, vmax=vmax_safe))
                         
# Plot exact analytical envelope
ax[0].plot(v_vals, C_env_analytical, color='white', linewidth=1.2, linestyle='--')

cbar0 = plt.colorbar(mesh0, ax=ax[0])
cbar0.set_label('Freq. ($f = 1 / t_p^*$) [Log]')
cbar0.ax.yaxis.set_label_coords(4.0, 0.5)

ax[0].set_title('Optimal Measurement Frequency')
ax[0].set_ylabel(r'Cost ($C$)')
ax[0].text(0.02, C_max * 0.05, 'Active Engine', color='black', fontweight='bold')
ax[0].text(v_max * 0.55, C_max * 0.7, 'Bankrupt', color='white', fontweight='bold')

# --- BOTTOM PLOT: Viability Ratio ---
mesh1 = ax[1].pcolormesh(V_grid, C_grid, P_ratio, shading='auto', cmap='viridis', 
                         vmin=0, vmax=5.0) 

# Plot exact analytical envelope
ax[1].plot(v_vals, C_env_analytical, color='white', linewidth=1.2, linestyle='--')

cbar1 = plt.colorbar(mesh1, ax=ax[1], extend='max')
cbar1.set_label(r'Power Ratio ($\mathcal{P}_{fluct} / \mathcal{P}_{drag}$)')
cbar1.ax.yaxis.set_label_coords(4.0, 0.5)

ax[1].set_title('Thermodynamic Viability Ratio')
ax[1].set_xlabel('Macroscopic Velocity ($v$)')
ax[1].set_ylabel('Cost ($C$)')

ax[1].text(0.02, C_max * 0.05, r'Ratio $\gg 1$', color='black', fontweight='bold')
ax[1].text(v_max * 0.55, C_max * 0.7, r'Ratio $< 1$', color='white', fontweight='bold')

for a in ax:
    a.set_xlim(0, v_max)
    a.set_ylim(0, C_max * 1.05)

plt.tight_layout()
plt.savefig('./Fig2.pdf', dpi=600, bbox_inches='tight')
#plt.show()
