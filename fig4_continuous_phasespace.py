import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import minimize_scalar
import matplotlib as mpl

# ==========================================
# 0. Formatting (PRE Column Format)
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
# 1. Physical Parameters
# ==========================================
kBT = 1.0           
kappa = 1.0         
gamma = 1.0         

# Thermodynamic limits for the Thermostat Regime
c_crit_limit = (kBT**2) / (2 * kappa) 
v_max_theoretical = np.sqrt((kappa * kBT) / gamma**2)

# ==========================================
# 2. Setup the 2D Phase Space Grid
# ==========================================
v_vals = np.linspace(0, v_max_theoretical * 1.05, 400)
c_vals = np.linspace(0, c_crit_limit * 1.05, 400)

V_grid, c_grid = np.meshgrid(v_vals, c_vals)

# ==========================================
# 3. Compute the Exact Steady-State Arrays
# ==========================================
Sigma_opt_1D = np.zeros(len(c_vals))
P_fluct_1D = np.zeros(len(c_vals))
C_rate_1D = np.zeros(len(c_vals))

for i, c in enumerate(c_vals):
    if c >= c_crit_limit:
        Sigma_opt_1D[i] = kBT / kappa
        P_fluct_1D[i] = 0.0
        C_rate_1D[i] = 0.0
    elif c == 0.0:
        # Zero cost yields infinite precision and zero variance
        Sigma_opt_1D[i] = 0.0
        dot_Sigma_meas = 2 * kBT / gamma
        P_fluct_1D[i] = (kappa / 2.0) * dot_Sigma_meas
        C_rate_1D[i] = 0.0
    else:
        # Exact cubic root via Cardano
        term1 = (2 * c * kBT) / (kappa**2)
        term2 = np.sqrt((4 * c**2 * (kBT)**2) / (kappa**4) + (8 * c**3) / (27 * kappa**3))
        Sigma = np.cbrt(term1 + term2) + np.cbrt(term1 - term2)
        
        Sigma_opt_1D[i] = Sigma
        
        # Power definitions based on continuous variance reduction rate
        dot_Sigma_meas = (2 * kBT / gamma) - (2 * kappa / gamma) * Sigma
        P_fluct_1D[i] = (kappa / 2.0) * dot_Sigma_meas
        C_rate_1D[i] = (c / Sigma**2) * dot_Sigma_meas

Net_Profit_1D = P_fluct_1D - C_rate_1D

# ==========================================
# 4. Apply the Strict Profitability Constraint
# ==========================================
P_drag_grid = gamma * V_grid**2

Net_Profit_2D = Net_Profit_1D[:, np.newaxis]
Net_Profit_2D_full = np.broadcast_to(Net_Profit_2D, V_grid.shape)

# Continuous 'Precision Effort'
Precision_Effort_1D = np.zeros(len(c_vals))
for i, c in enumerate(c_vals):
    if c == 0.0:
        # Use NaN instead of inf to prevent LogNorm upper-bound overflow
        Precision_Effort_1D[i] = np.nan
    elif c < c_crit_limit:
        Precision_Effort_1D[i] = ((2 * kBT / gamma) - (2 * kappa / gamma) * Sigma_opt_1D[i]) / (Sigma_opt_1D[i]**2)

Precision_Effort_2D = np.broadcast_to(Precision_Effort_1D[:, np.newaxis], V_grid.shape)

# Active Engine Mask (Filter out NaNs and strictly bound > 0)
active_engine_mask = (Net_Profit_2D_full >= P_drag_grid) & (Net_Profit_2D_full > 0) & ~np.isnan(Precision_Effort_2D)
Effort_map_masked = np.ma.masked_where(~active_engine_mask, Precision_Effort_2D)

# Ratio Calculation
epsilon = 1e-10 
P_drag_safe = gamma * (V_grid + epsilon)**2
P_ratio = Net_Profit_2D_full / P_drag_safe

# ==========================================
# 5. The Exact Analytical Boundary Curve c_crit(v)
# ==========================================
nu = (v_vals / v_max_theoretical)**2
valid_idx = nu <= 1.0
s_star = np.zeros_like(v_vals)

s_star[valid_idx] = (4 - nu[valid_idx] - np.sqrt(nu[valid_idx]**2 + 8 * nu[valid_idx])) / 4.0

c_crit_curve = np.zeros_like(v_vals)
c_crit_curve[valid_idx] = c_crit_limit * (s_star[valid_idx]**3) / (2 - s_star[valid_idx])

# ==========================================
# 6. Plotting the Side-by-Side Heatmaps
# ==========================================
fig, ax = plt.subplots(2, 1, figsize=(3.375, 4.5), sharex=True)

# --- TOP PLOT: Precision Effort ---
ax[0].set_facecolor('black')

# Safely extract finite min/max for the LogNorm bounds
if Effort_map_masked.count() > 0:
    vmin_safe = max(float(Effort_map_masked.min()), 1e-8)
    vmax_safe = float(Effort_map_masked.max())
else:
    vmin_safe, vmax_safe = 1e-3, 1.0

# Guarantee vmin < vmax to prevent zero-width LogNorm crashes
if vmin_safe >= vmax_safe:
    vmin_safe = vmax_safe * 0.1

mesh0 = ax[0].pcolormesh(V_grid, c_grid, Effort_map_masked, shading='auto', cmap='magma', 
                         norm=LogNorm(vmin=vmin_safe, vmax=vmax_safe))

ax[0].plot(v_vals[valid_idx], c_crit_curve[valid_idx], color='white', linewidth=1.2, linestyle='--')

cbar0 = plt.colorbar(mesh0, ax=ax[0])
cbar0.set_label(r'Precision Effort ($\dot{\Sigma}_{meas} / \Sigma^2$) [Log]')
cbar0.ax.yaxis.set_label_coords(4.0, 0.5)

ax[0].set_title('Continuous Precision Effort')
ax[0].set_ylabel('Cost Scalar ($c$)')
ax[0].text(0.05, c_crit_limit * 0.05, 'Active Engine', color='black', fontweight='bold')
ax[0].text(v_max_theoretical * 0.55, c_crit_limit * 0.5, 'Bankrupt', color='white', fontweight='bold')

# --- BOTTOM PLOT: Thermodynamic Viability Ratio ---
mesh1 = ax[1].pcolormesh(V_grid, c_grid, P_ratio, shading='auto', cmap='viridis', 
                         vmin=0, vmax=5.0) 

ax[1].plot(v_vals[valid_idx], c_crit_curve[valid_idx], color='white', linewidth=1.2, linestyle='--')

cbar1 = plt.colorbar(mesh1, ax=ax[1], extend='max') 
cbar1.set_label(r'Power Ratio ($\mathcal{P}_{fluct} / \mathcal{P}_{drag}$)')
cbar1.ax.yaxis.set_label_coords(4.0, 0.5)

ax[1].set_title('Thermodynamic Viability Ratio')
ax[1].set_xlabel('Macroscopic Velocity ($v$)')
ax[1].set_ylabel('Cost Scalar ($c$)')

ax[1].text(0.05, c_crit_limit * 0.05, r'Ratio $\gg 1$', color='black', fontweight='bold')
ax[1].text(v_max_theoretical * 0.55, c_crit_limit * 0.5, r'Ratio $< 1$', color='white', fontweight='bold')

for a in ax:
    a.set_xlim(0, v_max_theoretical)
    a.set_ylim(0, c_crit_limit * 1.05)

plt.tight_layout()
plt.savefig('./Fig4.pdf', dpi=600, bbox_inches='tight')
plt.show()
