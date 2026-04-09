import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import minimize_scalar
import matplotlib.ticker as ticker

# ==========================================
# 1. Formatting and Global Parameters
# ==========================================
np.random.seed(42)  
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

# Physical Parameters
kBT, kappa, gamma, dt, t_f = 1.0, 1.0, 1.0, 0.01, 20.0
alpha = np.exp(-kappa * dt / gamma)
alpha_bar = 1.0 - alpha
Sigma_eq = (kBT / kappa) * (1 - alpha**2)
Sigma_max = kBT / kappa  
N_total = int(t_f / dt)
time_steps = np.arange(N_total)

# Boundary Conditions
x0 = np.random.normal(0, np.sqrt(Sigma_max))
lam_f = 3.0

A_inf = kappa / 2.0
target_fraction = 0.60
c_coeff = A_inf * (target_fraction * Sigma_max)**2

# ==========================================
# 1b. Steady-State Thermostat Target
# ==========================================
def steady_state_profit(S):
    sig_minus = alpha**2 * S + Sigma_eq
    profit = A_inf * (sig_minus - S) - c_coeff * (1.0 / S - 1.0 / sig_minus)
    return -profit

res = minimize_scalar(steady_state_profit, bounds=(1e-6, Sigma_max), method='bounded')
true_steady_state_target = res.x
greedy_steady_state_target = np.sqrt(c_coeff / A_inf)

print("--- Steady-State Analytics ---")
print(f"Analytical Greedy Asymptote: {greedy_steady_state_target:.4f}")
print(f"Exact DP Asymptote:          {true_steady_state_target:.4f}\n")

# ==========================================
# 2. Backward Induction: Precompute Policy
# ==========================================
P = 0.0  # Base case for the final step target
A_n = np.zeros(N_total + 1)
K_seq = np.zeros(N_total + 1)      # Feedback gains for the control law

# The 1D Riccati Recurrence
for n in range(1, N_total + 1):
    # Calculate intermediate terms
    B = alpha_bar * (P * alpha - kappa)
    A = alpha_bar * (P * alpha_bar + 2 * kappa)
    
    # The optimal control gain
    K_seq[n] = -B / A
    
    # Update the variance penalty
    P_next = P * alpha**2 - (B**2) / A
    
    P = P_next
    A_n[n] = -0.5 * P

# ==========================================
# 3. DP Continuous-State Value Function
# ==========================================
true_sigma_opt = np.zeros(N_total + 1)
true_sigma_opt[0] = np.inf

print("Solving DP Continuous-State Value Function backward in time...")

# Store the value functions recursively
g_funcs = []
g_funcs.append(lambda sigma: 0.0)  # Base case: deadline cost is 0

# Factory function to properly scope the backward recursion
def build_continuous_step(A_val, g_prev):
    
    # The objective is to minimize g
    def g_target(S):
        # (A * S^+ + c/S^+ - g_evolved(S^+) 
        return A_val * S + (c_coeff / S) - g_prev(alpha**2 * S + Sigma_eq)
        
    # Search for the optimal target variance continuously 
    res = minimize_scalar(g_target, bounds=(1e-6, Sigma_max * 5.0), method='bounded')
    target_sigma = res.x
    
    # Pre-evaluate the evolved function at the target to optimize the closure execution
    g_evolved_at_target = g_prev(alpha**2 * target_sigma + Sigma_eq)
    
    # The piecewise value function for the prior state
    def g_current(sigma):
        if sigma <= target_sigma:
            # Below target: measurement is not profitable, state diffuses naturally
            return g_prev(alpha**2 * sigma + Sigma_eq)
        else:
            # Above target: system applies precise continuous measurement down to target
            return A_val * (sigma - target_sigma) - c_coeff * (1.0/target_sigma - 1.0/sigma) + g_evolved_at_target
            
    return target_sigma, g_current

# Execute DP
for n in range(1, N_total + 1):
    target, g_new = build_continuous_step(A_n[n], g_funcs[n-1])
    true_sigma_opt[n] = target
    g_funcs.append(g_new)

true_target_forward = np.array([true_sigma_opt[N_total - t] for t in range(N_total)])

# ==========================================
# 4. Simulation & Work Calculation
# ==========================================
x, mu_minus, mu_plus, lam = np.zeros(N_total), np.zeros(N_total), np.zeros(N_total), np.zeros(N_total)
var_minus, var_plus, meas_intensity = np.zeros(N_total), np.zeros(N_total), np.zeros(N_total)

x[0], mu_plus[0], var_plus[0] = x0, 0, Sigma_max
lam[0] = K_seq[N_total] * mu_plus[0] + (1 - K_seq[N_total]) * lam_f

work_step = np.zeros(N_total)

for t in range(1, N_total):
    n_rem = N_total - t
    
    x[t] = alpha * x[t-1] + (1 - alpha) * lam[t-1] + np.random.normal(0, np.sqrt(Sigma_eq))
    mu_minus[t] = alpha * mu_plus[t-1] + (1 - alpha) * lam[t-1]
    var_minus[t] = var_plus[t-1] * alpha**2 + Sigma_eq
    
    target = true_sigma_opt[n_rem]
    if var_minus[t] > target and target <= Sigma_max:
        var_plus[t] = target
        L_t = 1 - (var_plus[t] / var_minus[t]) 
        R_t = (var_minus[t] * var_plus[t]) / (var_minus[t] - var_plus[t])
        z_t = x[t] + np.random.normal(0, np.sqrt(R_t))
        mu_plus[t] = mu_minus[t] + L_t * (z_t - mu_minus[t])
        meas_intensity[t] = L_t
    else:
        var_plus[t], mu_plus[t], meas_intensity[t] = var_minus[t], mu_minus[t], 0.0
        
    lam[t] = K_seq[n_rem] * mu_plus[t] + (1 - K_seq[n_rem]) * lam_f
    
    work_step[t] = 0.5 * kappa * (x[t] - lam[t])**2 - 0.5 * kappa * (x[t] - lam[t-1])**2

total_work = np.sum(work_step)

# ==========================================
# 5. Plotting
# ==========================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.875, 4.0), sharex=True, 
                               gridspec_kw={'height_ratios': [2.2, 1]})

# Top Panel
ax1.plot(time_steps, x, color='gray', linewidth=0.8, alpha=0.5, zorder=1, label='True Position ($x$)')
std_dev = np.sqrt(var_plus)
ax1.fill_between(time_steps, mu_plus - std_dev, mu_plus + std_dev, color='blue', alpha=0.2, edgecolor='none', zorder=2, label=r'Uncertainty ($\mu \pm \sigma$)')
ax1.step(time_steps, lam, where='post', color='red', linewidth=1.2, zorder=3, label=r'Trap ($\lambda$)')

points = np.array([time_steps, mu_plus]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

active_intensity = meas_intensity[meas_intensity > 1e-4]
if len(active_intensity) > 1:
    vmax = np.percentile(active_intensity, 95) * 1.5
else:
    vmax = np.max(meas_intensity) * 1.1

norm = plt.Normalize(0, vmax) 
lc = LineCollection(segments, cmap='viridis', norm=norm, zorder=5, linewidth=1.5)
lc.set_array(meas_intensity[:-1])
ax1.add_collection(lc)

ax1.plot([], [], color='blue', linestyle='--', linewidth=1.2, label=r'Belief Mean ($\mu$)')

divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="3%", pad=0.05) 
cbar = plt.colorbar(lc, cax=cax)

formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-2, 2)) 
cbar.ax.yaxis.set_major_formatter(formatter)

plt.draw() 
offset_text = cbar.ax.yaxis.get_offset_text()
offset_text.set_horizontalalignment('center')
offset_text.set_x(4.0)

cbar.set_label(r'Sensor Precision ($L_t$)', rotation=90, labelpad=10)

ax1.set_ylabel("Position")
ax1.set_title('Maxwell Demon with Continuous Sensor')
ax1.legend(loc='lower center', framealpha=0.9, handlelength=1.5, labelspacing=0.3, borderpad=0.3)
ax1.grid(True, color='lightgray', alpha=0.5, linewidth=0.5)

# Bottom Panel
ax2.set_xlim(0, N_total)

ax2.plot(time_steps, var_plus, color='purple', linewidth=1.2, label=r'Variance ($\Sigma$)')
ax2.plot(time_steps, true_target_forward, color='darkorange', linestyle='--', linewidth=1.2, label=r'DP Target ($\Sigma_{opt}^+$)')

ax2.axhline(Sigma_max, color='gray', linestyle=':', linewidth=1.0, label=r'Eq. Limit ($\Sigma_{\kappa}$)')
ax2.axhline(true_steady_state_target, color='mediumseagreen', linestyle='-.', linewidth=1.2, alpha=0.9, label=r'Asymptote ($\Sigma_\infty^*$)')

blind_idx = np.where(true_target_forward[1:] >= kBT/kappa)[0]
if len(blind_idx) > 0:
    blind_step = time_steps[blind_idx[0] + 1] 
    for ax in [ax1, ax2]:
        ax.axvspan(blind_step, N_total, color='gray', alpha=0.15, linewidth=0)

divider2 = make_axes_locatable(ax2)
divider2.append_axes("right", size="3%", pad=0.05).axis('off')

ax2.set_xlabel('Time Step (k)')
ax2.set_ylabel('Variance')
ax2.set_ylim(0, 1.1) 
ax2.legend(loc='lower left', ncol=2, framealpha=0.9, handlelength=1.5)
ax2.grid(True, color='lightgray', alpha=0.5, linewidth=0.5)

plt.tight_layout()
plt.savefig('./Fig3.pdf', dpi=600, bbox_inches='tight')
#plt.show()
