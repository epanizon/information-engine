import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import brentq

# ==========================================
# 0. Configure Global Font Sizes 
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
# 1. Physics & Thermodynamics Parameters
# ==========================================
kT = 1.0
kappa = 1.0
gamma = 1.0
dt = 0.01 
t_f = int(20 / dt)
lambda_f = 1.0
fraction_of_C_crit = 0.6

# Reset variance. 
sigma_min = 0.00000000

# System constants
alpha = np.exp(-kappa * dt / gamma)
alpha_bar = 1.0 - alpha
sigma_eq = (kT / kappa) * (1 - alpha**2)
Sigma_max = (kT / kappa)

C_crit = 0.5 * kT 
C = C_crit * fraction_of_C_crit      

# ==========================================
# 2. Backward Induction: Precompute Policy 
# ==========================================
P = 0.0  # Base case for the final step target
A_seq = np.zeros(t_f + 1)
K_seq = np.zeros(t_f + 1)      # Feedback gains for the control law

# The 1D Riccati Recurrence
for n in range(1, t_f + 1):
    # Calculate intermediate terms
    B = alpha_bar * (P * alpha - kappa)
    A = alpha_bar * (P * alpha_bar + 2 * kappa)
    
    # The optimal control gain
    K_seq[n] = -B / A
    
    # Update the variance penalty
    P_next = P * alpha**2 - (B**2) / A
    
    P = P_next
    A_seq[n] = -0.5 * P

# 1D Variance DP (Measurement Thresholds) via Exact Root Finding
thresholds = np.zeros(t_f + 1)
g_funcs = []

# Base case: At the deadline, informational cost is zero
g_funcs.append(lambda sigma: 0.0)

# 
def build_step(A_n, g_prev):
    # The baseline future cost of measuring is constant for this step
    next_prior_after_meas = sigma_min * (alpha**2) + sigma_eq
    g_min_evolved = g_prev(next_prior_after_meas)
    
    def Q_unmeas(sigma):
        return g_prev(sigma * (alpha**2) + sigma_eq)
        
    def Q_meas(sigma):
        return C - A_n * sigma + g_min_evolved
        
    def diff(sigma):
        return Q_meas(sigma) - Q_unmeas(sigma)
        
    # Search for the root (threshold) up to a safe physical upper bound
    sigma_upper_bound = (kT / kappa) * 10.0 
    
    if diff(0.0) <= 0:
        th = 0.0  # Measure regardless of certainty
    elif diff(sigma_upper_bound) > 0:
        th = np.inf  # Never measure
    else:
        # Find exact root to machine precision
        th = brentq(diff, 0.0, sigma_upper_bound)
        
    # Create the piecewise value function for the next backward step
    def g_current(sigma):
        return Q_meas(sigma) if sigma >= th else Q_unmeas(sigma)
        
    return th, g_current

for n in range(1, t_f + 1):
    th, g_new = build_step(A_seq[n], g_funcs[n-1])
    thresholds[n] = th
    g_funcs.append(g_new)

# ==========================================
# 3. Forward Simulation: An Individual Trajectory
# ==========================================
np.random.seed(42)  # For reproducible fluctuations

# Storage arrays
x = np.zeros(t_f + 1)       # True physical position
mu = np.zeros(t_f + 1)      # Agent's estimated mean
var = np.zeros(t_f + 1)     # Agent's estimated variance
lam = np.zeros(t_f + 1)     # Trap position
measured = np.zeros(t_f, dtype=bool)

# Initial conditions
x[0] = np.random.normal(0, np.sqrt(Sigma_max))
mu[0] = 0.0
var[0] = Sigma_max  # Assume we start with equilibrium uncertainty
lam_prev = 0.0
total_work = 0.0
measure_count = 0

print("Simulating forward trajectory...")

for t in range(t_f):
    n = t_f - t  # Steps remaining
    
    # --- PHASE 1: Observe ---
    if var[t] > thresholds[n]:
        measured[t] = True
        measure_count += 1
        # Measurement done. 
        mu_plus = x[t]
        var_plus = sigma_min
    else:
        measured[t] = False
        mu_plus = mu[t]
        var_plus = var[t]
        
    # --- PHASE 2: Act (Control) ---
    # Optimal trap control
    lam[t] = K_seq[n] * mu_plus + (1 - K_seq[n]) * lambda_f
    
    # Calculate physical work done on the particle during the jump
    jump_work = -kappa * (lam[t] - lam_prev) * (x[t] - (lam[t] + lam_prev) / 2.0)
    total_work += jump_work
    lam_prev = lam[t]
    
    # --- PHASE 3: Evolve (Physics) ---
    # True position evolves via Exact Ornstein-Uhlenbeck update
    noise = np.random.normal(0, np.sqrt(sigma_eq))
    x[t+1] = x[t] * alpha + lam[t] * alpha_bar + noise
    
    # Belief state evolves analytically
    mu[t+1] = mu_plus * alpha + lam[t] * alpha_bar
    var[t+1] = var_plus * alpha**2 + sigma_eq

# Snap the final trap position to the target to complete the protocol
lam[t_f] = lambda_f
final_jump_work = -kappa * (lam[t_f] - lam_prev) * (x[t_f] - (lam[t_f] + lam_prev) / 2.0)
total_work += final_jump_work

print("Simulation Complete.")
print(f"Total Measurements Taken: {measure_count}")
print(f"Total Thermodynamic Work Done: {total_work/kT:.4f}")

# ==========================================
# 4. Visualization
# ==========================================
time_steps = np.arange(t_f + 1)
std_dev = np.sqrt(var)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.375, 4.0), sharex=True, 
                               gridspec_kw={'height_ratios': [2.2, 1]})

# --- Top Plot: Trajectories ---
ax1.plot(time_steps, x, color='gray', alpha=0.5, linewidth=0.8, label='True Position ($x$)')
ax1.plot(time_steps, mu, color='blue', linestyle='--', linewidth=1.2, label=r'Belief Mean ($\mu$)')
ax1.fill_between(time_steps, mu - std_dev, mu + std_dev, color='blue', alpha=0.2, edgecolor='none', label=r'Uncertainty ($\mu \pm \sigma$)')
ax1.step(time_steps, lam, where='post', color='red', linewidth=1.2, label=r'Trap ($\lambda$)')

# Highlight measurement events
meas_times = np.where(measured)[0]
if len(meas_times) > 0:
    ax1.scatter(meas_times, x[meas_times], color='black', zorder=5, marker='o', s=12, label='Measurement')

ax1.set_ylabel("Position")
ax1.set_title("Maxwell Demon with Binary Measurements")
ax1.legend(loc='upper left', framealpha=0.9, handlelength=1.5, labelspacing=0.3, borderpad=0.3)
ax1.grid(True, alpha=0.3, linewidth=0.5)

# --- Bottom Plot: Variance and Thresholds ---
ax2.plot(time_steps, var, color='purple', linewidth=1.2, label=r'Variance ($\Sigma$)')
ax2.plot(time_steps[:-1], thresholds[1:][::-1], color='orange', linestyle='--', linewidth=1.2, label=r'Threshold ($\Sigma_{th}$)')
ax2.axhline((kT / kappa), color='black', linestyle=':', alpha=0.5, linewidth=1.0, label=r'Eq. Limit ($\Sigma_{\kappa}$)')

ax2.set_xlabel("Time Step (k)")
ax2.set_ylabel("Variance")
ax2.set_ylim(sigma_eq * 0.5, (kT/kappa) * 2.7)
ax2.legend(loc='upper left', framealpha=0.95, handlelength=1.5, labelspacing=0.3, borderpad=0.3)
ax2.grid(True, alpha=0.3, linewidth=0.5)

# Deadline Blindness Shading
blind_idx = np.where(thresholds[1:][::-1] > 1)[0]
if len(blind_idx) > 0:
    blind_step = time_steps[blind_idx[0]]
    for ax in [ax1, ax2]:
        ax.axvspan(blind_step, t_f, color='gray', alpha=0.15, linewidth=0)

plt.tight_layout()
plt.savefig('./Fig1.png', dpi=600, bbox_inches='tight')
#plt.show()
