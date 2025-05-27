import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'Helvetica'
ngrid = 500
fontsize = 26
markersize = 9
cmap = plt.cm.RdBu

fig = plt.figure(figsize=(8, 6))
ax1 = plt.subplot2grid((1, 1), (0, 0))
ax1.tick_params(labelsize=22)

# Set y-axis ticks 
tick_positions = [2, 2.5, 3, 3.5]
tick_labels = [r'$2.0$', r'$2.5$', r'$3.0$', r'$3.5$']
ax1.set_yticks(tick_positions)
ax1.set_yticklabels(tick_labels)
minor_locator = ticker.AutoMinorLocator(5)
ax1.yaxis.set_minor_locator(minor_locator)

# Set x-axis ticks 
tick_positions = [0, 0.5, 1]
tick_labels = [r'$0$', r'$0.5$', r'$1$']
ax1.set_xticks(tick_positions)
ax1.set_xticklabels(tick_labels)
minor_locator = ticker.AutoMinorLocator(5)
ax1.xaxis.set_minor_locator(minor_locator)

# Adjust margins
left_margin = 0.1   
right_margin = 0.97  
bottom_margin = 0.125 
top_margin = 0.96    
wspace = 0.25
hspace = 0.25
plt.subplots_adjust(left=left_margin, right=right_margin, bottom=bottom_margin, top=top_margin, wspace=wspace, hspace=hspace)

def read_two_column_file(file_name):
    with open(file_name, 'r') as data:
        x1, x2 = [], []
        for line in data:
            p = line.split()
            x1.append(float(p[0]))
            x2.append(float(p[1]))
    return x1, x2

# Model parameters
D = 0.1
rho0 = 1.0
wo = 1.0
alpha = 10.0
q = 1.0  # This q is used in the stability functions below

# Define analytic critical Pe function.
def critical_Pe(won):
    """
    Compute the critical Péclet number:
      Pe_c = ((1+q_c^2)/((1+alpha*wo)*psi0*fpsi))*(1+D + (wo+won)/q_c^2)
    where:
      q_c = ((wo+won)/(1+D))^(1/4),
      psi0 = (won/(wo+won))*rho0, and fpsi = 1/(1+psi0)^2.
    For won very close to zero, returns np.nan.
    """
    if won < 1e-6:
        return np.nan
    psi0 = (won/(wo+won))*rho0
    fpsi = 1/(1+psi0)**2
    q_c = ((wo+won)/(1+D))**(1/4)
    return (1+q_c**2)/((1+alpha*wo)*psi0*fpsi) * (1+D + (wo+won)/(q_c**2))

def function_lambda_real(won, Pe):
    psi0 = (won/(wo+won))*rho0
    phi0 = (wo/(wo+won))*rho0
    A = 1 + D
    B = Pe*(1+alpha*wo)*(psi0/(1+psi0)**2)
    m11 = -(q**2)*(1.0 - (B/(1+q**2))) - wo
    m22 = -(q**2)*D - won
    m12 = won
    m21 = -(q**2)*(-((Pe*(phi0-alpha*wo*psi0)*(1/(1+psi0)**2))/(1+q**2))) + wo
    trace = m11 + m22
    det = m11*m22 - m12*m21
    delta = trace**2 - 4.0 * det
    if delta < 0:
        lambda_real_values = trace/2.0
    else:
        lambda_real_values = np.real((trace + np.sqrt(delta))/2.0)
    return lambda_real_values

def function_lambda_im(won, Pe):
    psi0 = (won/(wo+won))*rho0
    phi0 = (wo/(wo+won))*rho0
    A = 1 + D
    B = Pe*(1+alpha*wo)*(psi0/(1+psi0)**2)
    m11 = -(q**2)*(1.0 - (B/(1+q**2))) - wo
    m22 = -(q**2)*D - won
    m12 = won
    m21 = -(q**2)*(-((Pe*(phi0-alpha*wo*psi0)*(1/(1+psi0)**2))/(1+q**2))) + wo
    trace = m11 + m22
    det = m11*m22 - m12*m21
    delta = trace**2 - 4.0 * det
    if delta < 0:
        lambda_im_values = np.sqrt(-delta)/2.0
    else:
        lambda_im_values = 0
    return lambda_im_values

def function_lambdap_real(won, Pe):
    psi0 = (won/(wo+won))*rho0
    phi0 = (wo/(wo+won))*rho0
    A = 1 + D
    B = Pe*(1+alpha*wo)*(psi0/(1+psi0)**2)
    m11 = -(q**2)*(1.0 - (B/(1+q**2))) - wo
    m22 = -(q**2)*D - won
    m12 = won
    m21 = -(q**2)*(-((Pe*(phi0-alpha*wo*psi0)*(1/(1+psi0)**2))/(1+q**2))) + wo
    p11 = (m11 + m21 + m12 + m22) / 2.0
    p12 = (m11 + m21 - m12 - m22) / 2.0
    p21 = (m11 - m21 + m12 - m22) / 2.0
    p22 = (m11 - m21 - m12 + m22) / 2.0
    tracep = p11 + p22
    detp = p11*p22 - p12*p21
    deltap = tracep**2 - 4.0 * detp
    if deltap < 0:
        lambdap_real_values = tracep/2.0
    else:
        lambdap_real_values = np.real((tracep + np.sqrt(deltap))/2.0)
    return lambdap_real_values

def function_lambdap_im(won, Pe):
    psi0 = (won/(wo+won))*rho0
    phi0 = (wo/(wo+won))*rho0
    A = 1 + D
    B = Pe*(1+alpha*wo)*(psi0/(1+psi0)**2)
    m11 = -(q**2)*(1.0 - (B/(1+q**2))) - wo
    m22 = -(q**2)*D - won
    m12 = won
    m21 = -(q**2)*(-((Pe*(phi0-alpha*wo*psi0)*(1/(1+psi0)**2))/(1+q**2))) + wo
    p11 = (m11 + m21 + m12 + m22) / 2.0
    p12 = (m11 + m21 - m12 - m22) / 2.0
    p21 = (m11 - m21 + m12 - m22) / 2.0
    p22 = (m11 - m21 - m12 + m22) / 2.0
    tracep = p11 + p22
    detp = p11*p22 - p12*p21
    deltap = tracep**2 - 4.0 * detp
    if detp < 0:
        lambdap_im_values = p11
    else:
        lambdap_im_values = 0
    return lambdap_im_values

# Use a won range that avoids zero (start at a small positive value)
won_values = np.linspace(1e-6, 1.1, ngrid)
Pe_values = np.linspace(2, 3.75, ngrid)

lambda_real = np.zeros((len(won_values), len(Pe_values)))
lambda_im = np.zeros((len(won_values), len(Pe_values)))
lambdap_real = np.zeros((len(won_values), len(Pe_values)))
lambdap_im = np.zeros((len(won_values), len(Pe_values)))
for j, won in enumerate(won_values):
    for i, Pe in enumerate(Pe_values):
        lambda_real[i, j] = function_lambda_real(won,Pe)
        lambda_im[i, j] = function_lambda_im(won,Pe)
        lambdap_real[i, j] = function_lambdap_real(won,Pe)
        lambdap_im[i, j] = function_lambdap_im(won,Pe)

ax1.contour(won_values, Pe_values, lambda_real, levels=[0], colors='k', linewidths=3.0)
masked_lambda_im = np.ma.masked_where(lambda_real <= 0, lambda_im)
ax1.contour(won_values, Pe_values, masked_lambda_im, levels=[0], colors='darkred', linewidths=3.0, linestyles='dashed')

ax1.contourf(won_values, Pe_values, lambda_real, levels=[min(lambda_real.flatten()), 0], colors='k', alpha = 0.1, extend='min')

mask = (lambda_real > 0) & (lambda_im > 0)
ax1.contourf(won_values, Pe_values, mask, levels=[0.5, 1], colors='darkred', alpha=0.1)

mask = (lambda_real > 0)
ax1.contourf(won_values, Pe_values, mask, levels=[0.5, 1], colors='darkblue', alpha=0.1)




# --- NEW: Add analytic critical Pe line in the oscillatory regime ---
won_line = []
Pe_crit_line = []
for w in won_values:
    Pe_c = critical_Pe(w)
    # Only include points in the oscillatory regime where the imaginary part is positive.
    if not np.isnan(Pe_c) and function_lambda_im(w, Pe_c) > 0:
        won_line.append(w)
        Pe_crit_line.append(Pe_c)

if won_line:
    ax1.plot(won_line, Pe_crit_line, '--', color='m', linewidth=3) #, label=r'$\mathbf{Pe_c~(Eq.~(10))}$')
# -------------------------------------------------------------------------


ax1.tick_params(which='major', direction='in', bottom=True, top=True, left=True, right=True)
ax1.tick_params(which='minor', direction='in', bottom=True, top=True, left=True, right=True)
ax1.set_xlim([0, 1.02])
ax1.set_ylim([2.25, 3.55])
ax1.set_xlabel(r'$\omega_{\mathrm{on}}$', fontsize=fontsize, labelpad=-5)
ax1.set_ylabel('$Pe$', fontsize=fontsize, labelpad=-5)

# Plot phase data from external CSV files
subdir = f'/Users/amirshee/my_work/active_fluids/numerics/one_dimension/phase_diagram/phase_data_homogeneous.csv'
df = pd.read_csv(subdir)
won_data = df['won'].values
Pe_data = df['Pe'].values
ax1.plot(won_data, Pe_data, 'o', color='k', markerfacecolor='none', markersize=markersize, label='Homogeneous')

subdir = f'/Users/amirshee/my_work/active_fluids/numerics/one_dimension/phase_diagram/phase_data_stationary_in_phase.csv'
df = pd.read_csv(subdir)
won_data = df['won'].values
Pe_data = df['Pe'].values
ax1.plot(won_data, Pe_data, 's', color='darkblue', markersize=markersize, label='Stationary 1')

subdir = f'/Users/amirshee/my_work/active_fluids/numerics/one_dimension/phase_diagram/phase_data_stationary_out_of_phase.csv'
df = pd.read_csv(subdir)
won_data = df['won'].values
Pe_data = df['Pe'].values
ax1.plot(won_data, Pe_data, '^', color='darkgreen', markersize=markersize, label='Stationary 2')

subdir = f'/Users/amirshee/my_work/active_fluids/numerics/one_dimension/phase_diagram/phase_data_moving.csv'
df = pd.read_csv(subdir)
won_data = df['won'].values
Pe_data = df['Pe'].values
ax1.plot(won_data, Pe_data, 'o', color='darkred', markersize=markersize, label='Moving')

ax1.legend(loc='upper right', fontsize=fontsize-5, markerfirst=True, labelspacing=0.02)

masked_lambdap_im = np.ma.masked_where(lambdap_real <= 0, lambdap_im)
ax1.contour(won_values, Pe_values, masked_lambdap_im, levels=[0], colors='k', linewidths=2.0, linestyles='dashed')

plt.show()
file_base_name = "fig4"  # change to your desired filename without extension
fig.savefig(f"{file_base_name}.png", dpi=600)   # PNG format
