import networkx as nx
import powerlaw
import matplotlib.pyplot as plt
import itertools
from networkx.algorithms.community import girvan_newman


G = nx.read_graphml('temp/graphs/graph.graphml')

# 1. Check Node Degree Distribution
degrees = [d for n, d in G.degree()]
fit = powerlaw.Fit(degrees)
tau1_estimated = fit.power_law.alpha

print(f"Estimated exponent for node degree distribution: {tau1_estimated}")

# Plot the degree distribution
fig, ax = plt.subplots()
fit.plot_pdf(ax=ax, label='Empirical Data')
fit.power_law.plot_pdf(ax=ax, color='r', linestyle='--', label='Power Law Fit')
ax.legend()
plt.savefig("temp/communities-powerlaw-fit.png")

# 2. Check Community Size Distribution
# Detect communities using the Girvan-Newman method
def get_communities(G, k):
    comp = girvan_newman(G)
    next_layer = comp
    for _ in range(10):
        next_layer = next(comp)
    communities_gn = [list(c) for c in next_layer]
    return communities_gn

k = 5  # Number of communities to detect
communities = get_communities(G, k)
community_sizes = [len(c) for c in communities]

fit_community = powerlaw.Fit(community_sizes)
tau2_estimated = fit_community.power_law.alpha

print(f"Estimated exponent for community size distribution: {tau2_estimated}")

# Plot the community size distribution
fig, ax = plt.subplots()
fit_community.plot_pdf(ax=ax, label='Empirical Data')
fit_community.power_law.plot_pdf(ax=ax, color='r', linestyle='--', label='Power Law Fit')
ax.legend()
plt.savefig("temp/communities-size-distribution.png")

# Compare the estimated exponents with the given τ1 and τ2
tau1 = 2.5  # Replace with your given τ1
tau2 = 1.5  # Replace with your given τ2

print(f"Given τ1: {tau1}, Estimated τ1: {tau1_estimated}")
print(f"Given τ2: {tau2}, Estimated τ2: {tau2_estimated}")

if abs(tau1 - tau1_estimated) < 0.1:
    print("Node degrees follow a power law distribution with the given exponent τ1.")
else:
    print("Node degrees do not follow a power law distribution with the given exponent τ1.")

if abs(tau2 - tau2_estimated) < 0.1:
    print("Community sizes follow a power law distribution with the given exponent τ2.")
else:
    print("Community sizes do not follow a power law distribution with the given exponent τ2.")