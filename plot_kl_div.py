import matplotlib.pyplot as plt
import numpy as np

def render(ax, name, bpw, points):
    if isinstance(points, (int, float)):
        points = [(0, points)]
    bpws = [bpw + x for x, y in points]
    kl_divs = [y for x, y in points]

    sc = ax.scatter(bpws, kl_divs)
    ax.annotate(name, xy=(bpws[0], kl_divs[0]),
        ha='center', xytext=(0, 7), textcoords='offset points')
    for bpw1, bpw2, kl_div1, kl_div2 in zip(bpws, bpws[1:], kl_divs, kl_divs[1:]):
        print(bpw1, bpw2, kl_div1, kl_div2)
        ax.annotate("", xy=(bpw2, kl_div2), xytext=(bpw1, kl_div1),
            arrowprops=dict(arrowstyle="->", color=sc.get_edgecolor()))

fig, (ax1, ax2) = plt.subplots(2, 1)


ax1.set_xlabel('Bits per weight')
ax1.set_ylabel('KL divergence (mean), SlimOrca')
ax1.set_yscale('log')

repaired_kl_div = 9.554240e-2

ax1.hlines([repaired_kl_div], 0, 1, transform=ax1.get_yaxis_transform(),
    linestyle='dotted')

render(ax1, 'Q2_K', 3.167, [
    (0, 4.333135e-01),
    (0.184, repaired_kl_div),
])
render(ax1, 'Q3_K_M', 4.004, [
    (0, 9.097217e-02),
    #(0.184, 1.342204e-01),
])
render(ax1, 'Q4_K_S', 4.676, [
    (0, 4.503101e-02),
    #(0.184, 1.142839e-01),
])
render(ax1, 'Q4_K_M', 4.903, [
    (0, 3.345483e-02),
    #(0.184, 1.055249e-01),
])
render(ax1, 'Q5_K_M', 5.712, [
    (0, 9.328476e-03),
    #(0.184, 9.894287e-02),
])
render(ax1, 'Q6_K', 6.572, [
    (0, 5.011368e-03),
    #(0, 9.557699e-02),
])

render(ax1, 'IQ2_XS', 2.597, [
    (0, 5.132598e-01),
    #(0.184, 3.923437e-01),
])
#render(ax1, 'IQ2_XXS', 2.391, [
#    (0, ???),
#])


ax2.set_xlabel('Bits per weight')
ax2.set_ylabel('KL divergence (mean), wikitext')
ax2.set_yscale('log')

repaired_kl_div = 2.521393e-01

ax2.hlines([repaired_kl_div], 0, 1, transform=ax2.get_yaxis_transform(),
    linestyle='dotted')

render(ax2, 'Q2_K', 3.167, [
    (0, 1.569036e+00),
    (0.184, repaired_kl_div),
])
render(ax2, 'Q3_K_M', 4.004, [
    (0, 3.953914e-01),
    #(0.184, 7.094271e-01),
])
render(ax2, 'Q4_K_S', 4.676, [
    (0, 1.214022e-01),
    #(0.184, 9.848665e-01),
])
render(ax2, 'Q4_K_M', 4.903, [
    (0, 6.792033e-02),
    #(0.184, 7.697396e-01),
])
render(ax2, 'Q5_K_M', 5.712, [
    (0, 1.972850e-02),
    #(0.184, 7.992139e-01),
])
render(ax2, 'Q6_K', 6.572, [
    (0, 1.755188e-02),
    #(0.184, 7.607292e-01),
])

render(ax2, 'IQ2_XS', 2.597, [
    (0, 2.102656e+00),
    #(0.184, 2.048991e+00),
])
#render(ax2, 'IQ2_XXS', 2.391, [
#    (0, ???),
#])


#fig.suptitle('Llama3 8B Instruct\nLoRA trained on SlimOrca, 50M tokens\nBase quant: Q2_K')
fig.suptitle('Llama3 8B Instruct\nLoRA trained on SlimOrca, 50M tokens')
fig.set_size_inches(8, 10)
fig.savefig('graph.png')
plt.show()
