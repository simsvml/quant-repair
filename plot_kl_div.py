import matplotlib.pyplot as plt
import numpy as np

def render(name, bpw, points):
    if isinstance(points, (int, float)):
        points = [(0, points)]
    bpws = [bpw + x for x, y in points]
    kl_divs = [y for x, y in points]

    sc = plt.scatter(bpws, kl_divs)
    plt.annotate(name, xy=(bpws[0], kl_divs[0]),
        ha='center', xytext=(0, 7), textcoords='offset points')
    for bpw1, bpw2, kl_div1, kl_div2 in zip(bpws, bpws[1:], kl_divs, kl_divs[1:]):
        print(bpw1, bpw2, kl_div1, kl_div2)
        plt.annotate("", xy=(bpw2, kl_div2), xytext=(bpw1, kl_div1),
            arrowprops=dict(arrowstyle="->", color=sc.get_edgecolor()))

plt.xlabel('Bits per weight')
plt.ylabel('KL divergence (mean)')
plt.yscale('log')
render('Q2_K', 3.167, [
    (0, 2.825637e-01),
    #(0.184, 1.102156e-01),
    (0.184, 9.422227e-02),
])
render('Q3_K_M', 4.004, 8.982828e-02)
render('Q4_K_M', 4.903, 3.029343e-02)
render('Q4_K_S', 4.676, 4.992203e-02)
render('Q5_K_M', 5.712, 9.637030e-03)
render('Q6_K', 6.572, 5.705326e-03)

plt.savefig('graph.png')
plt.show()
