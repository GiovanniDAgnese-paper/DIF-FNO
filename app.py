import gradio as gr
import numpy as np
import matplotlib.pyplot as plt

def visualize_diffeomorphism(curvature, barrier_active):
    x = np.linspace(0, 1, 32)
    y = np.linspace(0, 1, 32)
    X, Y = np.meshgrid(x, y)
    
    # Deformazione sinusoidale del dominio
    def_x = X + curvature * 0.1 * np.sin(2 * np.pi * Y)
    def_y = Y + curvature * 0.1 * np.cos(2 * np.pi * X)
    
    # Simula effetto Barrier Loss
    det_J = 1.0 + curvature * 0.2 * np.cos(2 * np.pi * X) * np.sin(2 * np.pi * Y)
    if not barrier_active:
        det_J = det_J - curvature * 0.35 # Provoca grid folding
        
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
    ax[0].pcolormesh(def_x, def_y, det_J, cmap='magma', shading='auto')
    ax[0].set_title(f"Mappa Diffeomorfa (min det(J) = {np.min(det_J):.3f})")
    ax[0].set_aspect('equal')
    
    ax[1].plot(def_x, def_y, 'k-', alpha=0.3)
    ax[1].plot(def_x.T, def_y.T, 'k-', alpha=0.3)
    ax[1].set_title("Griglia Fisica Preservata" if np.min(det_J) > 0 else "Singolarità / Collasso Metrico")
    ax[1].set_aspect('equal')
    
    plt.tight_layout()
    return fig

demo = gr.Interface(
    fn=visualize_diffeomorphism,
    inputs=[gr.Slider(0.1, 2.0, value=1.0, label="Curvatura del Dominio"), gr.Checkbox(value=True, label="Barrier Loss Attiva")],
    outputs="plot",
    title="DIF-FNO: Interactive Diffeomorphism Playground"
)

if __name__ == "__main__":
    demo.launch()
