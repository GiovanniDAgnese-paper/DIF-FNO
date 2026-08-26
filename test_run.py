import torch
from dagnese_barrier import DAgneseBarrierLoss, get_compiled_dagnese_loss

def run_tests():
    print("=== Test D'Agnese Barrier Loss ===")
    
    # 1. Test standard con gradienti
    J_dummy = torch.randn(32, 64, 64, 2, 2, requires_grad=True)
    criterion = DAgneseBarrierLoss()
    
    loss = criterion(J_dummy)
    loss.backward()
    
    print(f"Perdita Calcolata: {loss.item():.4f}")
    print(f"Gradiente presente: {J_dummy.grad is not None}")
    assert J_dummy.grad is not None, "Errore: il gradiente non è stato calcolato!"
    
    # 2. Test stabilità su determinanti negativi (mesh piegata)
    J_folded = torch.tensor([[[-1.0, 0.0], [0.0, -1.0]]], requires_grad=True) # det = 1 - 0 = 1? No, (-1)*(-1) = 1.
    J_inverted = torch.tensor([[[1.0, 0.0], [0.0, -1.0]]], requires_grad=True) # det = -1 (piegato)
    
    loss_inv = criterion(J_inverted)
    loss_inv.backward()
    print(f"Gradiente su determinante negativo (Softplus attivo): {J_inverted.grad[0,0,0].item():.4f}")
    assert J_inverted.grad[0,0,0].item() != 0.0, "Errore: zona morta nel gradiente!"

    # 3. Test torch.compile()
    print("\n[+] Test della compilazione PyTorch (Kernel Fusion)...")
    compiled_criterion = get_compiled_dagnese_loss()
    loss_compiled = compiled_criterion(J_dummy)
    print(f"Perdita con torch.compile(): {loss_compiled.item():.4f}")
    print("\n[✓] TUTTI I TEST PASSATI CON SUCCESSO!")

if __name__ == "__main__":
    run_tests()
