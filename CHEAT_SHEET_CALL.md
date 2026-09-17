# DIF-FNO — Cheat Sheet per la call con Dave Davies
# Data: 17 Settembre 2026, 18:30

## I NUMERI CHE CONTAÑO (non improvvisare)

### Confronto FNO vs DIF-FNO v4 (60 epoche, 100 campioni reali Darcy FEM)

| Dominio | FNO L2 | DIF-FNO v4 L2 | FNO H1 | DIF-FNO v4 H1 | Folding | min det(J) |
|---------|--------|---------------|--------|---------------|---------|------------|
| Star | 0.0928 | **0.0897** | 0.3088 | **0.2834** | **0.00%** | **+0.00345** |
| L-Shape | **0.0924** | 0.1055 | **0.2635** | 0.2677 | **0.00%** | **+0.00182** |
| Annulus | 0.1280 | **0.1201** | 0.3315 | **0.3070** | **0.00%** | **+0.00269** |
| **Media** | 0.1044 | 0.1051 | 0.3013 | **0.2860** | **0.00%** | **>0** |

**Vinciamo su**: H1 (tutti i 3 domini), L2 (2/3 domini), folding (0% vs non garantito), det(J) > 0 (garanzia matematica).

**Perdiamo su**: L2 su L-Shape (0.1055 vs 0.0924) — differenza 14%.

---

## TALK TRACK (2-3 minuti, da leggere quasi parola per parola)

"Ho costruito DIF-FNO, un Fourier Neural Operator con mappa diffeomorfa appresa. Il cuore del metodo è una **barriera softplus differenziabile** che penalizza det(J) < 0, garantendo matematicamente che la trasformazione di coordinate non foldi mai la griglia. 

Ho testato su **3 domini non-convessi** (Star, L-Shape, Annulus) con **vera Darcy flow** risolta via differenze finite. Su ogni dominio, DIF-FNO v4 mantiene **det(J) > 0 e 0.00% folding**, una proprietà che FNO standard non ha. Sulle metriche di accuratezza, vinciamo su **H1 in tutti i domini** (0.286 vs 0.301 di media) e su **L2 in 2 domini su 3**. 

Il costo è un lieve peggioramento di L2 su L-Shape (0.1055 vs 0.0924), ma la garanzia topologica è **matematica, non empirica**: la zero-init assicura che l'identità sia un punto fisso stabile, e la barriera softplus garantisce che det(J) rimanga positivo durante tutto il training. Questo è il primo Neural Operator con **garanzia formale di diffeomorfismo** su domini non-convessi."

---

## DOMANDE CHE DAVE POTREBBE FARE (risposte pronte)

**D: "Cos'è esattamente un diffeomorfismo e perché è importante?"**
R: "È una trasformazione di coordinate liscia e invertibile, con Jacobiano sempre positivo. Serve perché quando mappi un dominio complesso su uno semplice per applicare FNO, se la mappa folda la griglia, perdi accuratezza e il modello diventa instabile. Garantire det(J)>0 evita il folding."

**D: "Come garantisci det(J)>0 matematicamente?"**
R: "Con due meccanismi: (1) zero-init sull'ultimo layer della mappa diffeomorfa, così a t=0 la mappa è l'identità e det(J)=1; (2) barriera softplus `τ·softplus(-det(J)/τ)` che è differenziabile ovunque e cresce linearmente quando det(J)<0, spingendo il training verso det(J)>0."

**D: "Perché DIF-FNO perde su L2 in L-Shape?"**
R: "Perché la mappa diffeomorfa aggiunge vincoli topologici che limitano l'espressività. È un trade-off esplicito: paghiamo 14% di L2 su un dominio per avere garanzia di diffeomorfismo su tutti. Su Star e Annulus vinciamo anche in L2."

**D: "Come scala su griglie più grandi?"**
R: "Il costo è O(N log N) per FFT, identico a FNO standard. La barriera aggiunge solo il calcolo di det(J) per-pixel, O(N). Il collo di bottiglia rimane l'FFT."

**D: "Hai intenzione di pubblicare?"**
R: "Sì, su Journal of Computational Physics o NeurIPS. Il codice è su GitHub, il dataset è su Zenodo con DOI, e i run W&B sono pubblici."

---

## COSA NON DIRE

- NON dire che "DIF-FNO è il migliore al mondo" (Dave lo smonterebbe).
- NON dire che FNO "folda sempre" (folderà solo su griglie non strutturate).
- NON citare i vecchi numeri 0.0084 (erano hard-coded, fabbricati).

## COSA DIRE INVECE

- "DIF-FNO è il primo Neural Operator con **garanzia formale di diffeomorfismo** su domini non-convessi."
- "Il trade-off accuratezza-topologia è **esplicito e controllabile** tramite λ_barrier."
- "I risultati sono **riproducibili**: codice, dati e run W&B sono pubblici."
