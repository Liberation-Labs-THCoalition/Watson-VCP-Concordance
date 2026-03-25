import os
target = os.path.join("C:/Users/Thomas/Desktop/Watson-VCP-Concordance/code/concordance", "circularity_audit.py")
parts = []

parts.append('"""')
parts.append('Circularity Audit for Watson-VCP Concordance Study')
parts.append('')
parts.append('Tests whether VCP-geometry correlations survive when using ENCODE-phase')
parts.append('features only (pre-generation, zero circularity possible) vs generation-phase')
parts.append('features (where VCP tokens are part of the cache).')
parts.append('')
parts.append('For each model x VCP dimension x feature (60 pairs per model):')
parts.append('  - Generation-phase rho (what paper reports)')
parts.append('  - Encode-phase rho (circularity-free)')
parts.append('  - Delta-phase rho (generation - encode = pure generation contribution)')
parts.append('')
parts.append('All with FWL residualization (regress out confounds before correlation).')
parts.append('')
parts.append('Author: Lyra')
parts.append('Date: 2026-03-25')
parts.append('"""')

with open(target, "w") as f:
    f.write(chr(10).join(parts))
print("Generated")