#!/usr/bin/env python
"""
Script de prueba para verificar la carga correcta de modelos
"""
import sys
from pathlib import Path

# Agregar raíz del proyecto al path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from controllers.browser_integration import ModelBrowser

print("=" * 60)
print("PRUEBA DE CARGA DE MODELOS")
print("=" * 60)

browser = ModelBrowser()

print(f"\n✓ Raíz del proyecto: {browser.root}")
print(f"✓ Carpeta models existe: {(browser.root / 'models').exists()}")

# Verificar archivos disponibles
models_folder = browser.root / 'models'
if models_folder.exists():
    pkl_files = list(models_folder.glob("*.pkl"))
    print(f"✓ Archivos .pkl encontrados: {len(pkl_files)}")
    for f in pkl_files:
        print(f"  - {f.name}")

# Probar cada tipo de modelo
print("\n" + "=" * 60)
print("PROBANDO CARGA DE MODELOS")
print("=" * 60)

model_types = ['binary', 'tfidf', 'bm25']

for model_type in model_types:
    print(f"\n[{model_type.upper()}]")
    path = browser.get_model_path(model_type)
    if path:
        print(f"✓ Ruta encontrada: {path}")
        success, message = browser.load(path)
        print(f"{'✓' if success else '✗'} {message}")
        
        if success:
            # Probar búsqueda simple
            print(f"\n  Probando búsqueda con query 'test'...")
            results = browser.search("test", k=3)
            print(f"  {'✓' if results else '✗'} {len(results)} resultados")
            if results:
                for i, r in enumerate(results[:2], 1):
                    print(f"    {i}. {r}")
    else:
        print(f"✗ No se encontró archivo para {model_type}")

print("\n" + "=" * 60)
print("PRUEBA COMPLETADA")
print("=" * 60)
