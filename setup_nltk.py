#!/usr/bin/env python
"""
Script para descargar los datos necesarios de NLTK
Ejecuta esto una sola vez al principio
"""
import nltk
import ssl

# Solucionar problemas de SSL si es necesario
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

print("Descargando datos de NLTK necesarios...")
print("=" * 60)

# Descargar punkt (tokenización de palabras y oraciones)
print("\nDescargando 'punkt'...")
nltk.download('punkt')
nltk.download('punkt_tab')

# Descargar stopwords (palabras comunes)
print("\nDescargando 'stopwords'...")
nltk.download('stopwords')

print("\n" + "=" * 60)
print("✓ Descarga completada")
print("=" * 60)
