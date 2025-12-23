# 🛣️ Roadmap to Production: Risk Scoring Engine

Este documento redefine el camino hacia la maestría en MLOps, enfocándose en la infraestructura sólida, la simulación realista y el monitoreo continuo, evitando distracciones prematuras (como agentes autónomos) hasta dominar los fundamentos.

## 🎯 Meta Final

Tener un sistema que procese datos reales (Kaggle), se despliegue automáticamente (CI/CD), viva en contenedores optimizados (Docker) y tenga un dashboard de monitoreo en tiempo real que simule la operación diaria de un banco.

---

## 🏗️ FASE 1: Fundamentos de Infraestructura (Docker & CI/CD)
>
> **Estado:** 🚧 En Progreso
> **Objetivo:** "Si funciona en mi máquina, funciona en la nube".

### 1.1. Docker Pro (Infraestructura Inmutable)

Ya tenemos un `Dockerfile` básico, pero para producción real necesitamos:

- **Optimization:** Reducir el tamaño de la imagen (de ~1GB a ~200MB) usando *Multi-stage builds*.
- **Security:** Gestión segura de secretos (no hardcodeados).
- **Entendimiento Profundo:** Diferenciar entre construir la imagen (`build`) y correr el contenedor (`run`), y cómo exponer puertos correctamente.

### 1.2. GitHub Actions (El Robot Guardián)

Ya creamos el archivo YAML, pero necesitamos "sentir el dolor" para aprender:

- **Simulación de Fallo:** Introducir un error intencional (ej. data leakage) y ver cómo GitHub Actions bloquea el despliegue.
- **Continuous Integration (CI):** Entender el flujo `Push` -> `Test` -> `Build`.

---

## 👁️ FASE 2: Observabilidad y Monitoreo (Drift)
>
> **Estado:** 📅 Pendiente
> **Objetivo:** "¿Mi modelo sigue siendo inteligente o se ha vuelto tonto?"

### 2.1. Conceptos de Drift

Entender que los datos cambian con el tiempo (ej. inflación afecta ingresos).

- **Data Drift:** Cambios en la distribución de las variables de entrada (`AMT_INCOME`).
- **Concept Drift:** Cambios en la relación entre variables y el target (la definición de "moroso" cambia).

### 2.2. Estrategia de Monitoreo

Diseñar un sistema que alerte si:

- El % de nulos sube repentinamente.
- La distribución de predicciones cambia drásticamente.

---

## 🏦 FASE 3: La Gran Simulación (Real-World Emulation)
>
> **Estado:** 📅 Pendiente
> **Objetivo:** Simular un entorno bancario vivo.

### 3.1. Ingesta de Datos Reales

- Cargar el dataset completo de Kaggle (`application_train.csv` ~300k filas).
- Adaptar `make_dataset.py` para manejar archivos grandes sin explotar la memoria.

### 3.2. Estrategia de "Viaje en el Tiempo"

Dividir los 300k datos en:

- **Historia (Training):** Los primeros 280,000 clientes (ordenados por fecha si fuera posible, o aleatorio).
- **Futuro (Inference):** Los últimos 20,000 clientes, reservados para simular la llegada de nuevos solicitantes día a día.

### 3.3. Simulador de Tráfico (El "Cliente")

Crear un script en Python que actúe como el sistema del banco:

- Lee los 20k datos reservados.
- Envía peticiones `POST /predict` a nuestra API Dockerizada cada pocos segundos.
- Simula picos de tráfico.

### 3.4. Dashboard de Control (Streamlit)

Construir un centro de mando visual que consuma los resultados de la API y muestre:

- **Aprobaciones vs Rechazos** en tiempo real.
- **Histograma de Riesgo** actualizado al segundo.
- **Alertas de Calidad** (Drift detectado).

---

## 🚫 Fuera del Alcance (Por ahora)

- **Agentes Autónomos (Deep Research/Coding Agents):** Distracción. Primero debemos construir el sistema que el agente eventualmente operaría.
- **Cloud Deploy (GCP/AWS):** Primero dominaremos Docker localmente. Desplegar una imagen Docker optima es trivial si la imagen está bien hecha.

---

## 📝 Siguientes Pasos Inmediatos

1. **Terminar Fase 1:** Optimizar Docker y validar GitHub Actions.
2. **Conseguir Datos:** Descargar dataset de Kaggle.
3. **Iniciar Fase 3:** Construir el simulador.
