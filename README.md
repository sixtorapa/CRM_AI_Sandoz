# 🏥 CRM-AI Sandoz

![Logo del Proyecto](assests/Logo_Sandoz.png)

## 🚀 Descripción
Este proyecto es un sistema **CRM** enriquecido con **Inteligencia Artificial**, que combina la **transcripción de audio** (usando OpenAI Whisper) y la **gestión de datos** (visitas, historial, ventas, etc.) en una interfaz creada con **Gradio**. Además, implementa consultas de tipo _Retrieval-Augmented Generation_ (RAG) con un **VectorStore FAISS** y modelos de **OpenAI** vía **LangChain**.


🔹 **Tecnologías utilizadas**:
- 🛠️ **Python**, `pandas`, `sklearn`, `flask`
- ☁️ **Azure Cognitive Services** para transcripción de audio
- 📊 **Machine Learning** para predicción y análisis de datos

## 🌍 Acceso a la Aplicación en Azure  

🚀 **Prueba la aplicación en vivo:**  
👉 [**CRM-AI Web App**](https://huggingface.co/spaces/sixtorapa/CRM-AI-SANDOZ)  

🔐 **Credenciales de Acceso:**  
- 👤 **Usuario:** `lauragomez`  
- 🔑 **Contraseña:** `realmadrid` 


## 📊 Funcionalidades

- 🎙️ **Transcripción Automática con OpenAI Whisper**  
  Convierte archivos de audio en texto de manera precisa, optimizando el registro de visitas y la captura de insights sin esfuerzo manual.

- 🧠 **Análisis Semántico con GPT-4**  
  Extrae y valida automáticamente información clave (cuenta, contacto, temas discutidos), además de generar resúmenes coherentes que condensan la visita en pocos párrafos.

- 📂 **Gestión y Seguimiento de Visitas**  
  Almacena cada interacción en un CSV histórico, manteniendo un registro detallado de puntos clave, contactos y compromisos. La información se indexa en FAISS para búsquedas semánticas eficientes.

- 🤝 **Consultas Inteligentes (RAG)**  
  Realiza preguntas sobre la cuenta y el historial de ventas; el sistema recupera contexto relevante y responde con GPT-4, ofreciendo recomendaciones estratégicas y conclusiones basadas en la información disponible.

- 📈 **Dashboard Interactivo**  
  Muestra, con visualizaciones integradas (matplotlib y plotly), el rendimiento comercial: visitas por representante, facturación por cuenta/producto y otros gráficos esenciales para la toma de decisiones.

- 🔐 **Sistema de Login Sencillo**  
  Controla el acceso y la modificación de datos mediante un archivo `Usuarios.csv`, garantizando que solo usuarios autorizados puedan gestionar la información sensible.


## 📂 Estructura del Proyecto


- **CRM-AI-Sandoz/**
  - `README.md` → Documentación del proyecto
  - `requirements.txt` → Dependencias del proyecto
  - **src/** (Código fuente principal)
    - `transcriptor_Azure_vf.py` → Script principal para transcripción
    - `otros_scripts.py` → Otros scripts auxiliares
  - **data/** (Archivos de datos utilizados)
    - `dataset.csv`
    - `processed_data.pkl`
  - **assets/** (Logos, imágenes y gráficos)
    - `Logo_Sandoz.png`
  - **.github/** (Configuración opcional para GitHub Actions, templates, etc.)

---

## 👥 Colaboradores  

Este proyecto fue desarrollado como parte de nuestro **Trabajo de Fin de Máster**.  

🔹 **Equipo:**  
- 🧑‍💻 **Sixto Ramírez Parras**  
- 🧑‍💻 **Carlos Juárez García**  
- 👩‍💻 **Beatriz Moraga Galán**  
- 🧑‍💻 **Francisco Jordán Medina** 


