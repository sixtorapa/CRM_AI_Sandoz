import gradio as gr
import requests
import tempfile
import os
from datetime import datetime
import time
import pandas as pd
import re
import json
from rapidfuzz import process, fuzz
import matplotlib.pyplot as plt
import plotly.express as px
import unicodedata
import os 

# ==================================
# IMPORTS DE LANGCHAIN
# ==================================
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# ==================================
# CONFIGURACIÓN OPENAI
# ==================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ==================================
# FUNCIÓN PARA NORMALIZAR NOMBRES DE CUENTAS
# ==================================
def normalize_account(account):
    """
    Normaliza un nombre de cuenta: lo pasa a minúsculas, elimina espacios extra y remueve diacríticos.
    """
    account = account.lower().strip()
    account = unicodedata.normalize('NFKD', account).encode('ASCII', 'ignore').decode('utf-8')
    return account

# ==================================
# CARGA DE DATOS Y CONFIGURACIÓN DE VECTORSTORE (VISITAS)
# ==================================
df = pd.read_csv("Registro_Sandoz.csv")
df["Cuenta_Normalizada"] = df["Cuenta"].astype(str).apply(normalize_account)

documents = df.apply(
    lambda x: f"Cuenta: {x['Cuenta']}, Puntos Clave: {x['Puntos Clave']}",
    axis=1
).tolist()

vectorstore = FAISS.from_texts(
    texts=documents,
    embedding=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4"),
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

# ==================================
# FUNCIÓN DE DETECCIÓN DE CUENTA
# ==================================
def detect_entities(transcribed_text):
    """
    Detecta el nombre de la cuenta en el texto transcrito y,
    si encuentra algo, lo pone con mayúscula inicial de cada palabra (ej. 'Farmacia San Juan').
    Si no encuentra nada, devuelve 'Cuenta no detectada'.
    """
    prompt = (
        "Eres un analizador de texto que extrae el nombre de la cuenta. "
        "Dado un texto en idioma español, tu tarea es identificar el nombre de la cuenta o lugar (por ejemplo, 'Clínica San Juan', 'Farmacia Central', etc.) que se menciona en el texto. "
        "Si no encuentras nada claro, devuelve 'Cuenta no detectada'.\n\n"
        "Devuelve SOLAMENTE un JSON con la clave:\n"
        "{\n"
        "  \"cuenta_detectada\": \"...\"\n"
        "}\n"
        "No añadas explicación ni texto adicional."
    )
    user_prompt = f"Texto transcrito:\n{transcribed_text}\n"
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_prompt}
    ]
    raw_answer = call_chat_api(messages, model="gpt-4", temperature=0)
    try:
        data = json.loads(raw_answer)
        acc = data.get("cuenta_detectada", "Cuenta no detectada").strip()
    except:
        acc = "Cuenta no detectada"
    if not acc:
        acc = "Cuenta no detectada"

    # Forzamos la capitalización
    if acc.lower() != "cuenta no detectada":
        acc = " ".join([word.capitalize() for word in acc.split()])

    return acc if acc else "Cuenta no detectada"

# ==================================
# FUNCIÓN DE DETECCIÓN DE CONTACTO
# ==================================
def detect_contact(transcribed_text):
    """
    Utiliza un prompt para extraer el nombre de la persona o departamento de contacto 
    mencionado en el texto. Si no se detecta nada, devuelve 'Contacto no detectado'.
    """
    prompt = (
        "Eres un analizador de texto. Dado un texto en español que describe una visita comercial, "
        "identifica la persona o departamento de contacto al que se hace referencia. "
        "Si no queda claro, devuelve 'Contacto no detectado'.\n\n"
        "Devuelve SOLAMENTE un JSON con la clave:\n"
        "{\n"
        "  \"contacto_detectado\": \"...\"\n"
        "}\n"
        "No añadas explicación ni texto adicional."
    )
    user_prompt = f"Texto transcrito:\n{transcribed_text}\n"
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_prompt}
    ]
    raw_answer = call_chat_api(messages, model="gpt-4", temperature=0)
    try:
        data = json.loads(raw_answer)
        contact = data.get("contacto_detectado", "Contacto no detectado").strip()
    except:
        contact = "Contacto no detectado"

    # Forzamos la capitalización
    if contact.lower() != "contacto no detectado":
        contact = " ".join(word.capitalize() for word in contact.split())

    return contact or "Contacto no detectado"

# ==================================
# FUNCIONES AUXILIARES / AUDIO / TRANSCRIPCIÓN
# ==================================
def call_openai_audio_transcription(audio_path):
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    data = {"model": "whisper-1", "response_format": "json"}
    try:
        with open(audio_path, "rb") as audio_file:
            files = {"file": (os.path.basename(audio_path), audio_file, "audio/mpeg")}
            response = requests.post(url, headers=headers, data=data, files=files)
        if response.status_code == 200:
            return response.json()["text"]
        else:
            return f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return str(e)

def call_chat_api(messages, model="gpt-4", temperature=0):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {OPENAI_API_KEY}"}
    payload = {"model": model, "temperature": temperature, "messages": messages}
    resp = requests.post(url, headers=headers, json=payload)
    if resp.status_code == 200:
        return resp.json()["choices"][0]["message"]["content"]
    else:
        return f"Error {resp.status_code}: {resp.text}"

def call_chat_api_simple(prompt, model="gpt-4", temperature=0):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {OPENAI_API_KEY}"}
    payload = {"model": model, "temperature": temperature, "messages": [{"role": "user", "content": prompt}]}
    resp = requests.post(url, headers=headers, json=payload)
    if resp.status_code == 200:
        return resp.json()["choices"][0]["message"]["content"]
    return f"Error {resp.status_code}: {resp.text}"

def semantic_filter_transcription(raw_transcription):
    """
    Filtra datos sensibles (nombres de pacientes, direcciones, precios, competidores, etc.)
    Pero conserva el nombre de la cuenta/lugar y el contacto si aparece.
    """
    prompt = f"""
Eres un asistente de procesamiento de transcripciones para un sistema de CRM de ventas. 
Tu tarea es filtrar la siguiente transcripción de una reunión de negocios para eliminar 
información irrelevante y dejar únicamente los datos esenciales para la empresa.

Debes **eliminar**:
- Charlas triviales como saludos, despedidas y comentarios sin valor para el negocio.
- Conversaciones personales sobre vacaciones, familia, deportes, clima o temas no relacionados con la empresa.
- Frases de cortesía o relleno como “qué tal” etc.
- Información repetitiva que no aporte valor a la conversación.

Debes **mantener**:
- Información sobre la cuenta visitada.
- Preguntas y respuestas relacionadas con pedidos, productos, entregas y necesidades del cliente.
- Compromisos de acción, acuerdos o seguimientos.
- Cualquier comentario relevante para la estrategia comercial.
- El nombre de la persona de contacto o departamento mencionado.

Por favor, devuelve únicamente la transcripción limpia, sin explicaciones adicionales.

Transcripción:
{raw_transcription}
"""
    filtered_text = call_chat_api_simple(prompt)
    return filtered_text

# ==================================
# FUNCIÓN DE GENERAR RESUMEN
# ==================================
def generate_plain_text_summary(filtered_transcription, final_account, visit_date, historical_summary):
    """
    Genera un resumen en texto plano con los apartados:
      - Cuenta
      - Fecha
      - Histórico
      - Resumen de la visita
      - Puntos clave (basados únicamente en la visita actual)
    """
    prompt = f"""
Eres un asistente de CRM para Sandoz. Genera un resumen de la visita en formato de texto plano, 
utilizando saltos de línea para separar claramente cada apartado. El resumen debe contener los siguientes apartados:

Cuenta: {final_account}
Fecha: {visit_date}

Histórico:
{historical_summary}

Resumen de la visita:
[Genera un resumen claro y conciso de la visita basándote en la siguiente transcripción:]
{filtered_transcription}

Puntos clave:
[Extrae y resume exclusivamente los puntos clave de la visita actual basándote en la siguiente transcripción. No incluyas información de visitas anteriores:]
{filtered_transcription}

Por favor, devuelve el resumen estructurado con la siguiente estructura:

Cuenta: <nombre de la cuenta>
Fecha: <fecha y hora>
Histórico:
<resumen histórico en varias líneas>
Resumen de la visita:
<resumen de la visita actual en varias líneas>
Puntos clave:
<puntos clave de la visita actual en varias líneas>
"""
    response = call_chat_api_simple(prompt)
    return response.strip()

def save_text_to_file(text, filename):
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)
    return file_path

def extract_puntos_clave(gpt_output):
    """
    Extrae la sección de 'Puntos clave:' del texto generado por GPT.
    """
    pattern = r"Puntos clave:\s*(.+)$"
    match = re.search(pattern, gpt_output, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        return gpt_output

# ==================================
# FUNCIÓN PARA AÑADIR VISITA AL CSV (incluyendo Contacto)
# ==================================
def add_visit_to_csv(account_name, representative_name, visit_date, summary, contact):
    """
    Añade una nueva fila al CSV 'Registro_Sandoz.csv' con los datos de la visita, 
    incluyendo la columna 'Contacto'.
    """
    new_record = pd.DataFrame({
        "Cuenta": [account_name],
        "Representante": [representative_name],  # Quien se logea
        "Fecha Visita": [visit_date],
        "Puntos Clave": [summary],
        "Contacto": [contact]  # Nueva columna
    })
    new_record.to_csv("Registro_Sandoz.csv", mode='a', header=False, index=False)

def update_vectorstore_with_new_record(account, summary):
    new_text = f"Cuenta: {account}, Puntos Clave: {summary}"
    vectorstore.add_texts([new_text])

# ==================================
# FUNCIÓN DE VERIFICAR PUNTOS FALTANTES
# ==================================
def check_missing_points(transcribed_text):
    """
    Revisa si en la transcripción se cumplen algunos puntos mínimos, y devuelve un JSON
    con estado 'presente' / 'faltante' por cada punto.
    """
    prompt = f"""
Eres un verificador de integridad para reportes de visita de Sandoz.
Puntos mínimos:
1. Identificación de la cuenta
2. Persona de contacto o departamento
3. Productos o líneas discutidas
4. Feedback y acciones futuras

Compara la transcripción:
{transcribed_text}

Devuelve un JSON con:
{{
  "reporte_integridad": [
    {{
      "punto": "<nombre>",
      "estado": "presente" o "faltante",
      "comentario": "..."
    }},
    ...
  ],
  "recomendacion_global": "..."
}}
"""
    ans = call_chat_api_simple(prompt)
    try:
        data = json.loads(ans)
        return data
    except:
        return {
            "reporte_integridad": [],
            "recomendacion_global": "No se pudo parsear la respuesta del modelo."
        }

# ==================================
# FUNCIONES PARA RAG / HISTORIAL
# ==================================
def retrieve_account_records(account_name, limit):
    df_records = pd.read_csv("Registro_Sandoz.csv")
    df_records["Cuenta_Normalizada"] = df_records["Cuenta"].astype(str).apply(normalize_account)
    df_records["Fecha Visita"] = pd.to_datetime(df_records["Fecha Visita"], errors="coerce")
    norm_account = normalize_account(account_name)
    subset = df_records[df_records["Cuenta_Normalizada"] == norm_account]
    if subset.empty:
        return "No se encontraron registros para la cuenta ingresada."
    subset = subset.sort_values(by="Fecha Visita", ascending=False)
    try:
        limit = int(limit)
        if limit > 0:
            subset = subset.head(limit)
    except:
        pass
    result_lines = []
    for _, row in subset.iterrows():
        fecha_str = (
            row["Fecha Visita"].strftime('%Y-%m-%d %H:%M:%S') 
            if pd.notnull(row["Fecha Visita"]) else "Sin fecha"
        )
        result_lines.append(f"{fecha_str}\n{row['Puntos Clave']}")
    return "\n\n".join(result_lines)

def retrieve_sales_summary(account_name):
    try:
        df_sales = pd.read_csv("Ventas.csv")
    except Exception as e:
        return "Error al leer CSV de ventas: " + str(e)
    if df_sales.empty:
        return "No hay datos de ventas disponibles."
    df_sales["Cuenta_Normalizada"] = df_sales["Cuenta"].astype(str).apply(normalize_account)
    norm_account = normalize_account(account_name)
    subset_sales = df_sales[df_sales["Cuenta_Normalizada"] == norm_account]
    if subset_sales.empty:
        return "No se encontraron registros de ventas para la cuenta ingresada."
    subset_sales.loc[:, "Facturacion"] = pd.to_numeric(subset_sales["Facturacion"], errors="coerce")
    sales_summary = subset_sales.groupby("Producto")["Facturacion"].sum().reset_index()
    sales_summary = sales_summary.sort_values(by="Facturacion", ascending=False)
    output = ""
    for _, row in sales_summary.iterrows():
        output += f"{row['Producto']}: {row['Facturacion']}\n"
    return output

def retrieve_account_history(account_name, limit):
    visits_history = retrieve_account_records(account_name, limit)
    sales_history = retrieve_sales_summary(account_name)
    combined = "Historial de visitas:\n" + visits_history + "\n\nHistorial de ventas:\n" + sales_history
    return combined

def answer_commercial_query(query, account_name):
    if account_name.strip() == "":
        detected_account = detect_entities(query)
        if detected_account in ["Cuenta no detectada", "Sin cuenta detectada"]:
            detected_account = "Cuenta desconocida"
        else:
            detected_account = detected_account
    else:
        detected_account = account_name
    combined_context = retrieve_account_history(detected_account, 5)
    prompt = f"""
Eres un asistente comercial experto en Sandoz, con amplia experiencia en estrategias para sus cuentas cliente.
Utiliza la siguiente información histórica combinada (visitas y ventas) para la cuenta {detected_account}:
----------------------------------------------------
{combined_context}
----------------------------------------------------
Basándote en esta información y en la siguiente consulta:
"{query}"
Proporciona una respuesta concisa y profesional, incluyendo sugerencias estratégicas y recomendaciones específicas para la cuenta.
Asegúrate de utilizar todo el contexto histórico proporcionado.
"""
    response = call_chat_api_simple(prompt)
    return response

def dashboard_overview():
    # Gráfico 1: Número de visitas por comercial
    try:
        df_visits = pd.read_csv("Registro_Sandoz.csv")
    except Exception as e:
        fig1 = plt.figure()
        plt.text(0.5, 0.5, "Error al leer CSV de visitas:\n" + str(e), ha="center", va="center")
    else:
        if df_visits.empty:
            fig1 = plt.figure()
            plt.text(0.5, 0.5, "No hay datos de visitas", ha="center", va="center")
        else:
            rep_counts = df_visits.groupby("Representante").size().reset_index(name="Cantidad")
            rep_counts = rep_counts.sort_values(by="Cantidad", ascending=False)
            fig1, ax1 = plt.subplots()
            ax1.bar(rep_counts["Representante"], rep_counts["Cantidad"], color="skyblue")
            ax1.set_title("Número de visitas por comercial")
            ax1.set_xlabel("Comercial")
            ax1.set_ylabel("Visitas")
            for tick in ax1.get_xticklabels():
                tick.set_rotation(45)
            plt.tight_layout()

    # Gráfico 2: Facturación por cuenta
    try:
        df_sales = pd.read_csv("Ventas.csv")
    except Exception as e:
        fig2 = plt.figure()
        plt.text(0.5, 0.5, "Error al leer CSV de ventas:\n" + str(e), ha="center", va="center")
    else:
        if df_sales.empty:
            fig2 = plt.figure()
            plt.text(0.5, 0.5, "No hay datos de ventas", ha="center", va="center")
        else:
            df_sales["Facturacion"] = pd.to_numeric(df_sales["Facturacion"], errors="coerce")
            fact_by_cuenta = df_sales.groupby("Cuenta")["Facturacion"].sum().reset_index()
            fact_by_cuenta = fact_by_cuenta.sort_values(by="Facturacion", ascending=False)
            fig2, ax2 = plt.subplots()
            ax2.bar(fact_by_cuenta["Cuenta"], fact_by_cuenta["Facturacion"], color="salmon")
            ax2.set_title("Facturación por cuenta")
            ax2.set_xlabel("Cuenta")
            ax2.set_ylabel("Facturación")
            for tick in ax2.get_xticklabels():
                tick.set_rotation(45)
            plt.tight_layout()

    # Gráfico 3: Facturación por producto
    try:
        df_sales2 = pd.read_csv("Ventas.csv")
    except Exception as e:
        fig3 = plt.figure()
        plt.text(0.5, 0.5, "Error al leer CSV de ventas:\n" + str(e), ha="center", va="center")
    else:
        if df_sales2.empty:
            fig3 = plt.figure()
            plt.text(0.5, 0.5, "No hay datos de ventas", ha="center", va="center")
        else:
            df_sales2["Facturacion"] = pd.to_numeric(df_sales2["Facturacion"], errors="coerce")
            fact_by_producto = df_sales2.groupby("Producto")["Facturacion"].sum().reset_index()
            fact_by_producto = fact_by_producto.sort_values(by="Facturacion", ascending=False)
            fig3, ax3 = plt.subplots()
            ax3.bar(fact_by_producto["Producto"], fact_by_producto["Facturacion"], color="lightgreen")
            ax3.set_title("Facturación por producto")
            ax3.set_xlabel("Producto")
            ax3.set_ylabel("Facturación")
            for tick in ax3.get_xticklabels():
                tick.set_rotation(45)
            plt.tight_layout()

    # Gráfico 4 (plotly) Interactivo
    try:
        df_sales3 = pd.read_csv("Ventas.csv")
        if not df_sales3.empty:
            df_sales3["Facturacion"] = pd.to_numeric(df_sales3["Facturacion"], errors="coerce")
            fig4 = px.bar(
                df_sales3,
                x="Cuenta",
                y="Facturacion",
                color="Producto",
                barmode="group",
                title="Gráfico interactivo de facturación por cuenta y producto"
            )
        else:
            fig4 = px.bar(title="No hay datos de ventas para mostrar")
    except Exception as e:
        fig4 = px.bar(title="Error al leer CSV de ventas: " + str(e))

    return fig1, fig2, fig3, fig4

# ====================================================
# CÓDIGO DE LOGIN (CSV con columnas: Usuario, Contraseña, Representante)
# ====================================================
csv_filename = "Usuarios.csv"
if not os.path.exists(csv_filename):
    df_template = pd.DataFrame({
        "Usuario": ["usuario1", "usuario2", "usuario3", "usuario4"],
        "Contraseña": ["password1", "password2", "password3", "password4"],
        "Representante": ["Responsable 1", "Responsable 2", "Responsable 3", "Responsable 4"]
    })
    df_template.to_csv(csv_filename, index=False, encoding="utf-8-sig")
    print(f"✅ Archivo {csv_filename} creado correctamente.")

def load_users():
    try:
        df_users = pd.read_csv("Usuarios.csv", encoding="utf-8-sig")
        df_users.columns = [col.strip() for col in df_users.columns]
        df_users["Contraseña"] = df_users["Contraseña"].astype(str)
        return df_users
    except Exception as e:
        print(f"⚠️ Error al cargar usuarios: {e}")
        return pd.DataFrame(columns=["Usuario", "Contraseña", "Representante"])

df_users = load_users()

def verify_login(username, password):
    """Verifica las credenciales y, si son correctas, devuelve el valor de la columna 'Representante'."""
    user_row = df_users[df_users["Usuario"] == username]
    if not user_row.empty and user_row["Contraseña"].values[0] == password:
        return True, user_row["Representante"].values[0]
    return False, None

# ====================================================
# INTERFAZ GRADIO
# ====================================================
with gr.Blocks() as demo:
    gr.HTML('''
    <style>
    /* Forzamos fondo blanco en .white-container */
    .white-container {
        background-color: #ffffff !important;
        padding: 20px !important;
        margin-top: 20px !important;
        border-radius: 5px !important;
    }
    .white-container * {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    .dark .white-container * {
        background-color: #ffffff !important;
        color: #000000 !important;
    }

    .btn-spacing {
        margin-bottom: 20px !important;
    }

    .step-heading-1-5 {
        background-color: transparent !important;
        color: #333 !important;
        display: inline-block !important;
        padding: 2px !important;
        margin: 0 !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        border: none !important;
        box-shadow: none !important;
    }
    </style>
    ''')

    gr.HTML('''
    <script>
      const logoutChannel = new BroadcastChannel("logout_channel");

      logoutChannel.onmessage = (event) => {
        if (event.data === "logout") {
          localStorage.clear();
          sessionStorage.clear();
          window.location.reload(); 
        }
      };

      function broadcastLogout() {
        logoutChannel.postMessage("logout");
      }
    </script>
    ''')
    
    # Logo superior
    gr.Image(
        value=r"Logo_Sandoz.png", 
        show_label=False,
        interactive=False
    )

    # Estado de sesión para almacenar el usuario logeado (valor de 'Representante')
    user_session = gr.State(value=None)
    
    # ===================================================
    # PESTAÑA "LOGIN"
    # ===================================================
    with gr.Tab("Login") as tab_login:
        gr.Markdown("## Asistente CRM AI Sandoz - Login")
        username_input = gr.Textbox(label="Usuario")
        password_input = gr.Textbox(label="Contraseña", type="password")
        login_button = gr.Button("Iniciar sesión")
        login_message = gr.Markdown()
        
        def login(username, password):
            ok, representative = verify_login(username, password)
            if ok:
                return f"✅ Bienvenido {representative}", representative
            return "❌ Usuario o contraseña incorrectos.", None
        
        login_button.click(
            fn=login,
            inputs=[username_input, password_input],
            outputs=[login_message, user_session]
        )
    
    # ===================================================
    # PESTAÑA "REGISTRAR VISITA"
    # ===================================================
    with gr.Tab("Subir audio", visible=False) as tab_audio:
        gr.Markdown("## Registro de visita - transcripción de audios y resumen")
        
        # Estados para almacenar transcripciones y validaciones
        main_transcript_state = gr.State()
        combined_transcript_state = gr.State()
        filtered_transcription_state = gr.State()
        integrity_data_state = gr.State()

        contact_state = gr.State()

        # (MODIFICADO) Nuevo estado para controlar si ya se registró la visita
        visit_already_registered = gr.State(False)  # <-- Nuevo

        # ========== PASO 1 ==========
        with gr.Group(visible=True) as step1_container:
            gr.Markdown("### Paso 1: Subir audio y verificar contenido", elem_classes="step-heading-1-5")
            with gr.Group(elem_classes="white-container"):
                audio_input_main = gr.Audio(type="filepath", label="Audio Principal")
                btn_process_main = gr.Button("Procesar audio y verificar contenido", elem_classes="btn-spacing")
                main_audio_msg = gr.Markdown()
                verify_msg = gr.Markdown()
                account_input = gr.Textbox(label="Nombre de la cuenta (detectada o manual)")

        # ========== PASO 1.1 (Opcional) ==========
        with gr.Group(visible=False) as step3_container:
            gr.Markdown("### Pendiente de completar: Subir audio complementario y verificar contenido", elem_classes="step-heading-1-5")
            with gr.Group(elem_classes="white-container"):
                audio_input_second = gr.Audio(type="filepath", label="Audio complementario")
                btn_process_second = gr.Button("Procesar audio complementario y verificar contenido", elem_classes="btn-spacing")
                second_audio_msg = gr.Markdown()
                recheck_msg = gr.Markdown()

        # ========== PASO 2 ==========
        with gr.Group(visible=False) as step4_container:
            gr.Markdown("### Paso 2: Generar transcripción", elem_classes="step-heading-1-5")
            with gr.Group(elem_classes="white-container"):
                final_transcript_box = gr.Textbox(label="Transcripción final", lines=5)
                btn_filter_final = gr.Button("Obtener transcripción", elem_classes="btn-spacing")

        # ========== PASO 3 ==========
        with gr.Group(visible=False) as step5_container:
            gr.Markdown("### Paso 3: Generar ficha resumen de la visita", elem_classes="step-heading-1-5")
            with gr.Group(elem_classes="white-container"):
                summary_textbox = gr.Textbox(label="Resumen estructurado", lines=10)
                summary_file_out = gr.File(label="Descargar resumen de la visita")
                gen_summary_msg = gr.Markdown()
                btn_generate_summary = gr.Button("Generar resumen", elem_classes="btn-spacing")

        # ========== PASO 4 ==========
        with gr.Group(visible=False) as step6_container:
            gr.Markdown("### Paso 4: Registrar visita en el histórico", elem_classes="step-heading-1-5")
            with gr.Group(elem_classes="white-container"):
                register_msg = gr.Markdown()
                btn_register = gr.Button("Registrar visita")

        # ----------------------------------------
        # FUNCIONES / CALLBACKS
        # ----------------------------------------
        def check_integrity_callback(transcript):
            if not transcript:
                return (
                    "No hay transcripción para verificar.",
                    {"all_ok": False},
                    gr.update(visible=False),
                    gr.update(visible=False)
                )

            detected_acc = detect_entities(transcript)
            cuenta_no_detectada = detected_acc.lower() in ["cuenta no detectada", "ana", "luis", "maría"]
            detected_acc_msg = (
                "⚠️ Cuenta no detectada. Por favor, ingrésala manualmente."
                if cuenta_no_detectada
                else f"La cuenta es identificada como '{detected_acc}'."
            )

            data = check_missing_points(transcript)
            repint = data.get("reporte_integridad", [])
            recomend = data.get("recomendacion_global", "")
            msg = "### Resultado de la verificación\n"

            faltan_puntos = {
                "cuenta": False,
                "contacto": False,
                "productos": False,
                "feedback": False
            }

            for item in repint:
                punto = item["punto"].lower()
                estado = item["estado"].lower()
                comentario = item["comentario"]
                icon = "✅" if estado == "presente" else "❌"

                msg += f"- **{punto}**: {estado.capitalize()} {icon}\n"
                if comentario:
                    msg += f"  - {comentario}\n"

                if estado == "faltante":
                    if "cuenta" in punto:
                        faltan_puntos["cuenta"] = True
                    elif "contacto" in punto or "departamento" in punto:
                        faltan_puntos["contacto"] = True
                    elif "productos" in punto or "líneas discutidas" in punto:
                        faltan_puntos["productos"] = True
                    elif "feedback" in punto or "acciones futuras" in punto:
                        faltan_puntos["feedback"] = True

            msg += f"\n**Recomendación global:** {recomend}\n\n{detected_acc_msg}"
            data["all_ok"] = not any(faltan_puntos.values())

            # Caso 1: Falta solo la cuenta
            if faltan_puntos["cuenta"] and not (faltan_puntos["contacto"] or faltan_puntos["productos"] or faltan_puntos["feedback"]):
                return msg, data, gr.update(visible=False), gr.update(visible=True)

            # Caso 2: Falta algún otro punto
            if faltan_puntos["contacto"] or faltan_puntos["productos"] or faltan_puntos["feedback"]:
                return msg, data, gr.update(visible=True), gr.update(visible=False)

            # Caso 3: Todo OK
            return msg, data, gr.update(visible=False), gr.update(visible=True)

        def recheck_integrity_callback(transcript):
            if not transcript:
                return "No hay transcripción para verificar.", {"all_ok": False}, gr.update(visible=False)
            data = check_missing_points(transcript)
            repint = data.get("reporte_integridad", [])
            recomend = data.get("recomendacion_global", "")
            msg = "### Resultado de la verificación\n"

            all_ok = True
            for item in repint:
                punto = item.get("punto", "")
                estado = item.get("estado", "desconocido")
                comentario = item.get("comentario", "")
                icon = "✅" if estado.lower() == "presente" else "❌"
                msg += f"- **{punto}**: {estado.capitalize()} {icon}\n"
                if comentario:
                    msg += f"  - {comentario}\n"
                if estado.lower() == "faltante":
                    all_ok = False

            msg += f"\n**Recomendación global:** {recomend}\n"
            data["all_ok"] = all_ok

            if not all_ok:
                return msg, data, gr.update(visible=False)
            return msg, data, gr.update(visible=True)

        # ----------------------------------------
        # PROCESAR AUDIO PRINCIPAL
        # ----------------------------------------
        def process_main_audio_callback(audio, representative):
            progress_msg = ""
            if not representative:
                progress_msg = "⚠️ Debes iniciar sesión primero."
                yield progress_msg, "", None, None, "", {"all_ok": False}, gr.update(visible=False), gr.update(visible=False), ""
                return
            if not audio:
                progress_msg = "No has subido ningún audio principal."
                yield progress_msg, "", None, None, "", {"all_ok": False}, gr.update(visible=False), gr.update(visible=False), ""
                return

            progress_msg += "Procesando audio...\n"
            yield progress_msg, "", None, None, "", {"all_ok": False}, gr.update(visible=False), gr.update(visible=False), ""
            time.sleep(1)

            progress_msg += "Transcribiendo audio...\n"
            yield progress_msg, "", None, None, "", {"all_ok": False}, gr.update(visible=False), gr.update(visible=False), ""
            raw_transcription = call_openai_audio_transcription(audio)

            detected_acc = detect_entities(raw_transcription)
            if detected_acc.lower() == "cuenta no detectada":
                detected_acc = "Sin cuenta detectada"

            detected_contact = detect_contact(raw_transcription)

            main_transcript = raw_transcription
            combined = raw_transcription
            time.sleep(1)

            progress_msg += "Verificando contenido...\n"
            yield progress_msg, "", None, None, "", {"all_ok": False}, gr.update(visible=False), gr.update(visible=False), ""
            verify_m, integrity_d, step3_vis, step4_vis = check_integrity_callback(main_transcript)
            time.sleep(1)

            progress_msg += "✅ Proceso completado.\n"
            yield (
                progress_msg,           
                verify_m,               
                main_transcript,        
                combined,               
                detected_acc,           
                integrity_d,            
                step3_vis,              
                step4_vis,              
                detected_contact        
            )

        btn_process_main.click(
            fn=process_main_audio_callback,
            inputs=[audio_input_main, user_session],
            outputs=[
                main_audio_msg,          
                verify_msg,              
                main_transcript_state,   
                combined_transcript_state,
                account_input,           
                integrity_data_state,    
                step3_container,         
                step4_container,         
                contact_state            
            ],
            queue=True
        )

        # ----------------------------------------
        # PROCESAR AUDIO SECUNDARIO
        # ----------------------------------------
        def process_second_audio_callback(audio_second, representative, combined):
            progress_msg = ""
            if not representative:
                progress_msg = "⚠️ Debes iniciar sesión primero."
                yield progress_msg, "", combined, {"all_ok":False}, gr.update(visible=False)
                return
            if not audio_second:
                progress_msg = "No has subido ningún audio secundario."
                yield progress_msg, "", combined, {"all_ok":False}, gr.update(visible=False)
                return

            progress_msg += "Procesando audio secundario...\n"
            yield progress_msg, "", combined, {"all_ok":False}, gr.update(visible=False)
            time.sleep(1)

            progress_msg += "Transcribiendo audio...\n"
            yield progress_msg, "", combined, {"all_ok":False}, gr.update(visible=False)
            raw_transcription_2 = call_openai_audio_transcription(audio_second)
            new_combined = (combined or "") + "\n" + raw_transcription_2
            time.sleep(1)

            progress_msg += "Verificando contenido...\n"
            yield progress_msg, "", new_combined, {"all_ok":False}, gr.update(visible=False)
            recheck_m, integrity_d, step4_vis = recheck_integrity_callback(new_combined)
            time.sleep(1)

            progress_msg += "✅ Proceso completado.\n"
            yield progress_msg, recheck_m, new_combined, integrity_d, step4_vis

        btn_process_second.click(
            fn=process_second_audio_callback,
            inputs=[audio_input_second, user_session, combined_transcript_state],
            outputs=[
                second_audio_msg,
                recheck_msg,
                combined_transcript_state,
                integrity_data_state,
                step4_container
            ],
            queue=True
        )

        # ----------------------------------------
        # OBTENER TRANSCRIPCIÓN FILTRADA
        # ----------------------------------------
        def filter_final_callback(combined, representative):
            if not representative:
                return "⚠️ Debes iniciar sesión primero."
            if not combined:
                return "No hay transcripción combinada."
            filtered = semantic_filter_transcription(combined)
            return filtered

        btn_filter_final.click(
            fn=filter_final_callback,
            inputs=[combined_transcript_state, user_session],
            outputs=[final_transcript_box]
        )

        def store_filtered_and_show_step5(x):
            return x, gr.update(visible=True)

        final_transcript_box.change(
            fn=store_filtered_and_show_step5,
            inputs=[final_transcript_box],
            outputs=[filtered_transcription_state, step5_container]
        )

        # ----------------------------------------
        # GENERAR RESUMEN
        # ----------------------------------------
        def generate_summary_callback(filtered_transcription, user_account, representative):
            if not representative:
                return "", None, "⚠️ Debes iniciar sesión primero.", gr.update(visible=True), gr.update(visible=False)
            if not filtered_transcription:
                return "", None, "No hay transcripción final filtrada.", gr.update(visible=True), gr.update(visible=False)

            final_account = user_account.strip()
            if not final_account:
                final_account = "Cuenta no detectada"

            visit_date = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
            acc_norm = normalize_account(final_account)
            subset_df = df[df["Cuenta_Normalizada"] == acc_norm]
            if subset_df.empty:
                historical_summary = "No hay registros previos disponibles."
            else:
                historical_summary_list = subset_df['Puntos Clave'].dropna().tolist()
                historical_summary = ' '.join(historical_summary_list)

            summary_text = generate_plain_text_summary(
                filtered_transcription,
                final_account,
                visit_date,
                historical_summary
            )
            summary_file = save_text_to_file(summary_text, "resumen_visita.txt")
            if not os.path.exists(summary_file):
                return "", None, "⚠️ Error al generar el archivo de resumen.", gr.update(visible=False), gr.update(visible=False)
            summary_file_out = gr.File(value=summary_file, label="Descargar resumen de la visita")
            return summary_text, summary_file, "✅ Resumen generado con éxito.", gr.update(visible=True), gr.update(visible=True)

        btn_generate_summary.click(
            fn=generate_summary_callback,
            inputs=[filtered_transcription_state, account_input, user_session],
            outputs=[summary_textbox, summary_file_out, gen_summary_msg, step5_container, step6_container]
        )

        # ----------------------------------------
        # (MODIFICADO) REGISTRAR VISITA
        # ----------------------------------------
        def register_in_history_callback(final_account, representative, corrected_text, integrity_data, contact_value, already_registered):
            # Si ya se ha registrado, no repetimos
            if already_registered:  # <-- Uso del nuevo estado
                return "¡La visita ya está registrada y no se puede registrar dos veces!", True, gr.update(interactive=False)

            if not representative:
                return "⚠️ Debes iniciar sesión primero.", False, gr.update(interactive=True)
            if not final_account.strip():
                return "No se ha indicado la cuenta final para registrar.", False, gr.update(interactive=True)
            if not integrity_data or not integrity_data.get("all_ok"):
                return "❌ No se puede registrar: faltan puntos mínimos de la visita.", False, gr.update(interactive=True)

            visit_date = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
            puntos_clave = extract_puntos_clave(corrected_text)

            # Registramos la visita en CSV, con la columna 'Contacto'
            add_visit_to_csv(final_account, representative, visit_date, puntos_clave, contact_value)
            update_vectorstore_with_new_record(final_account, puntos_clave)

            # Marcamos el estado como True (para indicar que ya se registró)
            return "¡Visita registrada con éxito!", True, gr.update(interactive=False)

        # (MODIFICADO) btn_register con 3 salidas: mensaje, estado, propiedades del botón
        btn_register.click(
            fn=register_in_history_callback,
            inputs=[account_input, user_session, summary_textbox, integrity_data_state, contact_state, visit_already_registered],
            outputs=[register_msg, visit_already_registered, btn_register]
        )

    # ==========================================
    # PESTAÑA: "Preguntar al asistente" (RAG)
    # ==========================================
    with gr.Tab("Preguntar al asistente", visible=False) as tab_rag:
        gr.Markdown("### Consulta utilizando todos los datos disponibles de la cuenta")
        rag_query = gr.Textbox(label="Pregunta", placeholder="Escribe tu pregunta aquí...")
        account_for_query = gr.Textbox(label="Nombre de la cuenta (opcional)", placeholder="Si se deja vacío se intentará detectar")
        rag_response = gr.Textbox(label="Respuesta del asistente", lines=10)
        
        def answer_commercial_query_local(query, account_name):
            if account_name.strip() == "":
                detected_account = detect_entities(query)
                if detected_account in ["Cuenta no detectada", "Sin cuenta detectada"]:
                    detected_account = "Cuenta desconocida"
            else:
                detected_account = account_name
            combined_context = retrieve_account_history(detected_account, 5)
            prompt = f"""
Eres un asistente comercial experto en Sandoz, con amplia experiencia en estrategias para sus cuentas cliente.
Utiliza la siguiente información histórica combinada (visitas y ventas) para la cuenta {detected_account}:
----------------------------------------------------
{combined_context}
----------------------------------------------------
Basándote en esta información y en la siguiente consulta:
"{query}"
Proporciona una respuesta concisa y profesional, incluyendo sugerencias estratégicas y recomendaciones específicas para la cuenta.
Asegúrate de utilizar todo el contexto histórico proporcionado.
"""
            response = call_chat_api_simple(prompt)
            return response
        
        rag_btn = gr.Button("Obtener respuesta")
        rag_btn.click(
            fn=answer_commercial_query_local,
            inputs=[rag_query, account_for_query],
            outputs=[rag_response]
        )
    
    # ==========================================
    # PESTAÑA: "Consultar registros históricos"
    # ==========================================
    with gr.Tab("Consultar registros históricos más recientes", visible=False) as tab_history:
        account_query = gr.Textbox(label="Nombre de la cuenta", placeholder="Escribe el nombre de la cuenta...")
        limit_input = gr.Number(label="Número de registros a mostrar", value=5)
        records_output = gr.Textbox(label="Histórico combinado", lines=10)
        
        def retrieve_account_history_combined(account_name, limit):
            return retrieve_account_history(account_name, limit)
        
        retrieval_btn = gr.Button("Recuperar registros")
        retrieval_btn.click(
            fn=retrieve_account_history_combined,
            inputs=[account_query, limit_input],
            outputs=[records_output]
        )
    
    # ==========================================
    # PESTAÑA: "Dashboard"
    # ==========================================
    with gr.Tab("Dashboard", visible=False) as tab_dashboard:
        gr.Markdown("### Dashboard: Gráficos")
        with gr.Row():
            dashboard_plot1 = gr.Plot(label="Número de visitas por comercial", show_label=False)
            dashboard_plot2 = gr.Plot(label="Facturación por cuenta", show_label=False)
            dashboard_plot3 = gr.Plot(label="Facturación por producto", show_label=False)
        with gr.Row():
            dashboard_plot4 = gr.Plot(label="Gráfico Interactivo", show_label=False)
        
        dashboard_btn = gr.Button("Actualizar Dashboard")
        dashboard_btn.click(
            fn=dashboard_overview,
            inputs=[],
            outputs=[dashboard_plot1, dashboard_plot2, dashboard_plot3, dashboard_plot4]
        )
    
    # ==========================================
    # PESTAÑA: "Cerrar sesión"
    # ==========================================
    with gr.Tab("Cerrar sesión", visible=False) as tab_logout:
        gr.Markdown("## Cerrar sesión")
        
        logout_button = gr.Button("Cerrar sesión", elem_id="logout_button")
        logout_message = gr.Markdown()
        
        def logout_fn():
            return (
                None,                          # user_session -> None
                "✅ Sesión cerrada con éxito.", # logout_message
                "",                            # main_transcript_state
                "",                            # combined_transcript_state
                "",                            # filtered_transcription_state
                "",                            # integrity_data_state
                "",                            # account_input
                ""                             # contact_state
            )

        logout_button.click(
            fn=logout_fn,
            inputs=[],
            outputs=[
                user_session, logout_message,
                main_transcript_state, combined_transcript_state,
                filtered_transcription_state, integrity_data_state,
                account_input,
                contact_state
            ]
        )

        gr.HTML('''
        <script>
          const logoutChannel = new BroadcastChannel("logout_channel");
          logoutChannel.onmessage = (event) => {
            if (event.data === "logout") {
              localStorage.clear();
              sessionStorage.clear();
              window.location.reload();
            }
          };
          function broadcastLogout() {
            logoutChannel.postMessage("logout");
          }
        </script>
        ''')

    # ------------------------------------------
    # Control de visibilidad de pestañas (login)
    # ------------------------------------------
    def update_visibility(representative):
        is_logged = representative is not None
        return (
            gr.update(visible=is_logged),  # tab_audio
            gr.update(visible=is_logged),  # tab_rag
            gr.update(visible=is_logged),  # tab_history
            gr.update(visible=is_logged),  # tab_dashboard
            gr.update(visible=is_logged)   # tab_logout
        )
    
    user_session.change(
        update_visibility,
        inputs=[user_session],
        outputs=[tab_audio, tab_rag, tab_history, tab_dashboard, tab_logout]
    )

#Lanzar la app
demo.launch()
