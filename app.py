import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import networkx as nx
import matplotlib.pyplot as plt
from fpdf import FPDF
import graphviz
import tempfile
import os
import time  # <-- Importante para la animación

st.set_page_config(layout="wide", page_title="Generador PERT/CPM")

st.title("Calculadora PERT/CPM Animada y Reporte PDF")

# --- 1. ENTRADA DE DATOS (FORMULARIO Y TABLA) ---
st.header("1. Ingreso de Actividades (PERT)")

# Crear una "memoria" para guardar las actividades si aún no existe
if 'df_datos' not in st.session_state:
    st.session_state.df_datos = pd.DataFrame(columns=[
        "Actividad", "De_Nodo", "A_Nodo", "Optimista_a", "MasProbable_m", "Pesimista_b"
    ])

# Formulario interactivo para agregar actividades paso a paso
with st.expander("➕ Haz clic aquí para agregar una actividad en el formulario", expanded=True):
    with st.form("form_actividad", clear_on_submit=True):
        col1, col2, col3 = st.columns(3)
        act = col1.text_input("Nombre de Actividad (ej. A)")
        de_nodo = col2.text_input("De Nodo (ej. Inicio)")
        a_nodo = col3.text_input("A Nodo (ej. Central)")
        
        col4, col5, col6 = st.columns(3)
        opt = col4.number_input("T. Optimista (a)", min_value=0.0, step=0.1)
        prob = col5.number_input("T. Más Probable (m)", min_value=0.0, step=0.1)
        pes = col6.number_input("T. Pesimista (b)", min_value=0.0, step=0.1)
        
        # Al presionar el botón, se añade a nuestra "memoria"
        if st.form_submit_button("Agregar a la lista"):
            if act and de_nodo and a_nodo:
                nueva_fila = pd.DataFrame([{
                    "Actividad": act, "De_Nodo": de_nodo, "A_Nodo": a_nodo,
                    "Optimista_a": opt, "MasProbable_m": prob, "Pesimista_b": pes
                }])
                st.session_state.df_datos = pd.concat([st.session_state.df_datos, nueva_fila], ignore_index=True)
                st.rerun() # Recarga la página para mostrar los cambios
            else:
                st.warning("Por favor llena al menos el Nombre, De Nodo y A Nodo.")

st.markdown("**Lista de actividades:** (Puedes editar los números o borrar filas seleccionando la casilla izquierda y presionando 'Supr')")
# Mostrar la tabla permitiendo edición directa
edited_df = st.data_editor(st.session_state.df_datos, num_rows="dynamic", use_container_width=True)

# Sincronizar la tabla editada con nuestra memoria
st.session_state.df_datos = edited_df

# --- 2. CÁLCULOS PERT Y CPM ---
def calculate_pert_times(df):
    results_df = df.copy()
    if results_df.empty: return results_df
    
    cols_numericas = ['Optimista_a', 'MasProbable_m', 'Pesimista_b']
    for col in cols_numericas:
        results_df[col] = pd.to_numeric(results_df[col], errors='coerce').fillna(0)
        
    results_df['te'] = (results_df['Optimista_a'] + 4 * results_df['MasProbable_m'] + results_df['Pesimista_b']) / 6
    results_df['variance'] = ((results_df['Pesimista_b'] - results_df['Optimista_a']) / 6)**2
    return results_df

results_pert = calculate_pert_times(edited_df)

def find_critical_path(df):
    if df.empty: return None, None, None, None, None
    G = nx.DiGraph()
    for index, row in df.iterrows():
        G.add_edge(row['De_Nodo'], row['A_Nodo'], id=row['Actividad'], weight=row['te'])

    try:
        cp_nodes = nx.dag_longest_path(G, weight='weight')
        total_variance = 0
        cp_act_ids = []
        path_string_list = []
        
        for i in range(len(cp_nodes) - 1):
            u, v = cp_nodes[i], cp_nodes[i+1]
            edge_data = G.get_edge_data(u, v)
            act_id = edge_data['id']
            cp_act_ids.append(act_id)
            
            activity_var = df[df['Actividad'] == act_id]['variance'].values[0]
            total_variance += activity_var
            path_string_list.append(f"{u} -> {act_id} -> {v}")

        project_mean = nx.dag_longest_path_length(G, weight='weight')
        return cp_nodes, cp_act_ids, project_mean, np.sqrt(total_variance), " | ".join(path_string_list)
    except:
        return None, None, None, None, None

cp_nodes, cp_act_ids, project_mean, project_sd, path_str = find_critical_path(results_pert)

# --- 3. VISUALIZACIÓN ANIMADA Y RESULTADOS ---
if cp_nodes and not results_pert.empty:
    st.header("2. Visualización de la Red")
    
    # Botón para activar la animación
    animar = st.button("▶️ Construir Red Paso a Paso (Animación)")
    
    # Espacio reservado en blanco donde se inyectará el gráfico
    espacio_grafico = st.empty()
    
    dot = graphviz.Digraph(graph_attr={'rankdir': 'LR'})
    
    for index, row in results_pert.iterrows():
        actividad = row['Actividad']
        es_critica = actividad in cp_act_ids
        color = 'red' if es_critica else 'black'
        penwidth = '2.0' if es_critica else '1.0'
        
        dot.node(str(row['De_Nodo']), shape='circle', style='filled', fillcolor='lightcyan')
        dot.node(str(row['A_Nodo']), shape='circle', style='filled', fillcolor='lightcyan')
        
        etiqueta = f"{actividad}\nte={row['te']:.2f}"
        dot.edge(str(row['De_Nodo']), str(row['A_Nodo']), label=etiqueta, color=color, fontcolor=color, penwidth=penwidth)
        
        if animar:
            # Dibuja el estado actual y hace una pausa
            espacio_grafico.graphviz_chart(dot)
            time.sleep(0.8) # Pausa de 0.8 segundos entre cada flecha
            
    # Asegurar que se dibuje completo al final o si no se presionó animar
    espacio_grafico.graphviz_chart(dot)

    st.header("3. Análisis Probabilístico y Reporte")
    col1, col2 = st.columns(2)
    col1.metric("Duración Total Esperada (μ)", f"{project_mean:.2f}")
    col2.metric("Desviación Estándar (σ)", f"{project_sd:.4f}")
    st.markdown(f"**Ruta Crítica:** `{path_str}`")

    target_time = st.number_input("Tiempo Objetivo para finalizar el proyecto:", value=float(np.ceil(project_mean)), step=0.5)

    z_score = (target_time - project_mean) / project_sd
    prob_le = norm.cdf(z_score) * 100

    def create_plot():
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.linspace(project_mean - 4*project_sd, project_mean + 4*project_sd, 200)
        ax.plot(x, norm.pdf(x, project_mean, project_sd), 'b-')
        x_fill = np.linspace(project_mean - 4*project_sd, target_time, 200)
        ax.fill_between(x_fill, norm.pdf(x_fill, project_mean, project_sd), color='lightblue', alpha=0.5)
        ax.axvline(x=project_mean, color='green', linestyle='--')
        ax.axvline(x=target_time, color='red', linestyle='-')
        ax.set_title("Distribución Normal del Proyecto")
        return fig

    def generate_pdf():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(190, 10, 'Reporte PERT/CPM', 0, 1, 'C')
        pdf.ln(5)
        
        pdf.set_font('Arial', '', 12)
        pdf.cell(190, 8, f"Duracion Esperada: {project_mean:.2f}", 0, 1)
        pdf.cell(190, 8, f"Ruta Critica: {path_str}", 0, 1)
        pdf.cell(190, 8, f"Probabilidad (T <= {target_time}): {prob_le:.2f}%", 0, 1)
        pdf.ln(5)
        
        fig = create_plot()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            fig.savefig(tmpfile.name, format="png")
            pdf.image(tmpfile.name, w=170)
        os.unlink(tmpfile.name)
        
        return pdf.output(dest='S').encode('latin1')

    if st.button("Generar y Descargar Reporte PDF"):
        pdf_bytes = generate_pdf()
        st.download_button(label="Descargar PDF", data=pdf_bytes, file_name="Reporte_Proyecto.pdf", mime="application/pdf")
elif not results_pert.empty:
    st.warning("La red tiene un ciclo o falta de conexión. Revisa los nodos.")
else:
    st.info("Agrega actividades en el formulario de arriba para comenzar.")
