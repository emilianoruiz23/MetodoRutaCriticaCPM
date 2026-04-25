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

st.set_page_config(layout="wide", page_title="Generador PERT/CPM")

st.title("Calculadora PERT/CPM con Reporte PDF")
st.markdown("Ingresa los tiempos optimista, más probable y pesimista. La app calculará la ruta crítica y generará un reporte descargable.")

# --- ENTRADA DE DATOS MANUALES ---
st.header("1. Ingreso de Actividades (PERT)")
default_activities = [
    {"Actividad": "A", "De_Nodo": "Inicio", "A_Nodo": "Central", "Optimista_a": 3.0, "MasProbable_m": 4.16, "Pesimista_b": 6.0},
    {"Actividad": "B", "De_Nodo": "Central", "A_Nodo": "Superior", "Optimista_a": 2.0, "MasProbable_m": 2.3, "Pesimista_b": 3.0},
    {"Actividad": "C", "De_Nodo": "Central", "A_Nodo": "Abajo_Izq", "Optimista_a": 3.0, "MasProbable_m": 3.5, "Pesimista_b": 5.0},
    {"Actividad": "D", "De_Nodo": "Central", "A_Nodo": "Final", "Optimista_a": 1.0, "MasProbable_m": 1.25, "Pesimista_b": 2.0},
    {"Actividad": "E", "De_Nodo": "Superior", "A_Nodo": "Final", "Optimista_a": 5.0, "MasProbable_m": 5.66, "Pesimista_b": 7.0},
    {"Actividad": "Ficticia", "De_Nodo": "Abajo_Izq", "A_Nodo": "Superior", "Optimista_a": 0.0, "MasProbable_m": 0.0, "Pesimista_b": 0.0},
]

df_input = pd.DataFrame(default_activities)
edited_df = st.data_editor(df_input, num_rows="dynamic")

# --- CÁLCULOS PERT ---
def calculate_pert_times(df):
    results_df = df.copy()
    # Asegurar que las columnas numéricas sean float
    cols_numericas = ['Optimista_a', 'MasProbable_m', 'Pesimista_b']
    for col in cols_numericas:
        results_df[col] = pd.to_numeric(results_df[col], errors='coerce').fillna(0)
        
    results_df['te'] = (results_df['Optimista_a'] + 4 * results_df['MasProbable_m'] + results_df['Pesimista_b']) / 6
    results_df['variance'] = ((results_df['Pesimista_b'] - results_df['Optimista_a']) / 6)**2
    return results_df

results_pert = calculate_pert_times(edited_df)

# --- CPM ---
def find_critical_path(df):
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

if cp_nodes:
    # --- VISUALIZACIÓN DE LA RED EN LA APP ---
    st.header("2. Visualización de la Red")
    
    # Usamos Graphviz para dibujar el diagrama
    dot = graphviz.Digraph(attr={'rankdir': 'LR'}) # De izquierda a derecha
    
    for index, row in results_pert.iterrows():
        actividad = row['Actividad']
        es_critica = actividad in cp_act_ids
        
        color = 'red' if es_critica else 'black'
        penwidth = '2.0' if es_critica else '1.0'
        
        dot.node(str(row['De_Nodo']), shape='circle', style='filled', fillcolor='lightcyan')
        dot.node(str(row['A_Nodo']), shape='circle', style='filled', fillcolor='lightcyan')
        
        etiqueta = f"{actividad}\nte={row['te']:.2f}"
        dot.edge(str(row['De_Nodo']), str(row['A_Nodo']), label=etiqueta, color=color, fontcolor=color, penwidth=penwidth)

    st.graphviz_chart(dot)

    # --- RESULTADOS Y GENERACIÓN DE PDF ---
    st.header("3. Análisis Probabilístico y Reporte")
    col1, col2 = st.columns(2)
    col1.metric("Duración Total Esperada (µ)", f"{project_mean:.2f}")
    col2.metric("Desviación Estándar (s)", f"{project_sd:.4f}")
    st.markdown(f"**Ruta Crítica:** `{path_str}`")

    target_time = st.number_input("Tiempo Objetivo para finalizar el proyecto:", value=float(np.ceil(project_mean)), step=0.5)

    z_score = (target_time - project_mean) / project_sd
    prob_le = norm.cdf(z_score) * 100

    # Gráfico interno (no se muestra con st.pyplot)
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

    # Función PDF
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
        
        # Generar y guardar el gráfico temporalmente para el PDF
        fig = create_plot()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            fig.savefig(tmpfile.name, format="png")
            pdf.image(tmpfile.name, w=170)
        os.unlink(tmpfile.name)
        
        return pdf.output(dest='S').encode('latin1')

    if st.button("Generar y Descargar Reporte PDF"):
        pdf_bytes = generate_pdf()
        st.download_button(label="Descargar PDF", data=pdf_bytes, file_name="Reporte_Proyecto.pdf", mime="application/pdf")
else:
    st.warning("La red tiene un ciclo o falta de conexión. Revisa los nodos.")