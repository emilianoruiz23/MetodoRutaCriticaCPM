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

st.title("Calculadora PERT/CPM Directa con Reporte PDF")
st.markdown("Agrega tus actividades directamente en la tabla. La red y los cálculos se actualizarán automáticamente.")

# --- 1. ENTRADA DE DATOS DIRECTA ---
st.header("1. Ingreso de Actividades")

# Creamos una tabla vacía con los encabezados listos
df_inicial = pd.DataFrame(columns=[
    "Actividad", "De_Nodo", "A_Nodo", "Optimista_a", "MasProbable_m", "Pesimista_b"
])

st.info("💡 Consejo: Haz clic en el botón '+' al final de la tabla para agregar una nueva actividad.")

# El usuario interactúa directamente con esta tabla
edited_df = st.data_editor(df_inicial, num_rows="dynamic", use_container_width=True)

# --- 2. CÁLCULOS ---
def calculate_pert_times(df):
    results_df = df.copy()
    if results_df.empty: return results_df
    
    # Convertir a números por si acaso
    for col in ['Optimista_a', 'MasProbable_m', 'Pesimista_b']:
        results_df[col] = pd.to_numeric(results_df[col], errors='coerce').fillna(0)
        
    results_df['te'] = (results_df['Optimista_a'] + 4 * results_df['MasProbable_m'] + results_df['Pesimista_b']) / 6
    results_df['variance'] = ((results_df['Pesimista_b'] - results_df['Optimista_a']) / 6)**2
    return results_df

results_pert = calculate_pert_times(edited_df)

def find_critical_path(df):
    if df.empty: return None, None, None, None, None
    G = nx.DiGraph()
    for index, row in df.iterrows():
        # Validar que los nodos no estén vacíos antes de agregarlos
        if pd.notna(row['De_Nodo']) and pd.notna(row['A_Nodo']) and str(row['De_Nodo']).strip() != "" and str(row['A_Nodo']).strip() != "":
            G.add_edge(str(row['De_Nodo']), str(row['A_Nodo']), id=str(row['Actividad']), weight=row['te'])

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

# --- 3. VISUALIZACIÓN Y RESULTADOS ---
if cp_nodes and not results_pert.empty:
    st.header("2. Visualización de la Red")
    
    # Dibujar la red instantáneamente (sin animaciones)
    dot = graphviz.Digraph(graph_attr={'rankdir': 'LR'})
    
    for index, row in results_pert.iterrows():
        if pd.notna(row['De_Nodo']) and pd.notna(row['A_Nodo']) and str(row['De_Nodo']).strip() != "" and str(row['A_Nodo']).strip() != "":
            actividad = str(row['Actividad'])
            es_critica = actividad in cp_act_ids
            color = 'red' if es_critica else 'black'
            penwidth = '2.0' if es_critica else '1.0'
            
            dot.node(str(row['De_Nodo']), shape='circle', style='filled', fillcolor='lightcyan')
            dot.node(str(row['A_Nodo']), shape='circle', style='filled', fillcolor='lightcyan')
            
            etiqueta = f"{actividad}\nte={row['te']:.2f}"
            dot.edge(str(row['De_Nodo']), str(row['A_Nodo']), label=etiqueta, color=color, fontcolor=color, penwidth=penwidth)

    st.graphviz_chart(dot)

    st.header("3. Análisis Probabilístico y Reporte PDF")
    col1, col2 = st.columns(2)
    col1.metric("Duración Total Esperada (μ)", f"{project_mean:.2f}")
    col2.metric("Desviación Estándar (σ)", f"{project_sd:.4f}")
    st.markdown(f"**Ruta Crítica:** `{path_str}`")

    target_time = st.number_input("Tiempo Objetivo para finalizar el proyecto:", value=float(np.ceil(project_mean)), step=0.5)

    z_score = (target_time - project_mean) / project_sd if project_sd > 0 else 0
    prob_le = norm.cdf(z_score) * 100

    def create_plot():
        fig, ax = plt.subplots(figsize=(8, 4))
        # Validar si la desviación estándar es 0 para evitar errores en la gráfica
        sd_plot = project_sd if project_sd > 0 else 0.01 
        
        x = np.linspace(project_mean - 4*sd_plot, project_mean + 4*sd_plot, 200)
        ax.plot(x, norm.pdf(x, project_mean, sd_plot), 'b-')
        x_fill = np.linspace(project_mean - 4*sd_plot, target_time, 200)
        ax.fill_between(x_fill, norm.pdf(x_fill, project_mean, sd_plot), color='lightblue', alpha=0.5)
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
