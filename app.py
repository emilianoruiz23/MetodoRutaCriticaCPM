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

df_inicial = pd.DataFrame(columns=[
    "Actividad", "De_Nodo", "A_Nodo", "Optimista_a", "MasProbable_m", "Pesimista_b"
])

st.info("💡 Consejo: Haz clic en el botón '+' al final de la tabla para agregar una nueva actividad. Asegúrate de conectar los nodos de forma lógica (ej. 1->2, 2->3).")

edited_df = st.data_editor(df_inicial, num_rows="dynamic", use_container_width=True)

# Opciones adicionales para el usuario
mostrar_pasos = st.checkbox("¿Mostrar detalles de revisión hacia adelante y hacia atrás (ES, EF, LS, LF, Holgura)?", value=True)

# --- 2. CÁLCULOS ---
def calculate_pert_times(df):
    results_df = df.copy()
    if results_df.empty: return results_df
    
    for col in ['Optimista_a', 'MasProbable_m', 'Pesimista_b']:
        results_df[col] = pd.to_numeric(results_df[col], errors='coerce').fillna(0)
        
    results_df['te'] = (results_df['Optimista_a'] + 4 * results_df['MasProbable_m'] + results_df['Pesimista_b']) / 6
    results_df['variance'] = ((results_df['Pesimista_b'] - results_df['Optimista_a']) / 6)**2
    return results_df

results_pert = calculate_pert_times(edited_df)

def find_critical_path_and_cpm_metrics(df):
    if df.empty: return None, None, None, None, None, None
    G = nx.DiGraph()
    for index, row in df.iterrows():
        if pd.notna(row['De_Nodo']) and pd.notna(row['A_Nodo']) and str(row['De_Nodo']).strip() != "" and str(row['A_Nodo']).strip() != "":
            G.add_edge(str(row['De_Nodo']), str(row['A_Nodo']), id=str(row['Actividad']), weight=row['te'])

    try:
        # Validar si hay ciclos
        if not nx.is_directed_acyclic_graph(G):
            return None, None, None, None, None, None

        # 1. REVISIÓN HACIA ADELANTE (Early Times para los nodos)
        E = {n: 0 for n in G.nodes()}
        for u in nx.topological_sort(G):
            for v in G.successors(u):
                weight = G[u][v]['weight']
                E[v] = max(E[v], E[u] + weight)

        project_mean = max(E.values())

        # 2. REVISIÓN HACIA ATRÁS (Late Times para los nodos)
        L = {n: project_mean for n in G.nodes()}
        for u in reversed(list(nx.topological_sort(G))):
            for v in G.successors(u):
                weight = G[u][v]['weight']
                L[u] = min(L[u], L[v] - weight)

        # 3. Calcular métricas por actividad (arco)
        cpm_results = []
        total_variance = 0
        cp_act_ids = []
        
        for u, v, data in G.edges(data=True):
            act_id = data['id']
            t = data['weight']
            
            es = E[u]            # Early Start
            ef = es + t          # Early Finish
            lf = L[v]            # Late Finish
            ls = lf - t          # Late Start
            holgura = ls - es    # Slack
            
            es_critica = abs(holgura) < 1e-6
            
            if es_critica:
                cp_act_ids.append(act_id)
                activity_var = df[df['Actividad'] == act_id]['variance'].values[0]
                total_variance += activity_var
                
            cpm_results.append({
                'Actividad': act_id,
                'Duración (te)': round(t, 2),
                'Inicio Temprano (ES)': round(es, 2),
                'Fin Temprano (EF)': round(ef, 2),
                'Inicio Tardío (LS)': round(ls, 2),
                'Fin Tardío (LF)': round(lf, 2),
                'Holgura': round(holgura, 2),
                'Crítica': 'Sí' if es_critica else 'No'
            })
            
        df_cpm = pd.DataFrame(cpm_results)
        
        # Formatear el string de la ruta crítica
        cp_nodes = nx.dag_longest_path(G, weight='weight')
        path_string_list = []
        for i in range(len(cp_nodes) - 1):
            u, v = cp_nodes[i], cp_nodes[i+1]
            act_id = G.get_edge_data(u, v)['id']
            path_string_list.append(f"{u} -> {act_id} -> {v}")
            
        path_str = " | ".join(path_string_list)

        return cp_nodes, cp_act_ids, project_mean, np.sqrt(total_variance), path_str, df_cpm
    except Exception as e:
        return None, None, None, None, None, None

cp_nodes, cp_act_ids, project_mean, project_sd, path_str, df_cpm = find_critical_path_and_cpm_metrics(results_pert)

# --- 3. VISUALIZACIÓN Y RESULTADOS ---
if cp_nodes and not results_pert.empty:
    st.header("2. Visualización de la Red")
    
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

    # --- TABLA DE REVISIÓN ADELANTE / ATRÁS ---
    if mostrar_pasos and df_cpm is not None:
        st.subheader("📋 Detalles de Revisión (Adelante y Atrás)")
        st.markdown("""
        * **ES (Early Start):** Inicio más cercano | **EF (Early Finish):** Fin más cercano
        * **LS (Late Start):** Inicio más lejano | **LF (Late Finish):** Fin más lejano
        * **Holgura (Slack):** Margen de retraso permitido ($LS - ES$)
        """)
        
        # Resaltar filas críticas en la tabla de Streamlit
        def highlight_critical(val):
            color = '#ffcccc' if val == 'Sí' else ''
            return f'background-color: {color}'
            
        st.dataframe(df_cpm.style.map(highlight_critical, subset=['Crítica']), use_container_width=True)

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
        
elif not results_pert.empty:
    st.error("⚠️ La red no se puede procesar. Revisa que los nodos estén conectados lógicamente y que no haya ciclos infinitos.")
