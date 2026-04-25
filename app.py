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

st.title("Calculadora PERT/CPM (Por Predecesores) con PDF")
st.markdown("Ingresa tus actividades, descripciones y dependencias. Separa múltiples predecesores con comas (ej. `A, B`).")

# --- 1. ENTRADA DE DATOS (ESTILO AON) ---
st.header("1. Ingreso de Actividades")

default_data = [
    {"Actividad": "A", "Descripción": "Cimientos", "Predecesores": "", "b": 6.0, "a": 3.0, "m": 4.0},
    {"Actividad": "B", "Descripción": "Plomería/electricidad", "Predecesores": "A", "b": 5.0, "a": 1.0, "m": 2.0},
    {"Actividad": "C", "Descripción": "Techo", "Predecesores": "A", "b": 7.0, "a": 2.0, "m": 3.0},
    {"Actividad": "D", "Descripción": "Pintura exterior", "Predecesores": "A", "b": 3.0, "a": 0.5, "m": 1.0},
    {"Actividad": "E", "Descripción": "Pintura interior", "Predecesores": "B, C", "b": 10.0, "a": 4.0, "m": 5.0},
]

df_inicial = pd.DataFrame(default_data)

st.info("💡 Consejo: Para agregar más actividades, haz clic en el botón '+' debajo de la tabla. Usa guiones '-' o déjalo en blanco si no hay predecesor.")

edited_df = st.data_editor(df_inicial, num_rows="dynamic", use_container_width=True)

# --- MENÚ DE OPCIONES DE REVISIÓN ---
st.write("---")
st.subheader("Opciones de Visualización de la Tabla de CPM")
tipo_revision = st.radio(
    "Elige qué datos de revisión quieres analizar:",
    options=["Ambas (Completo)", "Solo Revisión hacia Adelante (ES, EF)", "Solo Revisión hacia Atrás y Holgura (LS, LF)"],
    horizontal=True
)

# --- 2. CÁLCULOS PERT Y CPM (AON) ---
def process_cpm_network(df):
    if df.empty: return None, None, None, None, None, None
    
    df_calc = df.copy()
    for col in ['b', 'a', 'm']:
        df_calc[col] = pd.to_numeric(df_calc[col], errors='coerce').fillna(0)
        
    df_calc['t'] = (df_calc['a'] + 4 * df_calc['m'] + df_calc['b']) / 6
    df_calc['variance'] = ((df_calc['b'] - df_calc['a']) / 6)**2
    df_calc['sigma'] = np.sqrt(df_calc['variance'])
    
    G = nx.DiGraph()
    for index, row in df_calc.iterrows():
        act = str(row['Actividad']).strip()
        if act:
            G.add_node(act, weight=row['t'], desc=row['Descripción'])
            
    for index, row in df_calc.iterrows():
        act = str(row['Actividad']).strip()
        preds = str(row['Predecesores']).replace('-', '').split(',')
        if act:
            for p in preds:
                p = p.strip()
                if p and p in G.nodes:
                    G.add_edge(p, act)

    try:
        if not nx.is_directed_acyclic_graph(G):
            return "ERROR_CICLO", None, None, None, None, None

        E_S = {}
        E_F = {}
        topo_order = list(nx.topological_sort(G))
        
        for node in topo_order:
            preds = list(G.predecessors(node))
            if not preds:
                E_S[node] = 0
            else:
                E_S[node] = max([E_F[p] for p in preds])
            E_F[node] = E_S[node] + G.nodes[node]['weight']
            
        project_duration = max(E_F.values()) if E_F else 0

        L_S = {}
        L_F = {}
        for node in reversed(topo_order):
            succs = list(G.successors(node))
            if not succs:
                L_F[node] = project_duration
            else:
                L_F[node] = min([L_S[s] for s in succs])
            L_S[node] = L_F[node] - G.nodes[node]['weight']

        cpm_data = []
        cp_nodes = []
        total_variance = 0
        
        for node in topo_order:
            slack = L_S[node] - E_S[node]
            es_critica = abs(slack) < 1e-6
            
            if es_critica:
                cp_nodes.append(node)
                activity_var = df_calc[df_calc['Actividad'] == node]['variance'].values[0]
                total_variance += activity_var
                
            cpm_data.append({
                'Actividad': node,
                'Duración (t)': round(G.nodes[node]['weight'], 2),
                'ES': round(E_S[node], 2),
                'EF': round(E_F[node], 2),
                'LS': round(L_S[node], 2),
                'LF': round(L_F[node], 2),
                'Holgura': round(slack, 2),
                'Crítica': 'Sí' if es_critica else 'No'
            })
            
        df_cpm = pd.DataFrame(cpm_data)
        path_str = " -> ".join(cp_nodes)
        
        return G, df_calc, cp_nodes, project_duration, np.sqrt(total_variance), path_str, df_cpm

    except Exception as e:
        return "ERROR", None, None, None, None, None, None

resultado = process_cpm_network(edited_df)

# --- 3. VISUALIZACIÓN ---
if isinstance(resultado, tuple) and len(resultado) == 7:
    G, df_calc, cp_nodes, project_mean, project_sd, path_str, df_cpm = resultado

    if G == "ERROR_CICLO":
        st.error("⚠️ Ciclo detectado: Una actividad no puede depender de sí misma o crear un bucle infinito.")
    elif G is not None:
        
        st.write("---")
        st.subheader("Tabla de Tiempos Esperados (PERT)")
        df_display = df_calc[['Actividad', 'Descripción', 'Predecesores', 'b', 'a', 'm', 't', 'sigma']].copy()
        df_display[['t', 'sigma']] = df_display[['t', 'sigma']].round(2)
        st.dataframe(df_display, use_container_width=True)

        st.header("2. Diagrama de Red (AON)")
        dot = graphviz.Digraph(graph_attr={'rankdir': 'LR'})
        for node in G.nodes():
            es_critica = node in cp_nodes
            color = 'red' if es_critica else 'black'
            fill = '#ffcccc' if es_critica else 'lightcyan'
            label = f"{node}\n(t={G.nodes[node]['weight']:.1f})"
            dot.node(node, label=label, shape='box', style='filled', fillcolor=fill, color=color, penwidth='2.0' if es_critica else '1.0')
            
        for u, v in G.edges():
            es_critica = u in cp_nodes and v in cp_nodes
            color = 'red' if es_critica else 'gray'
            dot.edge(u, v, color=color, penwidth='2.0' if es_critica else '1.0')

        st.graphviz_chart(dot)

        # --- TABLA DE REVISIÓN FILTRADA POR EL USUARIO ---
        if df_cpm is not None:
            st.subheader(f"📋 Detalles: {tipo_revision}")
            
            df_mostrar = df_cpm.copy()
            
            if tipo_revision == "Solo Revisión hacia Adelante (ES, EF)":
                df_mostrar = df_mostrar[['Actividad', 'Duración (t)', 'ES', 'EF', 'Crítica']]
            elif tipo_revision == "Solo Revisión hacia Atrás y Holgura (LS, LF)":
                df_mostrar = df_mostrar[['Actividad', 'Duración (t)', 'LS', 'LF', 'Holgura', 'Crítica']]
                
            def highlight_critical(val):
                color = '#ffcccc' if val == 'Sí' else ''
                return f'background-color: {color}'
                
            st.dataframe(df_mostrar.style.map(highlight_critical, subset=['Crítica']), use_container_width=True)

        st.header("3. Análisis y Reporte PDF")
        col1, col2 = st.columns(2)
        col1.metric("Duración Total Esperada (μ)", f"{project_mean:.2f} semanas")
        col2.metric("Desviación Estándar (σ)", f"{project_sd:.4f}")
        st.markdown(f"**Ruta Crítica:** `{path_str}`")

        target_time = st.number_input("Tiempo Objetivo para finalizar:", value=float(np.ceil(project_mean)), step=0.5)
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
            pdf.cell(190, 10, 'Reporte PERT/CPM (Metodo Predecesores)', 0, 1, 'C')
            pdf.ln(5)
            
            pdf.set_font('Arial', '', 12)
            pdf.cell(190, 8, f"Duracion Esperada: {project_mean:.2f} semanas", 0, 1)
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
            st.download_button(label="Descargar PDF", data=pdf_bytes, file_name="Reporte_AON.pdf", mime="application/pdf")
