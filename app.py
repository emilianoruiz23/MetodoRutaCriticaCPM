import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from fpdf import FPDF
import graphviz
import tempfile
import os

st.set_page_config(layout="wide", page_title="CPM Avanzado - Etiquetas en Aristas")

st.title("Calculadora CPM/PERT: Modelo de Actividad en la Arista (AOA)")
st.markdown("En este modelo, los **nodos** son hitos (círculos pequeños) y las **flechas** llevan el etiquetado de costos y tiempos.")

# --- 1. ENTRADA DE DATOS ---
st.header("1. Configuración de Actividades")

default_data = [
    {"Actividad": "A", "Desde": "1", "Hasta": "2", "a": 3.0, "m": 4.17, "b": 6.0, "Costo": 5000.0, "Ficticia": False},
    {"Actividad": "B", "Desde": "2", "Hasta": "3", "a": 2.0, "m": 2.33, "b": 3.0, "Costo": 3000.0, "Ficticia": False},
    {"Actividad": "C", "Desde": "2", "Hasta": "4", "a": 3.0, "m": 3.5, "b": 5.0, "Costo": 4500.0, "Ficticia": False},
    {"Actividad": "F1", "Desde": "3", "Hasta": "4", "a": 0.0, "m": 0.0, "b": 0.0, "Costo": 0.0, "Ficticia": True},
    {"Actividad": "E", "Desde": "4", "Hasta": "5", "a": 5.0, "m": 5.67, "b": 7.0, "Costo": 2000.0, "Ficticia": False},
]

df_input = pd.DataFrame(default_data)
edited_df = st.data_editor(df_input, num_rows="dynamic", use_container_width=True)

# Opciones de visualización simplificadas
st.sidebar.header("Vista de Etiquetas")
vista = st.sidebar.radio("Mostrar en flechas:", ["Solo Adelante (Sumas)", "Solo Atrás (Restas)"])

# --- 2. MOTOR DE CÁLCULO CPM ---
def calculate_aoa_cpm(df):
    if df.empty: return None
    
    df_c = df.copy()
    for col in ['a', 'm', 'b', 'Costo']:
        df_c[col] = pd.to_numeric(df_c[col], errors='coerce').fillna(0)
    
    # PERT
    df_c['te'] = (df_c['a'] + 4 * df_c['m'] + df_c['b']) / 6
    df_c['var'] = ((df_c['b'] - df_c['a']) / 6)**2
    
    G = nx.DiGraph()
    for _, r in df_c.iterrows():
        G.add_edge(str(r['Desde']), str(r['Hasta']), 
                   id=r['Actividad'], t=r['te'], cost=r['Costo'], dummy=r['Ficticia'], var=r['var'])

    if not nx.is_directed_acyclic_graph(G): return "ERROR_CICLO"

    # Forward Pass (Tiempos de Nodos)
    E = {n: 0 for n in G.nodes()}
    for u in nx.topological_sort(G):
        for v in G.successors(u):
            E[v] = max(E[v], E[u] + G[u][v]['t'])
    
    project_duration = max(E.values())
    
    # Backward Pass (Tiempos de Nodos)
    L = {n: project_duration for n in G.nodes()}
    for u in reversed(list(nx.topological_sort(G))):
        for v in G.successors(u):
            L[u] = min(L[u], L[v] - G[u][v]['t'])
            
    # Métricas por Arista
    edges_results = []
    total_var = 0
    cp_edges = []
    total_cost = 0

    for u, v, d in G.edges(data=True):
        es = E[u]
        ef = es + d['t']
        lf = L[v]
        ls = lf - d['t']
        slack = ls - es
        crit = abs(slack) < 0.01
        
        total_cost += d['cost']
        if crit and not d['dummy']:
            total_var += d['var']
            cp_edges.append(d['id'])
            
        edges_results.append({
            'ID': d['id'], 'Desde': u, 'Hasta': v, 't': round(d['t'], 2), 'cost': d['cost'],
            'ES': round(es, 2), 'EF': round(ef, 2), 'LS': round(ls, 2), 'LF': round(lf, 2),
            'Slack': round(slack, 2), 'Crit': crit, 'Dummy': d['dummy']
        })

    return G, pd.DataFrame(edges_results), project_duration, np.sqrt(total_var), cp_edges, total_cost

res = calculate_aoa_cpm(edited_df)

# --- 3. DESPLIEGUE ---
if isinstance(res, tuple):
    G, df_res, duration, sd, cp_list, total_cost = res
    
    # --- RED VISUAL ---
    st.header("2. Diagrama de Red (Etiquetado en Aristas)")
    
    dot = graphviz.Digraph(graph_attr={'rankdir': 'LR', 'nodesep': '0.5', 'ranksep': '1.0'})
    
    # Nodos pequeños
    for n in G.nodes():
        dot.node(n, shape='circle', width='0.3', height='0.3', label=n, fontsize='10', style='filled', fillcolor='white')
        
    # Aristas con etiquetas
    for _, r in df_res.iterrows():
        color = 'red' if r['Crit'] else 'black'
        style = 'dashed' if r['Dummy'] else 'solid'
        pen = '2.5' if r['Crit'] else '1.0'
        
        # Construir etiqueta según vista
        label_txt = f"{r['ID']}\nt={r['t']}\n${r['cost']}"
        if vista == "Solo Adelante (Sumas)":
            label_txt += f"\nES:{r['ES']} EF:{r['EF']}"
        else:
            label_txt += f"\nLS:{r['LS']} LF:{r['LF']}"

        dot.edge(str(r['Desde']), str(r['Hasta']), label=label_txt, color=color, style=style, penwidth=pen, fontsize='9', fontcolor=color)

    st.graphviz_chart(dot)

    # --- TABLA E INFO ---
    st.header("3. Resultados y Reporte")
    col1, col2, col3 = st.columns(3)
    col1.metric("Duración (μ)", f"{duration:.2f}")
    col2.metric("Costo Total", f"${total_cost:,.2f}")
    col3.metric("Desviación (σ)", f"{sd:.4f}")

    st.dataframe(df_res.drop(columns=['Crit', 'Dummy']), use_container_width=True)

    # --- LÓGICA DEL PDF ---
    def generate_pdf(grafico_dot):
        pdf = FPDF()
        pdf.add_page()
        
        # Encabezado
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(190, 10, 'Reporte CPM - Resumen y Red del Proyecto', 0, 1, 'C')
        pdf.ln(5)
        
        # Información Relevante
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(190, 8, 'Métricas Clave del Proyecto:', 0, 1)
        pdf.set_font('Arial', '', 11)
        pdf.cell(190, 7, f"- Duracion Esperada (Total): {duration:.2f} unidades de tiempo", 0, 1)
        pdf.cell(190, 7, f"- Costo Estimado Total: ${total_cost:,.2f}", 0, 1)
        pdf.cell(190, 7, f"- Ruta Critica Identificada: {', '.join(cp_list)}", 0, 1)
        pdf.cell(190, 7, f"- Desviacion Estandar del Proyecto: {sd:.4f}", 0, 1)
        pdf.ln(10)
        
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(190, 8, f'Diagrama de Red ({vista}):', 0, 1)
        pdf.ln(2)
        
        # Renderizar la red en el PDF usando un archivo temporal
        png_data = grafico_dot.pipe(format='png')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(png_data)
            tmp_path = tmp.name
        
        # Insertar imagen (Ajustada al ancho de la hoja)
        pdf.image(tmp_path, x=10, w=190)
        os.unlink(tmp_path)
        
        return pdf.output(dest='S').encode('latin1')

    if st.button("Generar y Descargar Reporte PDF"):
        pdf_bytes = generate_pdf(dot)
        st.download_button(label="Descargar PDF", data=pdf_bytes, file_name="Reporte_Red_CPM.pdf", mime="application/pdf")

elif res == "ERROR_CICLO":
    st.error("Error: Se detectó un ciclo infinito en la red.")
