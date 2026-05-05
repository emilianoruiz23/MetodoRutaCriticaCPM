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
st.markdown("En este modelo, los **nodos** son hitos (círculos pequeños) y las **flechas** llevan el etiquetado. Las actividades críticas se resaltarán automáticamente.")

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
vista = st.sidebar.radio("Mostrar en diagrama:", ["Solo Adelante (Sumas)", "Solo Atrás (Restas y Holguras)"])

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
    
    for u, v, d in G.edges(data=True):
        es = E[u]
        ef = es + d['t']
        lf = L[v]
        ls = lf - d['t']
        slack = ls - es
        crit = abs(slack) < 0.01 # Tolerancia para decimales
        
        if crit and not d['dummy']:
            total_var += d['var']
            cp_edges.append(d['id'])
            
        edges_results.append({
            'Actividad': d['id'], 'Desde': u, 'Hasta': v, 't': round(d['t'], 2), 'Costo': d['cost'],
            'ES': round(es, 2), 'EF': round(ef, 2), 'LS': round(ls, 2), 'LF': round(lf, 2),
            'Holgura': round(slack, 2), 'Crit': crit, 'Dummy': d['dummy']
        })

    return G, pd.DataFrame(edges_results), project_duration, np.sqrt(total_var), cp_edges

res = calculate_aoa_cpm(edited_df)

# --- 3. DESPLIEGUE ---
if isinstance(res, tuple):
    G, df_res, duration, sd, cp_list = res
    
    # --- RED VISUAL ---
    st.header(f"2. Diagrama de Red ({vista})")
    
    dot = graphviz.Digraph(graph_attr={'rankdir': 'LR', 'nodesep': '0.6', 'ranksep': '1.2'})
    
    # Nodos pequeños
    for n in G.nodes():
        dot.node(n, shape='circle', width='0.25', height='0.25', label=n, fontsize='11', style='filled', fillcolor='#f0f2f6')
        
    # Aristas con etiquetas lógicas (Omitiendo 't' si es hacia atrás)
    for _, r in df_res.iterrows():
        color = 'red' if r['Crit'] else 'black'
        style = 'dashed' if r['Dummy'] else 'solid'
        pen = '2.5' if r['Crit'] else '1.0'
        
        # Lógica condicional para el texto de la etiqueta
        if vista == "Solo Adelante (Sumas)":
            label_txt = f"[{r['Actividad']}]\nt = {r['t']}  |  $ {r['Costo']}\nES: {r['ES']}  →  EF: {r['EF']}"
        else:
            # Aquí omitimos la "t" limpiando la vista hacia atrás
            label_txt = f"[{r['Actividad']}]\n$ {r['Costo']}\nLS: {r['LS']}  →  LF: {r['LF']}\nHolgura = {r['Holgura']}"

        dot.edge(str(r['Desde']), str(r['Hasta']), label=label_txt, color=color, style=style, penwidth=pen, fontsize='10', fontcolor=color)

    st.graphviz_chart(dot)

    # --- TABLA E INFO ---
    st.header("3. Resultados y Reporte")
    col1, col2 = st.columns(2)
    col1.metric("Duración Esperada (μ)", f"{duration:.2f} unidades")
    col2.metric("Desviación (σ)", f"{sd:.4f}")

    # Seleccionar columnas a mostrar
    columnas_mostrar = ['Actividad', 'Desde', 'Hasta', 't', 'ES', 'EF', 'LS', 'LF', 'Holgura']
    df_mostrar = df_res[columnas_mostrar]

    # Resaltar en rojo las filas críticas
    def highlight_critical_rows(row):
        color = '#ffe6e6' if abs(row['Holgura']) < 0.01 else ''
        return [f'background-color: {color}'] * len(row)

    st.dataframe(df_mostrar.style.apply(highlight_critical_rows, axis=1), use_container_width=True)

    # --- LÓGICA DEL PDF ROBUSTO ---
    def generate_pdf(grafico_dot):
        pdf = FPDF()
        pdf.add_page()
        
        # Encabezado
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(190, 10, 'Reporte Analitico CPM (Modelo AOA)', 0, 1, 'C')
        pdf.ln(5)
        
        # Información Relevante
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(190, 8, '1. Resumen Ejecutivo:', 0, 1)
        
        pdf.set_font('Arial', '', 11)
        pdf.cell(190, 6, f"   - Duracion Estimada del Proyecto: {duration:.2f} unidades", 0, 1)
        pdf.cell(190, 6, f"   - Desviacion Estandar: {sd:.4f}", 0, 1)
        pdf.cell(190, 6, f"   - Ruta Critica (Actividades sin holgura): {', '.join(cp_list)}", 0, 1)
        pdf.ln(5)

        pdf.set_font('Arial', 'B', 12)
        pdf.cell(190, 8, '2. Enfoque del Diagrama:', 0, 1)
        pdf.set_font('Arial', '', 11)
        
        if vista == "Solo Adelante (Sumas)":
            pdf.multi_cell(190, 6, "   El diagrama adjunto muestra el pase hacia adelante (Forward Pass). "
                                   "Se detallan el Inicio Temprano (ES) y Fin Temprano (EF) de cada actividad. "
                                   "Los costos y duraciones esperadas (t) se incluyen en los arcos.")
        else:
            pdf.multi_cell(190, 6, "   El diagrama adjunto muestra el pase hacia atras (Backward Pass). "
                                   "Se detallan el Inicio Tardio (LS), el Fin Tardio (LF) y se destaca "
                                   "explicita la 'Holgura' de cada actividad, omitiendo los tiempos para mayor claridad. "
                                   "Las rutas en rojo representan holgura cero.")
            
        pdf.ln(10)
        
        # Renderizar la red en el PDF (capturando el dot exacto con/sin la 't')
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(190, 8, '3. Diagrama de Red Estructurado:', 0, 1)
        
        png_data = grafico_dot.pipe(format='png')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(png_data)
            tmp_path = tmp.name
        
        # Insertar imagen
        pdf.image(tmp_path, x=10, w=190)
        os.unlink(tmp_path)
        
        return pdf.output(dest='S').encode('latin1')

    if st.button("Generar y Descargar Reporte PDF"):
        pdf_bytes = generate_pdf(dot)
        st.download_button(label="Descargar PDF", data=pdf_bytes, file_name="Reporte_Detallado_CPM.pdf", mime="application/pdf")

elif res == "ERROR_CICLO":
    st.error("Error: Se detectó un ciclo infinito en la red. Revisa los nodos 'Desde' y 'Hasta'.")
