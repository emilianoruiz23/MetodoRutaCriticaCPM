import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from fpdf import FPDF
import graphviz
import tempfile
import os

st.set_page_config(layout="wide", page_title="CPM Maestro - Vista Integral")

st.title("Calculadora CPM/PERT Avanzada: Memoria de Cálculo Integral")
st.markdown("Agrega tus actividades. El sistema calculará automáticamente los pases adelante, atrás y holguras.")

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

# Opciones de visualización
st.sidebar.header("Control de Visualización")
vista = st.sidebar.radio("Modo de Análisis:", 
                         ["Vista Mixta (Completa)", "Solo Adelante (Sumas)", "Solo Atrás (Restas y Holguras)"])

# --- 2. MOTOR DE CÁLCULO CPM ---
def calculate_integral_cpm(df):
    if df.empty: return None
    
    df_c = df.copy()
    for col in ['a', 'm', 'b', 'Costo']:
        df_c[col] = pd.to_numeric(df_c[col], errors='coerce').fillna(0)
    
    df_c['te'] = (df_c['a'] + 4 * df_c['m'] + df_c['b']) / 6
    df_c['var'] = ((df_c['b'] - df_c['a']) / 6)**2
    
    G = nx.DiGraph()
    for _, r in df_c.iterrows():
        G.add_edge(str(r['Desde']), str(r['Hasta']), 
                   id=r['Actividad'], t=r['te'], cost=r['Costo'], dummy=r['Ficticia'], var=r['var'])

    if not nx.is_directed_acyclic_graph(G): return "ERROR_CICLO"

    # Forward Pass
    E = {n: 0 for n in G.nodes()}
    for u in nx.topological_sort(G):
        for v in G.successors(u):
            E[v] = max(E[v], E[u] + G[u][v]['t'])
    
    project_duration = max(E.values())
    
    # Backward Pass
    L = {n: project_duration for n in G.nodes()}
    for u in reversed(list(nx.topological_sort(G))):
        for v in G.successors(u):
            L[u] = min(L[u], L[v] - G[u][v]['t'])
            
    edges_results = []
    total_var = 0
    cp_list = []
    
    for u, v, d in G.edges(data=True):
        es, ef = E[u], E[u] + d['t']
        lf, ls = L[v], L[v] - d['t']
        slack = ls - es
        crit = abs(slack) < 0.01 
        
        if crit and not d['dummy']:
            total_var += d['var']
            cp_list.append(d['id'])
            
        edges_results.append({
            'Actividad': d['id'], 'Desde': u, 'Hasta': v, 't': round(d['t'], 2), 'Costo': d['cost'],
            'ES': round(es, 2), 'EF': round(ef, 2), 'LS': round(ls, 2), 'LF': round(lf, 2),
            'Holgura': round(slack, 2), 'Crit': crit, 'Dummy': d['dummy'], 'Var': d['var']
        })

    return G, pd.DataFrame(edges_results), project_duration, np.sqrt(total_var), cp_list, total_var

res = calculate_integral_cpm(edited_df)

# --- 3. DESPLIEGUE ---
if isinstance(res, tuple):
    G, df_res, duration, sd, cp_list, total_var = res
    
    st.header(f"2. Diagrama de Red Integral")
    
    dot = graphviz.Digraph(graph_attr={'rankdir': 'LR', 'nodesep': '0.7', 'ranksep': '1.3'})
    
    for n in G.nodes():
        dot.node(n, shape='circle', width='0.25', height='0.25', label=n, fontsize='11', style='filled', fillcolor='#f0f2f6')
        
    for _, r in df_res.iterrows():
        color = 'red' if r['Crit'] else 'black'
        style = 'dashed' if r['Dummy'] else 'solid'
        pen = '2.5' if r['Crit'] else '1.0'
        
        # ETIQUETADO MIXTO / COMPLETO
        header = f"[{r['Actividad']}]  $ {r['Costo']}"
        fwd = f"Ida (ES→EF): {r['ES']} → {r['EF']}"
        bwd = f"Vuelta (LS→LF): {r['LS']} → {r['LF']}"
        hol = f"Holgura: {r['Holgura']}"
        
        if vista == "Vista Mixta (Completa)":
            label_txt = f"{header}\n{fwd}\n{bwd}\n{hol}"
        elif vista == "Solo Adelante (Sumas)":
            label_txt = f"{header}\nt = {r['t']}\n{fwd}"
        else:
            label_txt = f"{header}\n{bwd}\n{hol}"

        dot.edge(str(r['Desde']), str(r['Hasta']), label=label_txt, color=color, style=style, penwidth=pen, fontsize='9', fontcolor=color)

    st.graphviz_chart(dot)

    st.header("3. Memoria de Resultados Técnicos")
    col1, col2 = st.columns(2)
    col1.metric("Duración Total (μ)", f"{duration:.2f} unidades")
    col2.metric("Desviación Estándar (σ)", f"{sd:.4f}")

    # Tabla con Resaltado Rojo
    columnas = ['Actividad', 'Desde', 'Hasta', 't', 'ES', 'EF', 'LS', 'LF', 'Holgura']
    st.dataframe(df_res[columnas].style.apply(lambda r: ['background-color: #ffe6e6']*len(r) if abs(r['Holgura']) < 0.01 else ['']*len(r), axis=1), use_container_width=True)

    # --- LÓGICA DEL PDF MIXTO CON FIX DE IMAGEN ---
    def generate_pdf(grafico_dot):
        pdf = FPDF()
        pdf.add_page()
        
        # 1. Encabezado
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(190, 10, 'Reporte Integral CPM/PERT - Memoria Tecnica', 0, 1, 'C')
        pdf.ln(5)
        
        # 2. Resumen
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(190, 8, '1. Resumen de Proyecto:', 0, 1)
        pdf.set_font('Arial', '', 11)
        pdf.cell(190, 6, f"   - Duracion Final Calculada: {duration:.2f} unidades", 0, 1)
        pdf.cell(190, 6, f"   - Riesgo Estimado (sigma): {sd:.4f}", 0, 1)
        pdf.cell(190, 6, f"   - Actividades Criticas: {', '.join(cp_list)}", 0, 1)
        pdf.ln(5)

        # 3. Matemática Particular (DATOS REALES)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(190, 8, '2. Memoria de Calculo (Datos del Proyecto):', 0, 1)
        pdf.set_font('Arial', '', 10)
        
        crit_df = df_res[df_res['Crit'] == True]
        suma_t = " + ".join([f"{r['t']}" for _, r in crit_df.iterrows()])
        suma_v = " + ".join([f"{r['Var']:.2f}" for _, r in crit_df.iterrows()])
        
        pdf.cell(190, 6, f"   * Calculo Duracion: {suma_t} = {duration:.2f}", 0, 1)
        pdf.cell(190, 6, f"   * Calculo Riesgo: Raiz({suma_v}) = {sd:.4f}", 0, 1)
        pdf.ln(3)

        pdf.set_font('Arial', 'I', 10)
        pdf.cell(190, 6, "   * Verificacion de Pases en Ruta Critica:", 0, 1)
        pdf.set_font('Arial', '', 9)
        for _, r in crit_df.iterrows():
            if r['Dummy']: continue
            linea = f"     - [{r['Actividad']}]: IDA {r['ES']}+{r['t']}={r['EF']} | VUELTA {r['LF']}-{r['t']}={r['LS']} | Holgura: 0"
            pdf.cell(190, 5, linea, 0, 1)
        
        pdf.ln(10)

        # 4. Diagrama de Red
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(190, 8, f'3. Diagrama de Red ({vista}):', 0, 1)
        
        png_data = grafico_dot.pipe(format='png')
        
        # Bloque corregido para evitar el error 'Not a PNG file'
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(png_data)
            tmp_path = tmp.name
            
        # Fuera del 'with', la imagen se inserta y luego se borra
        pdf.image(tmp_path, x=10, w=190)
        os.unlink(tmp_path)
        
        return pdf.output(dest='S').encode('latin1')

    if st.button("Generar y Descargar Reporte PDF Integral"):
        pdf_bytes = generate_pdf(dot)
        st.download_button(label="Descargar PDF", data=pdf_bytes, file_name="Reporte_CPM_Integral.pdf", mime="application/pdf")

elif res == "ERROR_CICLO":
    st.error("Error: Se detectó un ciclo infinito en la red.")
