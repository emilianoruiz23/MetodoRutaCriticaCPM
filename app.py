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
    total_cost = 0
    
    for _, r in df_c.iterrows():
        total_cost += r['Costo'] if not r['Ficticia'] else 0
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

    return G, pd.DataFrame(edges_results), project_duration, np.sqrt(total_var), cp_list, total_var, total_cost

res = calculate_integral_cpm(edited_df)

# --- 3. DESPLIEGUE ---
if isinstance(res, tuple):
    G, df_res, duration, sd, cp_list, total_var, total_cost = res
    
    st.header(f"2. Diagrama de Red Integral")
    
    dot = graphviz.Digraph(graph_attr={'rankdir': 'LR', 'nodesep': '0.7', 'ranksep': '1.3'})
    
    for n in G.nodes():
        dot.node(n, shape='circle', width='0.25', height='0.25', label=n, fontsize='11', style='filled', fillcolor='#f0f2f6')
        
    for _, r in df_res.iterrows():
        color = 'red' if r['Crit'] else 'black'
        style = 'dashed' if r['Dummy'] else 'solid'
        pen = '2.5' if r['Crit'] else '1.0'
        
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
    col1, col2, col3 = st.columns(3)
    col1.metric("Duración Total (μ)", f"{duration:.2f} unidades")
    col2.metric("Desviación Estándar (σ)", f"{sd:.4f}")
    col3.metric("Costo Estimado", f"${total_cost:,.2f}")

    columnas = ['Actividad', 'Desde', 'Hasta', 't', 'Costo', 'ES', 'EF', 'LS', 'LF', 'Holgura']
    st.dataframe(df_res[columnas].style.apply(lambda r: ['background-color: #ffe6e6']*len(r) if abs(r['Holgura']) < 0.01 else ['']*len(r), axis=1), use_container_width=True)

    # --- LÓGICA DEL PDF ROBUSTECIDO ---
    def generate_pdf(grafico_dot):
        pdf = FPDF()
        pdf.add_page()
        
        # 1. Encabezado
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(190, 10, 'Dossier Analitico CPM/PERT - Reporte de Proyecto', 0, 1, 'C')
        pdf.ln(5)
        
        # 2. Resumen Ejecutivo
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(190, 8, '1. Resumen Ejecutivo del Proyecto:', 0, 1)
        pdf.set_font('Arial', '', 11)
        pdf.cell(190, 6, f"   - Duracion Final Calculada: {duration:.2f} unidades de tiempo", 0, 1)
        pdf.cell(190, 6, f"   - Inversion / Costo Total: ${total_cost:,.2f}", 0, 1)
        pdf.cell(190, 6, f"   - Riesgo Estimado (sigma): {sd:.4f}", 0, 1)
        pdf.cell(190, 6, f"   - Ruta Critica (Actividades sin margen de retraso): {', '.join(cp_list)}", 0, 1)
        pdf.ln(5)

        # 3. Glosario de Términos
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(190, 8, '2. Glosario de Terminos (Nomenclatura):', 0, 1)
        pdf.set_font('Arial', '', 10)
        pdf.cell(190, 5, "   * t (Tiempo Esperado): Duracion calculada por PERT (a+4m+b)/6.", 0, 1)
        pdf.cell(190, 5, "   * ES (Early Start): Inicio mas temprano posible de la actividad.", 0, 1)
        pdf.cell(190, 5, "   * EF (Early Finish): Fin mas temprano posible (ES + t).", 0, 1)
        pdf.cell(190, 5, "   * LS (Late Start): Inicio mas tardio permitido sin retrasar el proyecto (LF - t).", 0, 1)
        pdf.cell(190, 5, "   * LF (Late Finish): Fin mas tardio permitido.", 0, 1)
        pdf.cell(190, 5, "   * Holgura: Margen de retraso permitido (LS - ES). Si es 0, es Ruta Critica.", 0, 1)
        pdf.ln(5)

        # 4. Memoria de Cálculo (Ruta Crítica)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(190, 8, '3. Memoria de Calculo (Comprobacion Matematica):', 0, 1)
        pdf.set_font('Arial', '', 10)
        
        crit_df = df_res[df_res['Crit'] == True]
        suma_t = " + ".join([f"{r['t']}" for _, r in crit_df.iterrows()])
        suma_v = " + ".join([f"{r['Var']:.2f}" for _, r in crit_df.iterrows()])
        
        pdf.cell(190, 6, f"   * Calculo Duracion Total: {suma_t} = {duration:.2f}", 0, 1)
        pdf.cell(190, 6, f"   * Calculo Riesgo: Raiz_Cuadrada({suma_v}) = {sd:.4f}", 0, 1)
        pdf.ln(2)

        pdf.set_font('Arial', 'I', 10)
        pdf.cell(190, 6, "   * Verificacion de Pases en Ruta Critica (Ida y Vuelta):", 0, 1)
        pdf.set_font('Arial', '', 9)
        for _, r in crit_df.iterrows():
            if r['Dummy']: continue
            linea = f"     - [{r['Actividad']}]: IDA -> ES({r['ES']}) + t({r['t']}) = EF({r['EF']})  |  VUELTA -> LF({r['LF']}) - t({r['t']}) = LS({r['LS']})"
            pdf.cell(190, 5, linea, 0, 1)
        
        pdf.ln(8)

        # 5. Tabla Completa de Resultados
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(190, 8, '4. Tabla Maestra de Resultados por Actividad:', 0, 1)
        
        # Configuración de la tabla
        pdf.set_font('Arial', 'B', 9)
        pdf.set_fill_color(230, 230, 230)
        # Anchos: Act(20), De(15), A(15), t(15), Costo(25), ES(15), EF(15), LS(15), LF(15), Holgura(25) = Total 175
        col_widths = [20, 15, 15, 15, 25, 15, 15, 15, 15, 25]
        headers = ['Actividad', 'Desde', 'Hasta', 't', 'Costo ($)', 'ES', 'EF', 'LS', 'LF', 'Holgura']
        
        # Imprimir Cabeceras
        for col_name, w in zip(headers, col_widths):
            pdf.cell(w, 7, col_name, 1, 0, 'C', 1)
        pdf.ln()
        
        # Imprimir Filas
        pdf.set_font('Arial', '', 9)
        for _, r in df_res.iterrows():
            if r['Dummy']: continue # Omitir ficticias en la tabla impresa para mayor limpieza
            
            # Si es ruta crítica, ponemos texto en rojo
            if r['Crit']:
                pdf.set_text_color(200, 0, 0)
            else:
                pdf.set_text_color(0, 0, 0)
                
            pdf.cell(col_widths[0], 6, str(r['Actividad']), 1, 0, 'C')
            pdf.cell(col_widths[1], 6, str(r['Desde']), 1, 0, 'C')
            pdf.cell(col_widths[2], 6, str(r['Hasta']), 1, 0, 'C')
            pdf.cell(col_widths[3], 6, f"{r['t']:.2f}", 1, 0, 'C')
            pdf.cell(col_widths[4], 6, f"${r['Costo']:,.2f}", 1, 0, 'C')
            pdf.cell(col_widths[5], 6, f"{r['ES']:.2f}", 1, 0, 'C')
            pdf.cell(col_widths[6], 6, f"{r['EF']:.2f}", 1, 0, 'C')
            pdf.cell(col_widths[7], 6, f"{r['LS']:.2f}", 1, 0, 'C')
            pdf.cell(col_widths[8], 6, f"{r['LF']:.2f}", 1, 0, 'C')
            pdf.cell(col_widths[9], 6, f"{r['Holgura']:.2f}", 1, 0, 'C')
            pdf.ln()
            
        # Resetear color a negro
        pdf.set_text_color(0, 0, 0)
        
        # 6. Diagrama de Red (EN UNA NUEVA PÁGINA PARA MAYOR TAMAÑO)
        pdf.add_page() # Salto de página
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(190, 10, f'5. Diagrama de Red del Proyecto ({vista}):', 0, 1, 'C')
        pdf.ln(5)
        
        png_data = grafico_dot.pipe(format='png')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(png_data)
            tmp_path = tmp.name
            
        # Como está en su propia página, podemos hacerlo mucho más grande (w=190 abarca todo el ancho)
        pdf.image(tmp_path, x=10, w=190)
        os.unlink(tmp_path)
        
        return pdf.output(dest='S').encode('latin1')

    if st.button("Generar y Descargar Reporte PDF Integral"):
        pdf_bytes = generate_pdf(dot)
        st.download_button(label="Descargar PDF", data=pdf_bytes, file_name="Dossier_CPM_Completo.pdf", mime="application/pdf")

elif res == "ERROR_CICLO":
    st.error("Error: Se detectó un ciclo infinito en la red.")
