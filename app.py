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

st.set_page_config(layout="wide", page_title="Generador CPM/PERT Avanzado")

st.title("Calculadora PERT/CPM: Nodos Circulares, Costos y Matemáticas")
st.markdown("Ingresa tus actividades. Marca la casilla **Ficticia** para nodos que solo representan dependencias lógicas (t=0, Costo=0).")

# --- 1. ENTRADA DE DATOS ---
st.header("1. Ingreso de Actividades y Costos")

default_data = [
    {"Actividad": "A", "Descripción": "Cimientos", "Predecesores": "", "b": 6.0, "a": 3.0, "m": 4.0, "Costo": 5000.0, "Ficticia": False},
    {"Actividad": "B", "Descripción": "Plomería", "Predecesores": "A", "b": 5.0, "a": 1.0, "m": 2.0, "Costo": 3000.0, "Ficticia": False},
    {"Actividad": "C", "Descripción": "Techo", "Predecesores": "A", "b": 7.0, "a": 2.0, "m": 3.0, "Costo": 4500.0, "Ficticia": False},
    {"Actividad": "F1", "Descripción": "Dependencia Dummy", "Predecesores": "B", "b": 0.0, "a": 0.0, "m": 0.0, "Costo": 0.0, "Ficticia": True},
    {"Actividad": "E", "Descripción": "Pintura", "Predecesores": "F1, C", "b": 10.0, "a": 4.0, "m": 5.0, "Costo": 2000.0, "Ficticia": False},
]

df_inicial = pd.DataFrame(default_data)
edited_df = st.data_editor(df_inicial, num_rows="dynamic", use_container_width=True)

# --- MENÚ LATERAL ---
st.sidebar.header("Opciones de Visualización")
tipo_grafica = st.sidebar.radio(
    "¿Qué revisión mostrar en los arcos/nodos?",
    options=["Completo (Todos los datos)", "Revisión Adelante (Sumas)", "Revisión Atrás (Restas)"]
)

# --- 2. CÁLCULOS PERT Y CPM ---
def process_cpm_network(df):
    if df.empty: return None, None, None, None, None, None, None, None, None
    
    df_calc = df.copy()
    
    # Asegurar tipos numéricos y forzar a cero si es ficticia
    for col in ['b', 'a', 'm', 'Costo']:
        df_calc[col] = pd.to_numeric(df_calc[col], errors='coerce').fillna(0)
    
    df_calc.loc[df_calc['Ficticia'] == True, ['b', 'a', 'm', 'Costo']] = 0.0
        
    df_calc['t'] = (df_calc['a'] + 4 * df_calc['m'] + df_calc['b']) / 6
    df_calc['variance'] = ((df_calc['b'] - df_calc['a']) / 6)**2
    df_calc['sigma'] = np.sqrt(df_calc['variance'])
    
    G = nx.DiGraph()
    costo_total = 0.0
    
    # Crear Nodos
    for index, row in df_calc.iterrows():
        act = str(row['Actividad']).strip()
        if act:
            costo_act = row['Costo']
            costo_total += costo_act
            G.add_node(act, weight=row['t'], cost=costo_act, is_dummy=row['Ficticia'])
            
    # Crear Arcos
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
            return "ERROR_CICLO", None, None, None, None, None, None, None, None

        E_S = {}
        E_F = {}
        log_adelante = []
        topo_order = list(nx.topological_sort(G))
        
        # 1. REVISIÓN ADELANTE (SUMAS)
        for node in topo_order:
            t = G.nodes[node]['weight']
            preds = list(G.predecessors(node))
            if not preds:
                E_S[node] = 0
                log_adelante.append(f"Act {node}: Sin predecesores -> ES = 0. Se suma t({t:.2f}) -> EF = {t:.2f}")
            else:
                max_ef = max([E_F[p] for p in preds])
                E_S[node] = max_ef
                pred_str = ", ".join([f"{p}(EF={E_F[p]:.2f})" for p in preds])
                log_adelante.append(f"Act {node}: Predecesores [{pred_str}]. Se toma el MAYOR -> ES = {max_ef:.2f}. Se suma t({t:.2f}) -> EF = {max_ef + t:.2f}")
            E_F[node] = E_S[node] + t
            
        project_duration = max(E_F.values()) if E_F else 0

        # 2. REVISIÓN ATRÁS (RESTAS)
        L_S = {}
        L_F = {}
        log_atras = []
        for node in reversed(topo_order):
            t = G.nodes[node]['weight']
            succs = list(G.successors(node))
            if not succs:
                L_F[node] = project_duration
                log_atras.append(f"Act {node}: Nodo final -> LF = {project_duration:.2f}. Se resta t({t:.2f}) -> LS = {project_duration - t:.2f}")
            else:
                min_ls = min([L_S[s] for s in succs])
                L_F[node] = min_ls
                succ_str = ", ".join([f"{s}(LS={L_S[s]:.2f})" for s in succs])
                log_atras.append(f"Act {node}: Sucesores [{succ_str}]. Se toma el MENOR -> LF = {min_ls:.2f}. Se resta t({t:.2f}) -> LS = {min_ls - t:.2f}")
            L_S[node] = L_F[node] - t

        # 3. CONSOLIDAR RESULTADOS
        cpm_data = []
        cp_nodes = []
        total_variance = 0
        
        for node in topo_order:
            slack = L_S[node] - E_S[node]
            es_critica = abs(slack) < 1e-6
            is_dummy = G.nodes[node]['is_dummy']
            
            if es_critica and not is_dummy:
                cp_nodes.append(node)
                activity_var = df_calc[df_calc['Actividad'] == node]['variance'].values[0]
                total_variance += activity_var
                
            cpm_data.append({
                'Actividad': node,
                'Duración (t)': round(G.nodes[node]['weight'], 2),
                'Costo ($)': round(G.nodes[node]['cost'], 2),
                'ES': round(E_S[node], 2),
                'EF': round(E_F[node], 2),
                'LS': round(L_S[node], 2),
                'LF': round(L_F[node], 2),
                'Holgura': round(slack, 2),
                'Crítica': 'Sí' if es_critica else 'No',
                'Ficticia': 'Sí' if is_dummy else 'No'
            })
            
        df_cpm = pd.DataFrame(cpm_data)
        path_str = " -> ".join(cp_nodes)
        
        return G, df_calc, cp_nodes, project_duration, np.sqrt(total_variance), path_str, df_cpm, log_adelante, log_atras, costo_total

    except Exception as e:
        return "ERROR", None, None, None, None, None, None, None, None, None

# Ejecutar lógica
resultado = process_cpm_network(edited_df)

# --- 3. VISUALIZACIÓN ---
if isinstance(resultado, tuple) and len(resultado) == 10:
    G, df_calc, cp_nodes, project_mean, project_sd, path_str, df_cpm, log_adelante, log_atras, costo_total = resultado

    if G == "ERROR_CICLO":
        st.error("⚠️ Ciclo detectado: Una red no puede tener bucles (ej. A depende de B, y B depende de A).")
    elif G is not None:
        
        # --- MATEMÁTICAS EXPLÍCITAS ---
        st.header("2. Desarrollo Matemático (Paso a Paso)")
        col_ad, col_at = st.columns(2)
        with col_ad:
            st.subheader("Pase Adelante (Suma: ES + t = EF)")
            for linea in log_adelante:
                st.code(linea, language="text")
        with col_at:
            st.subheader("Pase Atrás (Resta: LF - t = LS)")
            for linea in log_atras:
                st.code(linea, language="text")

        # --- DIAGRAMA DE RED CIRCULAR ---
        st.header("3. Diagrama de Red (Nodos Circulares)")
        
        dot = graphviz.Digraph(graph_attr={'rankdir': 'LR', 'splines': 'ortho'})
        
        for node in G.nodes():
            row = df_cpm[df_cpm['Actividad'] == node].iloc[0]
            es_critica = row['Crítica'] == 'Sí'
            is_dummy = row['Ficticia'] == 'Sí'
            
            color = 'red' if es_critica else 'black'
            fill = '#ffcccc' if es_critica else 'white'
            style = 'dashed,filled' if is_dummy else 'filled'
            
            # Construcción de la etiqueta según el menú lateral
            label_parts = [f"Act: {node}", f"t={row['Duración (t)']}", f"${row['Costo ($)']}"]
            
            if tipo_grafica in ["Completo (Todos los datos)", "Revisión Adelante (Sumas)"]:
                label_parts.append(f"ES:{row['ES']} | EF:{row['EF']}")
            if tipo_grafica in ["Completo (Todos los datos)", "Revisión Atrás (Restas)"]:
                label_parts.append(f"LS:{row['LS']} | LF:{row['LF']}")
                
            label = "\n".join(label_parts)
            
            # Nodos explícitamente CIRCULARES
            dot.node(node, label=label, shape='circle', style=style, fillcolor=fill, color=color, penwidth='2.0' if es_critica else '1.0')
            
        for u, v in G.edges():
            is_u_crit = df_cpm[df_cpm['Actividad'] == u].iloc[0]['Crítica'] == 'Sí'
            is_v_crit = df_cpm[df_cpm['Actividad'] == v].iloc[0]['Crítica'] == 'Sí'
            is_dummy = G.nodes[v]['is_dummy'] or G.nodes[u]['is_dummy']
            
            color = 'red' if (is_u_crit and is_v_crit) else 'gray'
            style = 'dashed' if is_dummy else 'solid'
            
            # En el arco ponemos el símbolo + o - si el usuario lo pide
            edge_label = ""
            if tipo_grafica == "Revisión Adelante (Sumas)": edge_label = "+ suma"
            elif tipo_grafica == "Revisión Atrás (Restas)": edge_label = "- resta"
            
            dot.edge(u, v, label=edge_label, color=color, style=style, penwidth='2.0' if (is_u_crit and is_v_crit) else '1.0', fontcolor='gray')

        st.graphviz_chart(dot)

        # --- TABLA Y RESULTADOS ---
        st.header("4. Análisis Probabilístico y Financiero")
        col1, col2, col3 = st.columns(3)
        col1.metric("Duración Total Esperada (μ)", f"{project_mean:.2f}")
        col2.metric("Costo Total del Proyecto", f"${costo_total:,.2f}")
        col3.metric("Desviación Estándar (σ)", f"{project_sd:.4f}")
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
            
            # Título y Métricas
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(190, 10, 'Reporte PERT/CPM (Nodos Circulares y Costos)', 0, 1, 'C')
            pdf.ln(5)
            pdf.set_font('Arial', '', 12)
            pdf.cell(190, 8, f"Duracion Esperada: {project_mean:.2f}", 0, 1)
            pdf.cell(190, 8, f"Costo Total: ${costo_total:,.2f}", 0, 1)
            pdf.cell(190, 8, f"Ruta Critica: {path_str}", 0, 1)
            pdf.cell(190, 8, f"Probabilidad (T <= {target_time}): {prob_le:.2f}%", 0, 1)
            pdf.ln(5)
            
            # Log de Matemáticas (Suma/Resta)
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(190, 6, "Calculo Paso a Paso (Adelante / Sumas):", 0, 1)
            pdf.set_font('Arial', '', 9)
            for linea in log_adelante:
                pdf.cell(190, 5, linea, 0, 1)
            
            pdf.ln(3)
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(190, 6, "Calculo Paso a Paso (Atras / Restas):", 0, 1)
            pdf.set_font('Arial', '', 9)
            for linea in log_atras:
                pdf.cell(190, 5, linea, 0, 1)
                
            pdf.ln(5)
            
            # Gráfica
            fig = create_plot()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                fig.savefig(tmpfile.name, format="png")
                pdf.image(tmpfile.name, w=170)
            os.unlink(tmpfile.name)
            
            return pdf.output(dest='S').encode('latin1')

        if st.button("Generar y Descargar Reporte PDF Completo"):
            pdf_bytes = generate_pdf()
            st.download_button(label="Descargar PDF", data=pdf_bytes, file_name="Reporte_CPM_Completo.pdf", mime="application/pdf")
