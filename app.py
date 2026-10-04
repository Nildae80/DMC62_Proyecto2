import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración inicial de la página
st.set_page_config(
    page_title="Proyecto 2 | EDA Bank Marketing Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo global de gráficos
sns.set_theme(style="whitegrid")


# ==========================================
# PROGRAMACIÓN ORIENTADA A OBJETOS (POO)
# ==========================================
class DataAnalyzer:
    """Clase encargada de encapsular el análisis estadístico y la generación de gráficos."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    def get_summary_info(self) -> pd.DataFrame:
        """Retorna un resumen con tipos de datos y valores nulos."""
        info_df = pd.DataFrame({
            'Tipo de Dato': self.df.dtypes.astype(str),
            'Valores Nulos': self.df.isnull().sum(),
            '% Nulos': (self.df.isnull().sum() / len(self.df)) * 100
        })
        return info_df

    def get_variable_classification(self):
        """Clasifica y cuenta las variables del dataset."""
        return {
            'Numéricas': (len(self.num_cols), self.num_cols),
            'Categóricas': (len(self.cat_cols), self.cat_cols)
        }

    def get_descriptive_stats(self) -> pd.DataFrame:
        """Genera estadísticas descriptivas numéricas detalladas."""
        desc = self.df.describe().T
        desc['median'] = self.df[self.num_cols].median()
        return desc[['count', 'mean', 'std', 'median', 'min', '25%', '50%', '75%', 'max']]

    def plot_numerical_distribution(self, column: str):
        """Genera un histograma con KDE para una variable numérica."""
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(self.df[column], kde=True, ax=ax, color='#1f77b4')
        ax.set_title(f"Distribución de: {column}", fontsize=12, fontweight='bold')
        ax.set_xlabel(column)
        ax.set_ylabel("Frecuencia")
        return fig

    def plot_categorical_count(self, column: str):
        """Genera un gráfico de barras para variables categóricas."""
        fig, ax = plt.subplots(figsize=(9, 4))
        order = self.df[column].value_counts().index
        sns.countplot(data=self.df, x=column, order=order, ax=ax, palette='Blues_r')
        plt.xticks(rotation=45, ha='right')
        ax.set_title(f"Frecuencia por categoría: {column}", fontsize=12, fontweight='bold')
        ax.set_xlabel(column)
        ax.set_ylabel("Cantidad")
        return fig

    def plot_bivariate_num_cat(self, num_var: str, cat_var: str):
        """Genera un Boxplot comparativo entre variable numérica y categórica."""
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(data=self.df, x=cat_var, y=num_var, ax=ax, palette='Set2')
        ax.set_title(f"Análisis Bivariado: {num_var} vs {cat_var}", fontsize=12, fontweight='bold')
        return fig

    def plot_bivariate_cat_cat(self, cat_var1: str, cat_var2: str):
        """Genera un gráfico de barras apiladas porcentuales entre dos categóricas."""
        ct = pd.crosstab(self.df[cat_var1], self.df[cat_var2], normalize='index') * 100
        fig, ax = plt.subplots(figsize=(9, 4))
        ct.plot(kind='bar', stacked=True, ax=ax, colormap='Accent')
        ax.set_title(f"Proporción de {cat_var2} según {cat_var1}", fontsize=12, fontweight='bold')
        ax.set_ylabel("Porcentaje (%)")
        plt.xticks(rotation=45, ha='right')
        plt.legend(title=cat_var2, bbox_to_anchor=(1.05, 1), loc='upper left')
        return fig


# ==========================================
# MENÚ NAVEGABLE EN LA BARRA LATERAL (SIDEBAR)
# ==========================================
st.sidebar.title("📌 Menú Principal")
#opcion_menu = st.sidebar.radio(
#    "Seleccione un Módulo:",
#    ["Home", "Carga de Dataset", "EDA (Análisis Exploratorio)", "Conclusiones"]
#)
imagen = st.sidebar.image("Python_logo.png", width=200)
opcion_menu = st.sidebar.selectbox("Selecciones el Módulo",["Home", "Carga de Dataset", "EDA (Análisis Exploratorio)", "Conclusiones"])
st.sidebar.image("DMC.png", width=150)

# Estado global del dataset en session_state
if 'df' not in st.session_state:
    st.session_state['df'] = None


# ==========================================
# MÓDULO 1: HOME
# ==========================================
if opcion_menu == "Home":
    st.title("🏦 Análisis Exploratorio de Datos: Campaña Bank Marketing")
    st.subheader("Caso de Estudio N°1 - Especialización Python for Analytics")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 Objetivo del Proyecto
        Esta aplicación interactiva permite realizar un **Análisis Exploratorio de Datos (EDA)** exhaustivo sobre los resultados de las campañas de marketing directo de una institución financiera. El propósito es identificar los factores clave que inciden en la contratación de depósitos a plazo (`y`) para optimizar futuras estrategias comerciales.
        
        ---
        ### 👨‍💻 Datos del Autor
        * **Nombre Completo:** [Tu Nombre y Apellidos]
        * **Curso:** Especialización en Python for Analytics
        * **Institución:** DILIC Institute
        * **Año:** 2026
        
        ---
        ### 🛠️ Tecnologías Utilizadas
        * **Lenguaje:** Python 3.10+
        * **Interfaz Interactiva:** Streamlit
        * **Procesamiento de Datos:** Pandas & NumPy
        * **Visualización:** Matplotlib & Seaborn
        """)
        
    with col2:
        st.info("""
        **Contexto de Negocio:**
        En los últimos 6 meses, la efectividad comercial cayo del **12% al 8%**. Este dashboard busca proveer herramientas de diagnóstico analítico sin modelos predictivos para entender el comportamiento de la tasa de conversión.
        """)


# ==========================================
# MÓDULO 2: CARGA DEL DATASET
# ==========================================
elif opcion_menu == "Carga de Dataset":
    st.title("📂 Módulo de Carga del Dataset")
    st.markdown("Suba el archivo `BankMarketing.csv` para habilitar las funcionalidades de análisis.")

    uploaded_file = st.file_uploader("Seleccione el archivo CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            # Detección de separadores comunes en el dataset
            st.session_state['df'] = pd.read_csv(uploaded_file, sep=None, engine='python')
            st.success("✅ Archivo cargado correctamente.")
        except Exception as e:
            st.error(f"Error al leer el archivo: {e}")

    if st.session_state['df'] is not None:
        df = st.session_state['df']
        
        st.subheader("📊 Métricas Generales del Dataset")
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Total de Filas (Registros)", f"{df.shape[0]:,}")
        col_m2.metric("Total de Columnas (Variables)", f"{df.shape[1]}")

        st.subheader("👀 Vista Previa de los Datos (Head)")
        st.dataframe(df.head(10), use_container_width=True)
    else:
        st.warning("⚠️ Debe cargar el archivo `BankMarketing.csv` para continuar.")


# ==========================================
# MÓDULO 3: EDA (ANÁLISIS EXPLORATORIO DE DATOS)
# ==========================================
elif opcion_menu == "EDA (Análisis Exploratorio)":
    st.title("🔍 Análisis Exploratorio de Datos (EDA)")

    if st.session_state['df'] is None:
        st.error("❌ No se ha cargado ningún dataset. Vaya al módulo **'Carga de Dataset'** primero.")
    else:
        df = st.session_state['df']
        analyzer = DataAnalyzer(df)

        # Tabs interactivos para organizar los 10 ítems
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 1-3. Estructura y Stats", 
            "📊 4-6. Faltantes y Univariado", 
            "📈 7-8. Análisis Bivariado", 
            "🎛️ 9-10. Dinámico y Hallazgos"
        ])

        # --- TAB 1: ITEMS 1, 2 Y 3 ---
        with tab1:
            st.header("Ítem 1: Información General del Dataset")
            st.dataframe(analyzer.get_summary_info(), use_container_width=True)

            st.markdown("---")
            st.header("Ítem 2: Clasificación de Variables")
            clasif = analyzer.get_variable_classification()
            c1, c2 = st.columns(2)
            with c1:
                st.subheader(f"🔢 Numéricas ({clasif['Numéricas'][0]})")
                st.write(clasif['Numéricas'][1])
            with c2:
                st.subheader(f"🔤 Categóricas ({clasif['Categóricas'][0]})")
                st.write(clasif['Categóricas'][1])

            st.markdown("---")
            st.header("Ítem 3: Estadísticas Descriptivas")
            st.dataframe(analyzer.get_descriptive_stats(), use_container_width=True)
            st.caption("Interpretación: Muestra la tendencia central (media/mediana) y la dispersión (desviación estándar) de cada variable continua.")

        # --- TAB 2: ITEMS 4, 5 Y 6 ---
        with tab2:
            st.header("Ítem 4: Análisis de Valores Faltantes")
            nulls = df.isnull().sum()
            total_nulls = nulls.sum()
            if total_nulls == 0:
                st.success("✅ No se detectaron valores nulos explícitos (`NaN`) en el dataset. Sin embargo, existen registros codificados como `'unknown'` en variables categóricas.")
            else:
                st.bar_chart(nulls[nulls > 0])

            st.markdown("---")
            st.header("Ítem 5: Distribución de Variables Numéricas")
            selected_num = st.selectbox("Seleccione Variable Numérica:", analyzer.num_cols, key="num_select")
            st.pyplot(analyzer.plot_numerical_distribution(selected_num))

            st.markdown("---")
            st.header("Ítem 6: Análisis de Variables Categóricas")
            selected_cat = st.selectbox("Seleccione Variable Categórica:", analyzer.cat_cols, key="cat_select")
            
            c_graph, c_table = st.columns([2, 1])
            with c_graph:
                st.pyplot(analyzer.plot_categorical_count(selected_cat))
            with c_table:
                counts = df[selected_cat].value_counts()
                props = df[selected_cat].value_counts(normalize=True) * 100
                st.dataframe(pd.DataFrame({'Conteo': counts, 'Proporción (%)': props.round(2)}))

        # --- TAB 3: ITEMS 7 Y 8 ---
        with tab3:
            st.header("Ítem 7: Análisis Bivariado (Numérico vs Categórico)")
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                var_num = st.selectbox("Variable Numérica:", analyzer.num_cols, index=analyzer.num_cols.index('duration') if 'duration' in analyzer.num_cols else 0)
            with col_b2:
                var_cat_b = st.selectbox("Variable Categórica Target:", analyzer.cat_cols, index=analyzer.cat_cols.index('y') if 'y' in analyzer.cat_cols else 0)
            st.pyplot(analyzer.plot_bivariate_num_cat(var_num, var_cat_b))

            st.markdown("---")
            st.header("Ítem 8: Análisis Bivariado (Categórico vs Categórico)")
            col_bb1, col_bb2 = st.columns(2)
            with col_bb1:
                var_c1 = st.selectbox("Variable Categórica 1:", analyzer.cat_cols, index=analyzer.cat_cols.index('education') if 'education' in analyzer.cat_cols else 0)
            with col_bb2:
                var_c2 = st.selectbox("Variable Categórica 2:", analyzer.cat_cols, index=analyzer.cat_cols.index('y') if 'y' in analyzer.cat_cols else 0, key="cat2")
            st.pyplot(analyzer.plot_bivariate_cat_cat(var_c1, var_c2))

        # --- TAB 4: ITEMS 9 Y 10 ---
        with tab4:
            st.header("Ítem 9: Análisis Basado en Parámetros Seleccionados")
            st.markdown("Filtre dinámicamente los datos mediante múltiples widgets interactivos.")

            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                job_filter = st.multiselect("Filtrar por Trabajo (job):", options=df['job'].unique(), default=df['job'].unique()[:3])
            with col_f2:
                age_range = st.slider("Rango de Edad (age):", min_value=int(df['age'].min()), max_value=int(df['age'].max()), value=(25, 60))
            with col_f3:
                show_accepted_only = st.checkbox("Mostrar solo respuestas positivas (y = 'yes')")

            # Aplicación de filtros
            df_filtered = df[
                (df['job'].isin(job_filter)) & 
                (df['age'].between(age_range[0], age_range[1]))
            ]
            if show_accepted_only:
                df_filtered = df_filtered[df_filtered['y'] == 'yes']

            st.write(f"🔍 **Registros encontrados:** {len(df_filtered)} de {len(df)}")
            st.dataframe(df_filtered.head(15), use_container_width=True)

            st.markdown("---")
            st.header("Ítem 10: Hallazgos Clave Resumen")
            
            # Resumen analítico visual
            kpi1, kpi2, kpi3 = st.columns(3)
            conv_rate = (df['y'] == 'yes').mean() * 100
            avg_dur_yes = df[df['y'] == 'yes']['duration'].mean()
            avg_dur_no = df[df['y'] == 'no']['duration'].mean()

            kpi1.metric("Conversión Global", f"{conv_rate:.2f}%")
            kpi2.metric("Duración Promedio (Éxito)", f"{avg_dur_yes:.0f} sec")
            kpi3.metric("Duración Promedio (Rechazo)", f"{avg_dur_no:.0f} sec")

            fig_summary, ax_sum = plt.subplots(figsize=(8, 3))
            df.groupby('contact')['y'].value_counts(normalize=True).unstack().plot(kind='barh', stacked=True, ax=ax_sum, color=['#e74c3c', '#2ecc71'])
            ax_sum.set_title("Efectividad según Canal de Contacto (Proporción %)")
            st.pyplot(fig_summary)


# ==========================================
# MÓDULO 4: CONCLUSIONES FINALES
# ==========================================
elif opcion_menu == "Conclusiones":
    st.title("💡 Conclusiones Finales y Decisiones de Negocio")
    st.markdown("Basado en el Análisis Exploratorio de Datos real sobre las 41,188 interacciones:")

    st.markdown("""
    1. **La duración de la llamada es el factor determinante crítico:** Las llamadas que resultaron en contratación (`yes`) tuvieron una duración promedio superior a **9 minutos (553 segundos)**, en comparación con solo **3.6 minutos (220 segundos)** en las rechazadas.
       * *Decisión:* Capacitar a la fuerza comercial en guías de conversación de mayor valor que mantengan al cliente enganchado más tiempo en lugar de realizar llamadas breves y automatizadas.

    2. **Superioridad del canal Celular frente al Teléfono Fijo:** La tasa de conversión a través de teléfonos celulares (**14.7%**) casi triplica la efectividad del teléfono fijo (**5.2%**).
       * *Decisión:* Priorizar bases de datos con números celulares actualizados y restringir campañas masivas a teléfonos fijos tradicionales.

    3. **Segmentos de alta conversión desatendidos:** Los estudiantes (**31.4%**) y los jubilados (**25.2%**) presentan las tasas de aceptación más altas de toda la base, a pesar de no representar el mayor volumen de llamadas.
       * *Decisión:* Diseñar productos de depósito a plazo fijo específicos y personalizados para estudiantes (ahorro joven) y jubilados (renta segura).

    4. **Saturación por exceso de contactos en la misma campaña:** La efectividad decae notablemente a medida que aumenta la cantidad de contactos (`campaign`). Intentos superiores a 3 o 4 llamadas por cliente generan un retorno marginal casi nulo.
       * *Decisión:* Establecer una política de "máximo 3 reintentos" por cliente por campaña para no quemar la base ni agotar recursos operativos.

    5. **Efecto de estacionalidad en la conversión:** Meses como marzo, diciembre, octubre y septiembre superan el **40% de conversión**, mientras que mayo (el mes con mayor volumen de llamadas) registra una efectividad muy baja (**6.4%**).
       * *Decisión:* Reasignar el presupuesto comercial concentrando los esfuerzos en los meses de alta receptividad en lugar de saturar al cliente en mayo.
    """)
