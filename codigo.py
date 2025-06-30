import pulp
import pandas as pd
import openrouteservice as ors
import folium
from geopy.distance import geodesic
from tqdm import tqdm
import warnings
from datetime import datetime

# Configuración
CSV_FILE = r'/home/nando/Documentos/Investigacion de operaciones/TB_EESS.csv'  # Cambiar según tu archivo
ORS_API_KEY = '5b3ce3597851110001cf6248678c770556f749e09be74f98c1abd911'  # Tu API Key de OpenRouteService
OUTPUT_HTML = 'resultados_optimizacion.html'
MAX_UBICACIONES = 50  # Límite de ubicaciones para evitar procesamiento extenso
warnings.filterwarnings('ignore')

def detectar_columnas(archivo):
    """Detecta automáticamente las columnas relevantes en el archivo CSV"""
    try:
        df_sample = pd.read_csv(archivo, sep=';', nrows=5)
        
        print("\nColumnas disponibles en el archivo:")
        print(df_sample.columns.tolist())
        
        mapeo_columnas = {
            'id_eess': ['id_eess', 'codigo', 'id'],
            'nombre': ['nombre', 'establecimiento', 'eess'],
            'diresa': ['diresa', 'direccion regional', 'region'],
            'red': ['red', 'red de salud', 'microred'],
            'direccion': ['direccion', 'ubicacion'],
            'longitud': ['longitud', 'lon', 'x'],
            'latitud': ['latitud', 'lat', 'y']
        }
        
        columnas_encontradas = {}
        for col_esperada, posibles_nombres in mapeo_columnas.items():
            for nombre in posibles_nombres:
                if nombre in df_sample.columns:
                    columnas_encontradas[col_esperada] = nombre
                    break
        
        if 'latitud' not in columnas_encontradas or 'longitud' not in columnas_encontradas:
            raise ValueError("No se encontraron columnas de latitud/longitud")
        
        return columnas_encontradas
    except Exception as e:
        print(f"Error al detectar columnas: {e}")
        return None

def leer_y_filtrar_ubicaciones(archivo, diresa_seleccionada=None):
    """Lee y filtra ubicaciones desde archivo CSV"""
    try:
        df = pd.read_csv(archivo, sep=';')
        
        print("\nColumnas encontradas en el archivo:")
        print(df.columns.tolist())
        
        columnas_requeridas = ['nombre', 'longitud', 'latitud']
        for col in columnas_requeridas:
            if col not in df.columns:
                raise ValueError(f"Columna requerida '{col}' no encontrada")
        
        df = df.dropna(subset=['latitud', 'longitud'])
        df['latitud'] = pd.to_numeric(df['latitud'], errors='coerce')
        df['longitud'] = pd.to_numeric(df['longitud'], errors='coerce')
        df = df.dropna(subset=['latitud', 'longitud'])
        
        if diresa_seleccionada and 'diresa' in df.columns:
            df['diresa'] = df['diresa'].str.strip().str.upper()
            df = df[df['diresa'] == diresa_seleccionada.upper()]
        
        if len(df) > MAX_UBICACIONES:
            print(f"\nAdvertencia: Se procesarán las primeras {MAX_UBICACIONES} de {len(df)} ubicaciones")
            df = df.head(MAX_UBICACIONES)
        
        ubicaciones = []
        for _, row in df.iterrows():
            ubicacion = {
                'id': str(row.get('id_eess', '')),
                'codigo': str(row.get('codigo_renaes', '')),
                'nombre': str(row['nombre']).strip(),
                'coords': (float(row['latitud']), float(row['longitud']))
            }
            
            if 'diresa' in row:
                ubicacion['diresa'] = str(row['diresa']).strip()
            if 'red' in row:
                ubicacion['red'] = str(row['red']).strip()
            if 'direccion' in row:
                ubicacion['direccion'] = str(row['direccion']).strip()
            
            ubicaciones.append(ubicacion)
        
        return ubicaciones
    except Exception as e:
        print(f"\nError al procesar archivo: {str(e)}")
        return []           

def calcular_matriz_distancias(ubicaciones):
    """Calcula la matriz de distancias entre ubicaciones"""
    n = len(ubicaciones)
    distancias = [[0] * n for _ in range(n)]
    
    print("\nCalculando distancias entre ubicaciones...")
    for i in tqdm(range(n)):
        for j in range(i+1, n):
            dist = geodesic(ubicaciones[i]['coords'], ubicaciones[j]['coords']).km
            distancias[i][j] = dist
            distancias[j][i] = dist
    
    return distancias

def resolver_problema_ruteo(ubicaciones, distancias, punto_inicio=0):
    """Resuelve el problema del agente viajero usando programación lineal"""
    n = len(ubicaciones)
    prob = pulp.LpProblem("Problema_de_Ruteo", pulp.LpMinimize)
    
    x = pulp.LpVariable.dicts("x", ((i, j) for i in range(n) for j in range(n) if i != j), cat='Binary')
    u = pulp.LpVariable.dicts("u", (i for i in range(n)), lowBound=0, upBound=n-1, cat='Continuous')
    
    prob += pulp.lpSum(distancias[i][j] * x[i, j] for i in range(n) for j in range(n) if i != j)
    
    for j in range(n):
        prob += pulp.lpSum(x[i, j] for i in range(n) if i != j) == 1
    
    for i in range(n):
        prob += pulp.lpSum(x[i, j] for j in range(n) if j != i) == 1
    
    for i in range(n):
        for j in range(n):
            if i != j and i != punto_inicio and j != punto_inicio:
                prob += u[i] - u[j] + n * x[i, j] <= n - 1
    
    prob += u[punto_inicio] == 0
    
    solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=300)
    prob.solve(solver)
    
    ruta = [punto_inicio]
    actual = punto_inicio
    
    while True:
        for j in range(n):
            if j != actual and pulp.value(x[actual, j]) == 1:
                ruta.append(j)
                actual = j
                break
        if actual == punto_inicio or len(ruta) == n:
            break
    
    return ruta

def obtener_rutas_reales(ubicaciones, ruta_optima, api_key):
    """Obtiene las rutas reales usando OpenRouteService"""
    if len(ruta_optima) > 25:
        print("Advertencia: OpenRouteService tiene límites para más de 25 ubicaciones")
        return None
    
    client = ors.Client(key=api_key)
    coords = [ubicaciones[i]['coords'][::-1] for i in ruta_optima]
    
    try:
        ruta = client.directions(
            coordinates=coords,
            profile='driving-car',
            format='geojson',
            optimize_waypoints=True
        )
        return ruta
    except Exception as e:
        print(f"Error al obtener rutas de OpenRouteService: {e}")
        return None

def generar_mapa_mejorado(ubicaciones, ruta_optima, ruta_ors, diresa, embed=False):
    """Genera mapa interactivo con información detallada"""
    mapa = folium.Map(location=ubicaciones[ruta_optima[0]]['coords'], zoom_start=12)
    
    puntos_ruta = [ubicaciones[i]['coords'] for i in ruta_optima]
    folium.PolyLine(
        puntos_ruta,
        color='#2563eb',
        weight=4,
        opacity=0.8,
        tooltip=f'Ruta optimizada {diresa}'
    ).add_to(mapa)
    
    if ruta_ors:
        folium.GeoJson(
            ruta_ors,
            name='Ruta por carretera',
            style_function=lambda x: {'color': '#dc2626', 'weight': 5, 'opacity': 0.8},
            tooltip='Ruta real por carretera'
        ).add_to(mapa)
    
    for idx in ruta_optima:
        loc = ubicaciones[idx]
        is_start = (idx == ruta_optima[0])
        
        popup_content = f"""
        <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; min-width: 250px;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 12px; margin: -9px -9px 12px -9px; border-radius: 4px 4px 0 0;">
                <h4 style="margin: 0; font-size: 16px; font-weight: 600;">{loc['nombre']}</h4>
            </div>
            <div style="padding: 0 4px;">
                <p style="margin: 8px 0; display: flex; align-items: center;">
                    <span style="background: #3b82f6; color: white; padding: 2px 8px; border-radius: 12px; font-size: 12px; font-weight: 600; margin-right: 8px;">#{ruta_optima.index(idx)+1}</span>
                    <strong>Orden en ruta</strong>
                </p>
                <p style="margin: 8px 0;"><strong>DIRESA:</strong> <span style="color: #6b7280;">{loc.get('diresa', 'N/A')}</span></p>
                <p style="margin: 8px 0;"><strong>Red:</strong> <span style="color: #6b7280;">{loc.get('red', 'N/A')}</span></p>
                <p style="margin: 8px 0;"><strong>Dirección:</strong> <span style="color: #6b7280;">{loc.get('direccion', 'N/A')}</span></p>
                <p style="margin: 8px 0; font-size: 12px; color: #9ca3af;"><strong>Coordenadas:</strong> {loc['coords'][0]:.6f}, {loc['coords'][1]:.6f}</p>
            </div>
        </div>
        """
        
        folium.Marker(
            location=loc['coords'],
            popup=folium.Popup(popup_content, max_width=300),
            icon=folium.Icon(
                color='green' if is_start else 'blue', 
                icon='play' if is_start else 'info-sign',
                prefix='fa' if is_start else 'glyphicon'
            ),
            tooltip=f"{ruta_optima.index(idx)+1}. {loc['nombre']}"
        ).add_to(mapa)
    
    if embed:
        return mapa._repr_html_()
    else:
        mapa.save('mapa_temp.html')
        return None

def generar_reporte_html(ubicaciones, ruta_optima, ruta_ors, diresa_seleccionada, distancia_total):
    """Genera un reporte HTML completo con los resultados"""
    styles = """
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #1f2937;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header { 
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
            margin-bottom: 30px;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 10px;
        }
        
        .header h2 {
            font-size: 1.5rem;
            color: #4b5563;
            font-weight: 500;
            margin-bottom: 15px;
        }
        
        .header p {
            color: #6b7280;
            font-size: 1rem;
        }
        
        .section { 
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            margin-bottom: 30px;
            border-radius: 16px;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .section-header {
            background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%);
            padding: 20px 30px;
            border-bottom: 1px solid #e5e7eb;
        }
        
        .section-header h3 {
            font-size: 1.5rem;
            font-weight: 600;
            color: #374151;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .section-content {
            padding: 30px;
        }
        
        .map-container { 
            width: 100%;
            height: 600px;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        
        .stat-card {
            background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
            color: white;
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        .stat-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        }
        
        .stat-card.success {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        }
        
        .stat-card.warning {
            background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        }
        
        .stat-card.info {
            background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
        }
        
        .stat-number {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 5px;
            display: block;
        }
        
        .stat-label {
            font-size: 0.9rem;
            opacity: 0.9;
            font-weight: 500;
        }
        
        table { 
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
        }
        
        th {
            background: linear-gradient(135deg, #374151 0%, #1f2937 100%);
            color: white;
            padding: 16px 12px;
            text-align: left;
            font-weight: 600;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        td {
            padding: 16px 12px;
            border-bottom: 1px solid #f3f4f6;
            font-size: 0.9rem;
        }
        
        tr:nth-child(even) { 
            background: #f9fafb;
        }
        
        tr:hover {
            background: #f3f4f6;
            transition: background-color 0.2s ease;
        }
        
        .route-timeline {
            position: relative;
            padding-left: 30px;
        }
        
        .route-timeline::before {
            content: '';
            position: absolute;
            left: 15px;
            top: 0;
            bottom: 0;
            width: 2px;
            background: linear-gradient(to bottom, #3b82f6, #1d4ed8);
        }
        
        .route-step {
            position: relative;
            padding: 20px 25px;
            margin-bottom: 15px;
            background: white;
            border-radius: 12px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            border-left: 4px solid #3b82f6;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        .route-step:hover {
            transform: translateX(5px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
        }
        
        .route-step::before {
            content: '';
            position: absolute;
            left: -32px;
            top: 25px;
            width: 12px;
            height: 12px;
            background: #3b82f6;
            border-radius: 50%;
            border: 3px solid white;
            box-shadow: 0 0 0 2px #3b82f6;
        }
        
        .route-step.start::before {
            background: #10b981;
            box-shadow: 0 0 0 2px #10b981;
        }
        
        .route-step-number {
            display: inline-block;
            background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.8rem;
            margin-right: 12px;
        }
        
        .route-step-name {
            font-weight: 600;
            font-size: 1.1rem;
            color: #1f2937;
            margin-bottom: 5px;
        }
        
        .route-step-details {
            color: #6b7280;
            font-size: 0.9rem;
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }
        
        .distance-badge {
            background: #fef3c7;
            color: #92400e;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 500;
        }
        
        .legend {
            background: #f8fafc;
            padding: 15px 20px;
            border-radius: 8px;
            border-left: 4px solid #3b82f6;
            margin-top: 15px;
        }
        
        .legend-item {
            display: inline-flex;
            align-items: center;
            margin-right: 25px;
            font-size: 0.9rem;
            color: #4b5563;
        }
        
        .legend-color {
            width: 20px;
            height: 3px;
            margin-right: 8px;
            border-radius: 2px;
        }
        
        .legend-color.blue {
            background: #3b82f6;
        }
        
        .legend-color.red {
            background: #dc2626;
        }
        
        .order-badge {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            padding: 4px 8px;
            border-radius: 50%;
            font-weight: 600;
            font-size: 0.8rem;
            min-width: 24px;
            height: 24px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
        }
        
        .icon {
            width: 20px;
            height: 20px;
            margin-right: 8px;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }
            
            .header {
                padding: 20px;
            }
            
            .header h1 {
                font-size: 2rem;
            }
            
            .section-content {
                padding: 20px;
            }
            
            .stats-grid {
                grid-template-columns: 1fr;
            }
            
            .route-step-details {
                flex-direction: column;
                gap: 5px;
            }
            
            table {
                font-size: 0.8rem;
            }
            
            th, td {
                padding: 8px 6px;
            }
        }
        
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #3498db;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .fade-in {
            animation: fadeIn 0.5s ease-in;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
    """
    
    tabla_establecimientos = ""
    for i, idx in enumerate(ruta_optima):
        loc = ubicaciones[idx]
        distancia_anterior = 0 if i == 0 else geodesic(ubicaciones[ruta_optima[i-1]]['coords'], loc['coords']).km
        
        tabla_establecimientos += f"""
        <tr>
            <td><span class="order-badge">{i+1}</span></td>
            <td><strong>{loc.get('codigo', 'N/A')}</strong></td>
            <td>{loc['nombre']}</td>
            <td><span style="background: #e0f2fe; color: #0277bd; padding: 4px 8px; border-radius: 12px; font-size: 0.8rem;">{loc.get('red', 'N/A')}</span></td>
            <td><span class="distance-badge">{distancia_anterior:.2f} km</span></td>
            <td style="font-family: monospace; font-size: 0.8rem; color: #6b7280;">{loc['coords'][0]:.6f}, {loc['coords'][1]:.6f}</td>
        </tr>
        """
    
    # Generar timeline de ruta
    ruta_timeline = ""
    for i, idx in enumerate(ruta_optima):
        loc = ubicaciones[idx]
        distancia_anterior = 0 if i == 0 else geodesic(ubicaciones[ruta_optima[i-1]]['coords'], loc['coords']).km
        is_start = i == 0
        
        ruta_timeline += f"""
        <div class="route-step {'start' if is_start else ''}">
            <div class="route-step-name">
                <span class="route-step-number">{i+1}</span>
                {loc['nombre']}
            </div>
            <div class="route-step-details">
                <span>Red: {loc.get('red', 'N/A')}</span>
                {f'<span class="distance-badge">{distancia_anterior:.2f} km desde anterior</span>' if not is_start else '<span style="background: #dcfce7; color: #166534; padding: 2px 8px; border-radius: 12px; font-size: 0.8rem; font-weight: 500;">Punto de inicio</span>'}
            </div>
        </div>
        """
    
    mapa_html = generar_mapa_mejorado(ubicaciones, ruta_optima, ruta_ors, diresa_seleccionada, embed=True)
    
    # Calcular estadísticas adicionales
    redes_involucradas = set(loc.get('red', 'N/A') for loc in ubicaciones)
    tiempo_estimado = distancia_total / 60 * 1.5  # Estimación aproximada en horas
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Reporte de Optimización - DIRESA {diresa_seleccionada}</title>
        {styles}
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    </head>
    <body>
        <div class="container">
            <div class="header fade-in">
                <h1><i class="fas fa-route"></i> Optimización de Rutas</h1>
                <h2>DIRESA {diresa_seleccionada}</h2>
                <p><i class="fas fa-calendar-alt"></i> Generado el {datetime.now().strftime('%d de %B de %Y a las %H:%M:%S')}</p>
            </div>
            
            <div class="section fade-in">
                <div class="section-header">
                    <h3><i class="fas fa-chart-line"></i> Resumen Ejecutivo</h3>
                </div>
                <div class="section-content">
                    <div class="stats-grid">
                        <div class="stat-card">
                            <span class="stat-number">{len(ruta_optima)}</span>
                            <span class="stat-label">Establecimientos</span>
                        </div>
                        <div class="stat-card success">
                            <span class="stat-number">{distancia_total:.1f}</span>
                            <span class="stat-label">Kilómetros totales</span>
                        </div>
                        <div class="stat-card warning">
                            <span class="stat-number">{tiempo_estimado:.1f}</span>
                            <span class="stat-label">Horas estimadas</span>
                        </div>
                        <div class="stat-card info">
                            <span class="stat-number">{len(redes_involucradas)}</span>
                            <span class="stat-label">Redes de salud</span>
                        </div>
                    </div>
                    
                    <div style="background: #f0f9ff; border: 1px solid #0ea5e9; border-radius: 8px; padding: 20px; margin-top: 20px;">
                        <h4 style="color: #0c4a6e; margin-bottom: 10px;"><i class="fas fa-info-circle"></i> Información clave</h4>
                        <p style="margin-bottom: 8px;"><strong>Punto de inicio:</strong> {ubicaciones[ruta_optima[0]]['nombre']}</p>
                        <p style="margin-bottom: 8px;"><strong>Redes involucradas:</strong> {', '.join(sorted(redes_involucradas))}</p>
                        <p><strong>Método de optimización:</strong> Algoritmo del Viajante (TSP) con programación lineal</p>
                    </div>
                </div>
            </div>
            
            <div class="section fade-in">
                <div class="section-header">
                    <h3><i class="fas fa-map-marked-alt"></i> Mapa Interactivo</h3>
                </div>
                <div class="section-content">
                    <div class="map-container">
                        {mapa_html}
                    </div>
                    <div class="legend">
                        <div class="legend-item">
                            <div class="legend-color blue"></div>
                            Ruta lineal optimizada
                        </div>
                        <div class="legend-item">
                            <div class="legend-color red"></div>
                            Ruta por carretera (cuando disponible)
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="section fade-in">
                <div class="section-header">
                    <h3><i class="fas fa-route"></i> Secuencia de Visitas Recomendada</h3>
                </div>
                <div class="section-content">
                    <div class="route-timeline">
                        {ruta_timeline}
                    </div>
                </div>
            </div>
            
            <div class="section fade-in">
                <div class="section-header">
                    <h3><i class="fas fa-table"></i> Detalle Completo de Establecimientos</h3>
                </div>
                <div class="section-content">
                    <div style="overflow-x: auto;">
                        <table>
                            <thead>
                                <tr>
                                    <th><i class="fas fa-sort-numeric-down"></i> Orden</th>
                                    <th><i class="fas fa-barcode"></i> Código</th>
                                    <th><i class="fas fa-hospital"></i> Nombre del Establecimiento</th>
                                    <th><i class="fas fa-network-wired"></i> Red de Salud</th>
                                    <th><i class="fas fa-road"></i> Distancia</th>
                                    <th><i class="fas fa-map-pin"></i> Coordenadas</th>
                                </tr>
                            </thead>
                            <tbody>
                                {tabla_establecimientos}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            // Agregar animaciones suaves al cargar
            document.addEventListener('DOMContentLoaded', function() {{
                const elements = document.querySelectorAll('.fade-in');
                elements.forEach((el, index) => {{
                    setTimeout(() => {{
                        el.style.opacity = '1';
                        el.style.transform = 'translateY(0)';
                    }}, index * 200);
                }});
            }});
            
            // Mejorar interactividad de la tabla
            const rows = document.querySelectorAll('tbody tr');
            rows.forEach(row => {{
                row.addEventListener('click', function() {{
                    this.style.background = '#dbeafe';
                    setTimeout(() => {{
                        this.style.background = '';
                    }}, 1000);
                }});
            }});
        </script>
    </body>
    </html>
    """
    
    with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\nReporte completo generado en {OUTPUT_HTML}")

def main():
    print("=== OPTIMIZADOR DE RUTAS PARA ESTABLECIMIENTOS DE SALUD ===")
    print(f"\nCargando datos desde {CSV_FILE}...")
    
    # Paso 1: Mostrar opciones de DIRESA disponibles
    try:
        df = pd.read_csv(CSV_FILE, sep=';', nrows=1)
        
        diresa_col = None
        posibles_nombres = ['diresa', 'DIRESA', 'Diresa', 'Direccion_Regional']
        for col in df.columns:
            if col.lower() in [name.lower() for name in posibles_nombres]:
                diresa_col = col
                break
        
        if not diresa_col:
            print("\nNo se encontró columna DIRESA en el archivo. Se procesarán todos los establecimientos.")
            diresa_seleccionada = None
        else:
            df_diresas = pd.read_csv(CSV_FILE, sep=';', usecols=[diresa_col])
            diresas = sorted(df_diresas[diresa_col].str.strip().str.upper().dropna().unique())
            
            print("\nDIRESAS disponibles:")
            for i, diresa in enumerate(diresas):
                print(f"{i+1}. {diresa}")
            
            while True:
                seleccion = input("\nIngrese el número de la DIRESA que desea optimizar (o 0 para todas): ")
                if seleccion.isdigit():
                    seleccion = int(seleccion)
                    if 0 <= seleccion <= len(diresas):
                        break
                print("Por favor ingrese un número válido.")
            
            if seleccion == 0:
                diresa_seleccionada = None
                print("\nSe procesarán todas las DIRESAS")
            else:
                diresa_seleccionada = diresas[seleccion-1]
                print(f"\nDIRESA seleccionada: {diresa_seleccionada}")
    except Exception as e:
        print(f"\nError al leer DIRESAS: {e}")
        return
    
    # Paso 2: Filtrar y cargar ubicaciones
    ubicaciones = leer_y_filtrar_ubicaciones(CSV_FILE, diresa_seleccionada)
    
    if not ubicaciones:
        print("\nNo se encontraron ubicaciones con los criterios especificados")
        return
    
    print(f"\nSe trabajará con {len(ubicaciones)} ubicaciones:")
    for i, loc in enumerate(ubicaciones[:10]):
        print(f"{i+1}. {loc['nombre']} ({loc.get('red', 'Sin red')})")
    if len(ubicaciones) > 10:
        print(f"... y {len(ubicaciones)-10} más")
    
    # Paso 3: Seleccionar punto de inicio
    print("\nSELECCIONE EL PUNTO DE INICIO DE LA RUTA:")
    print("----------------------------------------")
    print("Establecimientos disponibles:")
    for i, loc in enumerate(ubicaciones):
        red = loc.get('red', 'Sin red')
        print(f"{i}. {loc['nombre']} ({red})")
    
    while True:
        try:
            punto_inicio = int(input(f"\nIngrese el número del establecimiento inicial (0-{len(ubicaciones)-1}): "))
            if 0 <= punto_inicio < len(ubicaciones):
                break
            print(f"Por favor ingrese un número entre 0 y {len(ubicaciones)-1}")
        except ValueError:
            print("Por favor ingrese un número válido.")
    
    print(f"\nPunto de inicio seleccionado: {ubicaciones[punto_inicio]['nombre']}")
    
    # Paso 4: Calcular distancias y optimizar ruta
    distancias = calcular_matriz_distancias(ubicaciones)
    ruta_optima = resolver_problema_ruteo(ubicaciones, distancias, punto_inicio)
    
    print("\nRuta óptima calculada:")
    for i, idx in enumerate(ruta_optima):
        distancia_anterior = 0 if i == 0 else geodesic(
            ubicaciones[ruta_optima[i-1]]['coords'], 
            ubicaciones[idx]['coords']
        ).km
        print(f"{i+1}. {ubicaciones[idx]['nombre']} ({distancia_anterior:.2f} km desde anterior)")
    
    distancia_total = sum(
        geodesic(ubicaciones[ruta_optima[i]]['coords'], 
                ubicaciones[ruta_optima[i+1]]['coords']).km
        for i in range(len(ruta_optima)-1)
    )
    print(f"\nDistancia total estimada: {distancia_total:.2f} km")
    
    # Paso 5: Obtener rutas reales (si aplica)
    if 1 < len(ruta_optima) <= 25:
        print("\nObteniendo rutas reales de OpenRouteService...")
        ruta_ors = obtener_rutas_reales(ubicaciones, ruta_optima, ORS_API_KEY)
    else:
        ruta_ors = None
        if len(ruta_optima) > 25:
            print("Demasiadas ubicaciones para obtener rutas reales (máximo 25)")
    
    # Paso 6: Generar reporte HTML completo
    generar_reporte_html(ubicaciones, ruta_optima, ruta_ors, 
                        diresa_seleccionada or "TODAS", distancia_total)

if __name__ == "__main__":
    main()