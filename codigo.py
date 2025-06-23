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
        color='blue',
        weight=4,
        opacity=0.7,
        tooltip=f'Ruta optimizada {diresa}'
    ).add_to(mapa)
    
    if ruta_ors:
        folium.GeoJson(
            ruta_ors,
            name='Ruta por carretera',
            style_function=lambda x: {'color': 'red', 'weight': 5, 'opacity': 0.8},
            tooltip='Ruta real por carretera'
        ).add_to(mapa)
    
    for idx in ruta_optima:
        loc = ubicaciones[idx]
        is_start = (idx == ruta_optima[0])
        
        popup_content = f"""
        <b>{loc['nombre']}</b><br>
        <b>Orden en ruta:</b> {ruta_optima.index(idx)+1}<br>
        <b>DIRESA:</b> {loc.get('diresa', 'N/A')}<br>
        <b>Red:</b> {loc.get('red', 'N/A')}<br>
        <b>Dirección:</b> {loc.get('direccion', 'N/A')}<br>
        <b>Coordenadas:</b> {loc['coords'][0]:.6f}, {loc['coords'][1]:.6f}
        """
        
        folium.Marker(
            location=loc['coords'],
            popup=folium.Popup(popup_content, max_width=300),
            icon=folium.Icon(color='green' if is_start else 'blue', icon='flag' if is_start else 'info-sign'),
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
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f8f9fa; padding: 20px; border-radius: 5px; }
        .section { margin-bottom: 30px; border-bottom: 1px solid #eee; padding-bottom: 20px; }
        .map-container { width: 100%; height: 600px; margin: 20px 0; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        .summary-card { background: #e9f7fe; padding: 15px; border-radius: 5px; margin: 15px 0; }
        .route-step { padding: 5px 0; border-bottom: 1px dotted #ddd; }
        .highlight { background-color: #fffde7; padding: 2px 5px; border-radius: 3px; }
    </style>
    """
    
    tabla_establecimientos = ""
    for i, idx in enumerate(ruta_optima):
        loc = ubicaciones[idx]
        distancia_anterior = 0 if i == 0 else geodesic(ubicaciones[ruta_optima[i-1]]['coords'], loc['coords']).km
        
        tabla_establecimientos += f"""
        <tr>
            <td>{i+1}</td>
            <td>{loc.get('codigo', '')}</td>
            <td>{loc['nombre']}</td>
            <td>{loc.get('red', 'N/A')}</td>
            <td>{distancia_anterior:.2f} km</td>
            <td>{loc['coords'][0]:.6f}, {loc['coords'][1]:.6f}</td>
        </tr>
        """
    
    mapa_html = generar_mapa_mejorado(ubicaciones, ruta_optima, ruta_ors, diresa_seleccionada, embed=True)
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Reporte de Optimización - DIRESA {diresa_seleccionada}</title>
        {styles}
    </head>
    <body>
        <div class="header">
            <h1>Reporte de Optimización de Rutas</h1>
            <h2>DIRESA {diresa_seleccionada}</h2>
            <p>Generado el {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h3>Resumen de la Optimización</h3>
            <div class="summary-card">
                <p><strong>Total de establecimientos:</strong> {len(ruta_optima)}</p>
                <p><strong>Distancia total estimada:</strong> <span class="highlight">{distancia_total:.2f} km</span></p>
                <p><strong>Punto de inicio:</strong> {ubicaciones[ruta_optima[0]]['nombre']}</p>
                <p><strong>Redes involucradas:</strong> {', '.join(set(loc.get('red', 'N/A') for loc in ubicaciones))}</p>
            </div>
        </div>
        
        <div class="section">
            <h3>Ruta Óptima Recomendada</h3>
            <div style="background: #f5f5f5; padding: 15px; border-radius: 5px;">
                {"".join(f'<div class="route-step"><strong>{i+1}.</strong> {ubicaciones[idx]["nombre"]} <small>({ubicaciones[idx].get("red", "N/A")})</small> - Distancia: {geodesic(ubicaciones[ruta_optima[i-1]]["coords"], ubicaciones[idx]["coords"]).km:.2f} km desde anterior</div>' 
                 for i, idx in enumerate(ruta_optima) if i > 0)}
            </div>
        </div>
        
        <div class="section">
            <h3>Mapa Interactivo</h3>
            <div class="map-container">
                {mapa_html}
            </div>
            <p><small>Línea azul: Ruta lineal óptima | Línea roja: Ruta por carretera (cuando disponible)</small></p>
        </div>
        
        <div class="section">
            <h3>Detalle de Establecimientos</h3>
            <table>
                <thead>
                    <tr>
                        <th>Orden</th>
                        <th>Código</th>
                        <th>Nombre</th>
                        <th>Red</th>
                        <th>Distancia desde anterior</th>
                        <th>Coordenadas</th>
                    </tr>
                </thead>
                <tbody>
                    {tabla_establecimientos}
                </tbody>
            </table>
        </div>
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