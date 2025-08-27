import numpy as np
import cv2
import json
import os
import argparse
from collections import deque
import heapq
from scipy.ndimage import distance_transform_edt

class StereoArucoPathPlanner:
    def __init__(self, alpha=0.5):
        self.stereo_image = None
        self.left_image = None
        self.mask_image = None
        self.aruco_data = None
        self.navigable_map = None
        self.distance_map = None
        self.WATER_COLOR = np.array([41, 167, 224])
        self.alpha = alpha  # Opacità per la fusione delle zone navigabili
        
    def load_data(self, stereo_path, mask_path, json_path):
        """Carica l'immagine stereo, la maschera e i dati ArUco"""
        # Carica immagine stereo
        self.stereo_image = cv2.imread(stereo_path)
        if self.stereo_image is None:
            raise FileNotFoundError(f"Impossibile caricare immagine stereo: {stereo_path}")
        
        # Divide immagine stereo a metà e prende la sinistra
        height, width = self.stereo_image.shape[:2]
        half_width = width // 2
        self.left_image = self.stereo_image[:, :half_width]
        
        # Carica maschera
        self.mask_image = cv2.imread(mask_path)
        if self.mask_image is None:
            raise FileNotFoundError(f"Impossibile caricare maschera: {mask_path}")
        
        # La maschera ha dimensioni 672x384 e deve essere ridimensionata per matchare l'immagine sinistra
        # L'immagine sinistra ha larghezza = metà dell'immagine stereo
        target_width = self.left_image.shape[1]  # larghezza immagine sinistra
        target_height = self.left_image.shape[0]  # altezza immagine sinistra
        
        print(f"Maschera originale: {self.mask_image.shape}")
        print(f"Target per ridimensionamento: {target_height}x{target_width}")
        
        # Ridimensiona maschera per matchare esattamente l'immagine sinistra
        self.mask_image = cv2.resize(self.mask_image, (target_width, target_height))
        
        # Carica dati JSON ArUco
        with open(json_path, 'r', encoding='utf-8') as f:
            self.aruco_data = json.load(f)
        
        print(f"Immagine stereo caricata: {self.stereo_image.shape}")
        print(f"Immagine sinistra estratta: {self.left_image.shape}")
        print(f"Maschera caricata: {self.mask_image.shape}")
        print(f"Dati ArUco caricati: {len(self.aruco_data.get('tags', {}))} marcatori trovati")
        
    def create_navigable_map(self):
        """Crea mappa navigabile e calcola distanze dalle zone non navigabili"""
        # Converti BGR a RGB per confronto
        mask_rgb = cv2.cvtColor(self.mask_image, cv2.COLOR_BGR2RGB)
        
        # Trova pixel d'acqua (navigabili)
        water_mask = np.all(mask_rgb == self.WATER_COLOR, axis=-1)
        self.navigable_map = water_mask.astype(np.uint8)
        
        # Calcola mappa delle distanze dalle zone non navigabili
        self.distance_map = distance_transform_edt(self.navigable_map)
        
        print(f"Pixel d'acqua trovati: {np.sum(self.navigable_map)}")
        print(f"Distanza massima dal bordo: {np.max(self.distance_map):.1f}")
        
    def get_aruco_center_pixel(self):
        """Estrae la posizione del marcatore ArUco in coordinate pixel"""
        tags = self.aruco_data.get('tags', {})
        
        if not tags:
            print("Nessun marcatore ArUco trovato nei dati")
            return None, None
        
        # Prendi il primo marcatore disponibile (o potresti scegliere un ID specifico)
        tag_id = list(tags.keys())[0]
        tag_data = tags[tag_id]
        
        # Prendi il centro dal frame sinistro
        center_left = tag_data.get('center_left', [])
        if not center_left:
            print("Centro marcatore non trovato")
            return None, None
        
        # Coordinate pixel (x, y) -> (row, col) per numpy
        pixel_x = int(center_left[0])  # colonna
        pixel_y = int(center_left[1])  # riga
        
        # Estrai anche i dati di orientamento
        orientation = tag_data.get('orientation', {})
        position_3d = tag_data.get('position_3d', {})
        
        print(f"Marcatore ArUco {tag_id} trovato:")
        print(f"  Posizione pixel: ({pixel_x}, {pixel_y})")
        print(f"  Posizione 3D: x={position_3d.get('x', 0):.1f}cm, y={position_3d.get('y', 0):.1f}cm, z={position_3d.get('z', 0):.1f}cm")
        print(f"  Orientamento: roll={orientation.get('roll', 0):.1f}°, pitch={orientation.get('pitch', 0):.1f}°, yaw={orientation.get('yaw', 0):.1f}°")
        
        return (pixel_y, pixel_x), tag_data  # (row, col), dati completi
        
    def find_nearest_navigable_point(self, target_pos, max_search_radius=100):
        """Trova il punto navigabile più vicino al marcatore ArUco con ricerca estesa"""
        target_row, target_col = target_pos
        height, width = self.navigable_map.shape
        
        print(f"Cercando punto navigabile vicino al marcatore in posizione: ({target_row}, {target_col})")
        
        # Strategia 1: Cerca in cerchi concentrici con criteri rilassati
        candidates = []
        
        for radius in range(1, max_search_radius + 1):
            points_in_circle = max(8, radius * 2)  # Meno punti per cerchio per essere più veloce
            
            for angle in np.linspace(0, 2*np.pi, points_in_circle, endpoint=False):
                test_row = int(target_row + radius * np.sin(angle))
                test_col = int(target_col + radius * np.cos(angle))
                
                if (0 <= test_row < height and 0 <= test_col < width and 
                    self.navigable_map[test_row, test_col] == 1):
                    
                    # Prima verifica: deve avere almeno 1 pixel navigabile attorno (molto rilassato)
                    if self.has_navigable_neighbors((test_row, test_col), min_neighbors=1):
                        distance_from_target = np.sqrt((test_row - target_row)**2 + (test_col - target_col)**2)
                        safety_score = self.distance_map[test_row, test_col]
                        candidates.append((test_row, test_col, distance_from_target, safety_score))
            
            # Se abbiamo trovato candidati, fermiamoci al raggio più piccolo possibile
            if len(candidates) >= 5:  # Almeno 5 opzioni prima di fermarsi
                break
        
        # Strategia 2: Se non troviamo nulla, cerca semplicemente il pixel navigabile più vicino
        if not candidates:
            print("Strategia di ricerca estesa: cercando qualsiasi pixel navigabile...")
            water_pixels = np.where(self.navigable_map == 1)
            if len(water_pixels[0]) > 0:
                # Calcola distanze da tutti i pixel navigabili
                distances = np.sqrt((water_pixels[0] - target_row)**2 + (water_pixels[1] - target_col)**2)
                
                # Prendi i 10 più vicini
                closest_indices = np.argsort(distances)[:10]
                
                for idx in closest_indices:
                    row, col = water_pixels[0][idx], water_pixels[1][idx]
                    distance = distances[idx]
                    safety_score = self.distance_map[row, col]
                    candidates.append((row, col, distance, safety_score))
        
        if candidates:
            # Ordina per distanza dal target
            candidates.sort(key=lambda x: x[2])  # Solo per distanza, ignora sicurezza per garantire successo
            chosen = candidates[0]
            print(f"Punto navigabile trovato: ({chosen[0]}, {chosen[1]}), distanza={chosen[2]:.1f}px, sicurezza={chosen[3]:.1f}")
            return (chosen[0], chosen[1])
        else:
            print("ERRORE CRITICO: Nessun pixel navigabile trovato nell'intera immagine!")
            return None
    
    def has_navigable_neighbors(self, pos, min_neighbors=1):
        """Verifica se un punto ha abbastanza vicini navigabili"""
        x, y = pos
        height, width = self.navigable_map.shape
        
        neighbors_count = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                new_x, new_y = x + dx, y + dy
                if (0 <= new_x < height and 0 <= new_y < width and 
                    self.navigable_map[new_x, new_y] == 1):
                    neighbors_count += 1
                    if neighbors_count >= min_neighbors:
                        return True
        return neighbors_count >= min_neighbors
        
    def find_start_point(self):
        """Trova punto di inizio: sempre in basso al centro"""
        water_pixels = np.where(self.navigable_map == 1)
        
        if len(water_pixels[0]) == 0:
            return None
        
        height, width = self.navigable_map.shape
        
        # Punto di INIZIO: sempre in basso al centro dell'immagine
        start_x = height - 1  # Ultima riga (basso dello schermo)
        start_y = width // 2  # Centro orizzontale
        
        # Trova il pixel d'acqua più vicino al punto desiderato
        if self.navigable_map[start_x, start_y] == 1:
            start = (start_x, start_y)
        else:
            # Cerca il pixel d'acqua più vicino al centro nella parte bassa
            for row in range(height - 1, height - 20, -1):  # Ultime 20 righe
                if row < 0:
                    break
                water_cols = np.where(self.navigable_map[row] == 1)[0]
                if len(water_cols) > 0:
                    # Trova colonna più vicina al centro
                    closest_col = water_cols[np.argmin(np.abs(water_cols - start_y))]
                    start = (row, closest_col)
                    break
            else:
                # Fallback: primo pixel d'acqua trovato nella parte bassa
                bottom_pixels = water_pixels[0] >= height - 50
                if np.any(bottom_pixels):
                    idx = np.where(bottom_pixels)[0][0]
                    start = (water_pixels[0][idx], water_pixels[1][idx])
                else:
                    start = (water_pixels[0][-1], water_pixels[1][-1])
        
        return start
    
    def heuristic(self, a, b):
        """Euristica per A*: distanza Manhattan"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def get_cost(self, pos):
        """Calcola costo del movimento favorendo zone lontane dai bordi"""
        distance_from_edge = self.distance_map[pos[0], pos[1]]
        
        # Costo base = 1, ma penalizza zone vicine ai bordi
        if distance_from_edge < 2:
            return 10  # Molto costoso vicino ai bordi
        elif distance_from_edge < 5:
            return 3   # Moderatamente costoso
        else:
            return 1   # Costo normale per zone sicure
    
    def get_neighbors(self, pos):
        """Ottieni vicini navigabili con i loro costi"""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                    
                new_x, new_y = pos[0] + dx, pos[1] + dy
                
                if (0 <= new_x < self.navigable_map.shape[0] and 
                    0 <= new_y < self.navigable_map.shape[1] and
                    self.navigable_map[new_x, new_y] == 1):
                    
                    # Costo movimento diagonale vs dritto
                    move_cost = 1.4 if abs(dx) + abs(dy) == 2 else 1.0
                    edge_cost = self.get_cost((new_x, new_y))
                    total_cost = move_cost * edge_cost
                    
                    neighbors.append(((new_x, new_y), total_cost))
        return neighbors
    
    def astar_pathfinding(self, start, goal):
        """Algoritmo A* modificato per evitare bordi"""
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == goal:
                # Ricostruisci percorso
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            
            for neighbor, move_cost in self.get_neighbors(current):
                tentative_g = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None
    
    def smooth_path(self, path):
        """Semplice smoothing del percorso"""
        if len(path) < 3:
            return path
            
        smoothed = [path[0]]
        
        for i in range(1, len(path) - 1):
            prev_point = smoothed[-1]
            curr_point = path[i]
            next_point = path[i + 1]
            
            # Media pesata per smoothing
            smooth_x = int(0.2 * prev_point[0] + 0.6 * curr_point[0] + 0.2 * next_point[0])
            smooth_y = int(0.2 * prev_point[1] + 0.6 * curr_point[1] + 0.2 * next_point[1])
            
            # Verifica che il punto smoothed sia ancora navigabile
            if (0 <= smooth_x < self.navigable_map.shape[0] and 
                0 <= smooth_y < self.navigable_map.shape[1] and
                self.navigable_map[smooth_x, smooth_y] == 1):
                smoothed.append((smooth_x, smooth_y))
            else:
                smoothed.append(curr_point)
        
        smoothed.append(path[-1])
        return smoothed
    
    def plan_path(self):
        """Funzione principale per calcolare il percorso verso il marcatore ArUco"""
        if self.navigable_map is None:
            self.create_navigable_map()
        
        # Trova punto di partenza
        start = self.find_start_point()
        if start is None:
            print("Errore: non trovate zone d'acqua navigabili per il punto di partenza")
            return None, None, None
        
        # Trova posizione del marcatore ArUco
        aruco_pos, aruco_data = self.get_aruco_center_pixel()
        if aruco_pos is None:
            print("Errore: marcatore ArUco non trovato")
            return None, None, None
        
        # Trova punto navigabile più vicino al marcatore
        end = self.find_nearest_navigable_point(aruco_pos)
        if end is None:
            print("Errore: nessun punto navigabile trovato vicino al marcatore ArUco")
            return None, None, None
            
        print(f"Start: {start}, End: {end}, Marcatore ArUco: {aruco_pos}")
        
        # Calcola percorso con A* modificato
        path = self.astar_pathfinding(start, end)
        
        if path:
            print(f"Percorso trovato! Lunghezza: {len(path)} punti")
            # Applica smoothing
            path = self.smooth_path(path)
            return path, aruco_pos, aruco_data
        else:
            print("Nessun percorso trovato!")
            return None, aruco_pos, aruco_data
    
    def visualize_path(self, path, aruco_pos, aruco_data):
        """Visualizza il percorso sull'immagine sinistra con evidenziazione del marcatore ArUco"""
        result = self.left_image.copy()
        
        # Prima sovrapponi le zone navigabili (acqua) in azzurro trasparente
        water_overlay = result.copy()
        water_overlay[self.navigable_map == 1] = self.WATER_COLOR  # Azzurro acqua
        result = cv2.addWeighted(result, 1.0 - self.alpha, water_overlay, self.alpha, 0)
        
        # Visualizza mappa delle distanze per debug (solo sulle zone navigabili)
        distance_normalized = (self.distance_map / np.max(self.distance_map) * 255).astype(np.uint8)
        distance_colored = cv2.applyColorMap(distance_normalized, cv2.COLORMAP_JET)
        
        # Sovrapponi mappa distanze in trasparenza solo dove c'è acqua
        mask_3d = np.stack([self.navigable_map] * 3, axis=-1)
        result = np.where(mask_3d, cv2.addWeighted(result, 0.8, distance_colored, 0.2, 0), result)
        
        # Evidenzia il marcatore ArUco
        if aruco_pos and aruco_data:
            aruco_pixel_x = int(aruco_data.get('center_left', [0, 0])[0])
            aruco_pixel_y = int(aruco_data.get('center_left', [0, 0])[1])
            
            # Disegna riquadro attorno al marcatore
            corners_left = aruco_data.get('corners_left', [])
            if corners_left:
                corners_array = np.array(corners_left, dtype=np.int32)
                cv2.polylines(result, [corners_array], True, (255, 0, 255), 3)  # Magenta
            
            # Disegna centro del marcatore
            cv2.circle(result, (aruco_pixel_x, aruco_pixel_y), 8, (255, 0, 255), -1)  # Magenta
            cv2.circle(result, (aruco_pixel_x, aruco_pixel_y), 12, (255, 255, 255), 2)  # Bordo bianco
            
            # Disegna frecce per orientamento
            orientation = aruco_data.get('orientation', {})
            if orientation:
                # Freccia per yaw (direzione principale)
                yaw = orientation.get('yaw', 0)
                
                # Testo con orientamento
                text_y = aruco_pixel_y - 25
                cv2.putText(result, f"Yaw: {yaw:.1f}deg", (aruco_pixel_x - 40, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                cv2.putText(result, f"Pitch: {orientation.get('pitch', 0):.1f}deg", 
                           (aruco_pixel_x - 40, text_y - 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                cv2.putText(result, f"Roll: {orientation.get('roll', 0):.1f}deg", 
                           (aruco_pixel_x - 40, text_y - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Testo con ID e distanza 3D
            position_3d = aruco_data.get('position_3d', {})
            distance_3d = position_3d.get('z', 0)
            tag_id = aruco_data.get('id', 'N/A')
            
            cv2.putText(result, f"ArUco ID: {tag_id}", (aruco_pixel_x - 40, aruco_pixel_y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(result, f"Dist: {distance_3d:.1f}cm", (aruco_pixel_x - 40, aruco_pixel_y + 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Disegna il percorso
        if path:
            for i in range(len(path) - 1):
                pt1 = (path[i][1], path[i][0])  # (y, x) -> (x, y)
                pt2 = (path[i+1][1], path[i+1][0])
                cv2.line(result, pt1, pt2, (255, 255, 255), 4)  # Bianco spesso
                cv2.line(result, pt1, pt2, (0, 0, 255), 2)      # Rosso sopra
            
            # Segna inizio e fine
            start_pt = (path[0][1], path[0][0])
            end_pt = (path[-1][1], path[-1][0])
            cv2.circle(result, start_pt, 10, (255, 0, 0), -1)    # Blu inizio
            cv2.circle(result, end_pt, 10, (0, 255, 255), -1)     # Giallo fine
            
            # Testo informativo
            cv2.putText(result, "START", (start_pt[0] - 25, start_pt[1] - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(result, "TARGET", (end_pt[0] - 30, end_pt[1] - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result

def parse_arguments():
    """Parse degli argomenti da linea di comando"""
    parser = argparse.ArgumentParser(
        description='Path Planner per Canali con Marcatore ArUco Stereo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempio di utilizzo:
  python canal_path_planner_stereo.py --stereo stereo_frame.png --mask out_prova.png --json detection_data.json --output results/

Input richiesti:
  - stereo: Immagine stereo rettificata (da stereo_apriltag_processor)
  - mask: Maschera delle zone navigabili (PNG con acqua in azzurro)
  - json: Dati dei marcatori ArUco (da stereo_apriltag_processor)
        """
    )
    
    parser.add_argument(
        '--stereo',
        type=str,
        required=True,
        help='Path dell\'immagine stereo rettificata (richiesto)'
    )
    
    parser.add_argument(
        '--mask',
        type=str,
        required=True,
        help='Path della maschera delle zone navigabili (richiesto)'
    )
    
    parser.add_argument(
        '--json',
        type=str,
        required=True,
        help='Path del file JSON con i dati dei marcatori ArUco (richiesto)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='.',
        help='Cartella di output per i risultati (default: cartella corrente)'
    )
    
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.5,
        help='Opacità per la visualizzazione delle zone navigabili (default: 0.5)'
    )
    
    return parser.parse_args()

def main():
    # Parse degli argomenti
    args = parse_arguments()
    
    print("Avvio Path Planner per Canali con ArUco Stereo")
    print(f"OpenCV version: {cv2.__version__}")
    
    try:
        # Inizializza planner
        planner = StereoArucoPathPlanner(alpha=args.alpha)
        
        # Carica dati
        planner.load_data(args.stereo, args.mask, args.json)
        
        # Calcola percorso
        path, aruco_pos, aruco_data = planner.plan_path()
        
        if path:
            # Visualizza risultato
            result = planner.visualize_path(path, aruco_pos, aruco_data)
            
            # Crea cartella di output se non esiste
            if not os.path.exists(args.output):
                os.makedirs(args.output)
            
            # Salva risultato
            output_path = os.path.join(args.output, 'canal_aruco_path_result.png')
            cv2.imwrite(output_path, result)
            print(f"Risultato salvato in: {output_path}")
            
            # Salva anche dati del percorso in JSON
            path_data = {
                'path_points': [(int(p[0]), int(p[1])) for p in path],
                'aruco_position': (int(aruco_pos[0]), int(aruco_pos[1])) if aruco_pos else None,
                'aruco_data': aruco_data,
                'path_length': len(path),
                'total_distance_pixels': sum(
                    np.sqrt((path[i+1][0] - path[i][0])**2 + (path[i+1][1] - path[i][1])**2)
                    for i in range(len(path)-1)
                ) if len(path) > 1 else 0
            }
            
            json_output_path = os.path.join(args.output, 'path_data.json')
            with open(json_output_path, 'w', encoding='utf-8') as f:
                json.dump(path_data, f, indent=2, ensure_ascii=False)
            print(f"Dati percorso salvati in: {json_output_path}")
            
            # Mostra finestra
            cv2.namedWindow('Path Planning Canale verso ArUco', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Path Planning Canale verso ArUco', result)
            
            print("Premi un tasto per chiudere...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Impossibile calcolare il percorso!")
            
    except FileNotFoundError as e:
        print(f"Errore: file non trovato - {e}")
    except Exception as e:
        print(f"Errore durante l'elaborazione: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()