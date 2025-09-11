#!/usr/bin/env python3
"""
Client per visualizzazione Web Service April Tag Distance Detection
Versione: Client 1.2 - Extended with Right Camera Data Display

Requisiti:
pip install opencv-python numpy requests

Uso:
python stereo_apriltag_client.py [--server IP_ADDRESS] [--port PORT]

Comandi durante la visualizzazione:
- Q: Esci
- R: Reset filtri server
- S: Screenshot
- D: Toggle debug info
- F: Toggle fullscreen
- P: Toggle posizione 3D
- O: Toggle orientamento
- 1: Vista standard
- 2: Vista dettagliata posizione
- 3: Vista dettagliata orientamento
"""

import cv2
import numpy as np
import requests
import json
import argparse
import time
import threading
from datetime import datetime
import logging
from config_loader import ConfigLoader

# =============================================================================
# CONFIGURAZIONE
# =============================================================================

class ClientConfig(ConfigLoader):
    """Configurazione per il client di visualizzazione"""
    
    def __init__(self):
        # Parametri di rete
        self.DEFAULT_SERVER_IP = self.get_nested_value("network", "default_server_ip")
        self.DEFAULT_SERVER_PORT = self.get_nested_value("network", "default_server_port")
        
        # Parametri visualizzazione
        self.WINDOW_NAME = self.get_nested_value("client_display", "window_name")
        self.UPDATE_INTERVAL = self.get_nested_value("client_display", "update_interval")
        
        # Colori per visualizzazione (convertiti in tuple BGR)
        self.COLOR_MARKER_BOX = tuple(self.get_nested_value("client_display", "colors", "marker_box"))
        self.COLOR_MARKER_CENTER = tuple(self.get_nested_value("client_display", "colors", "marker_center"))
        self.COLOR_TEXT_DISTANCE = tuple(self.get_nested_value("client_display", "colors", "text_distance"))
        self.COLOR_TEXT_INFO = tuple(self.get_nested_value("client_display", "colors", "text_info"))
        self.COLOR_TEXT_DEBUG = tuple(self.get_nested_value("client_display", "colors", "text_debug"))
        self.COLOR_TEXT_POSITION = tuple(self.get_nested_value("client_display", "colors", "text_position"))
        self.COLOR_TEXT_ORIENTATION = tuple(self.get_nested_value("client_display", "colors", "text_orientation"))
        self.COLOR_SEPARATOR = tuple(self.get_nested_value("client_display", "colors", "separator"))
        
        # Parametri testo
        self.FONT_SCALE_DISTANCE = self.get_nested_value("client_display", "fonts", "scale_distance")
        self.FONT_SCALE_INFO = self.get_nested_value("client_display", "fonts", "scale_info")
        self.FONT_SCALE_DEBUG = self.get_nested_value("client_display", "fonts", "scale_debug")
        self.FONT_SCALE_DETAIL = self.get_nested_value("client_display", "fonts", "scale_detail")
        self.FONT_THICKNESS = self.get_nested_value("client_display", "fonts", "thickness")

# =============================================================================
# CLIENT VISUALIZZATORE
# =============================================================================

class AprilTagClient:
    def __init__(self, server_ip, server_port):
        self.server_ip = server_ip
        self.server_port = server_port
        self.base_url = f"http://{server_ip}:{server_port}"
        self.config = ClientConfig()
        
        # Variabili per i dati
        self.current_frame = None
        self.detection_data = {}
        self.server_status = {}
        self.is_running = False
        
        # Lock per thread safety
        self.data_lock = threading.Lock()
        
        # Parametri visualizzazione
        self.show_debug = False
        self.show_position_3d = True
        self.show_orientation = True
        self.fullscreen = False
        self.view_mode = 1  # 1=standard, 2=dettaglio posizione, 3=dettaglio orientamento
        
        # Statistiche
        self.stats = {
            'frames_received': 0,
            'data_updates': 0,
            'last_update_time': 0,
            'fps': 0
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Client inizializzato per server {self.base_url}")
    
    def test_connection(self):
        """
        Testa la connessione al server
        """
        try:
            response = requests.get(f"{self.base_url}/status", timeout=5)
            if response.status_code == 200:
                self.server_status = response.json()
                self.logger.info("Connessione al server OK")
                self.logger.info(f"Server status: {self.server_status.get('status', 'unknown')}")
                return True
            else:
                self.logger.error(f"Server risponde con codice {response.status_code}")
                return False
        except Exception as e:
            self.logger.error(f"Errore connessione al server: {e}")
            return False
    
    def fetch_video_frame(self):
        """
        Scarica un frame video dal server
        """
        try:
            response = requests.get(f"{self.base_url}/video_feed", stream=True, timeout=1)
            if response.status_code == 200:
                # Leggi il boundary del multipart
                bytes_data = b''
                for chunk in response.iter_content(chunk_size=1024):
                    if not self.is_running:
                        break
                    bytes_data += chunk
                    
                    # Cerca marker di inizio frame
                    start_marker = b'\xff\xd8'  # JPEG start marker
                    end_marker = b'\xff\xd9'    # JPEG end marker
                    
                    start_pos = bytes_data.find(start_marker)
                    if start_pos != -1:
                        end_pos = bytes_data.find(end_marker, start_pos)
                        if end_pos != -1:
                            # Estrai frame JPEG
                            jpeg_data = bytes_data[start_pos:end_pos + 2]
                            
                            # Decodifica frame
                            nparr = np.frombuffer(jpeg_data, np.uint8)
                            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            
                            if frame is not None:
                                with self.data_lock:
                                    self.current_frame = frame
                                    self.stats['frames_received'] += 1
                                
                                # Calcola FPS
                                current_time = time.time()
                                if self.stats['last_update_time'] > 0:
                                    delta_time = current_time - self.stats['last_update_time']
                                    if delta_time > 0:
                                        self.stats['fps'] = 1.0 / delta_time
                                self.stats['last_update_time'] = current_time
                            
                            # Pulisci buffer
                            bytes_data = bytes_data[end_pos + 2:]
                            break
        except Exception as e:
            if self.is_running:
                self.logger.error(f"Errore ricezione frame: {e}")
    
    def fetch_detection_data(self):
        """
        Scarica i dati di rilevamento dal server
        """
        try:
            response = requests.get(f"{self.base_url}/detection_data", timeout=1)
            if response.status_code == 200:
                data = response.json()
                with self.data_lock:
                    self.detection_data = data
                    self.stats['data_updates'] += 1
        except Exception as e:
            if self.is_running:
                self.logger.error(f"Errore ricezione dati: {e}")
    
    def reset_server_filters(self):
        """
        Resetta i filtri sul server
        """
        try:
            response = requests.post(f"{self.base_url}/reset_filters", timeout=5)
            if response.status_code == 200:
                self.logger.info("Filtri server resettati")
                return True
        except Exception as e:
            self.logger.error(f"Errore reset filtri: {e}")
        return False
    
    def draw_marker_info_standard(self, frame, tag_data, is_right_frame=False):
        """
        Disegna informazioni del marcatore sul frame - Vista Standard
        """
        tag_id = tag_data['id']
        distance = tag_data['distance_smooth']
        
        # Seleziona i dati della camera appropriata
        if is_right_frame:
            corners = np.array(tag_data['corners_right'], dtype=np.float32)
            center = tag_data['center_right']
            position_3d = tag_data.get('right_camera', {}).get('position_3d_smooth', {})
            orientation = tag_data.get('right_camera', {}).get('orientation_smooth', {})
            camera_label = "R"
        else:
            corners = np.array(tag_data['corners_left'], dtype=np.float32)
            center = tag_data['center_left']
            position_3d = tag_data.get('left_camera', {}).get('position_3d_smooth', 
                                     tag_data.get('position_3d_smooth', tag_data.get('position_3d', {})))
            orientation = tag_data.get('left_camera', {}).get('orientation_smooth',
                                     tag_data.get('orientation_smooth', tag_data.get('orientation', {})))
            camera_label = "L"
        
        # Disegna riquadro del marcatore
        corners_int = corners.astype(int)
        cv2.polylines(frame, [corners_int], True, self.config.COLOR_MARKER_BOX, 3)
        
        # Disegna centro
        center_int = (int(center[0]), int(center[1]))
        cv2.circle(frame, center_int, 5, self.config.COLOR_MARKER_CENTER, -1)
        
        # Informazioni da visualizzare
        info_lines = [f"ID {tag_id} ({camera_label}): {distance:.1f}cm"]
        
        if self.show_position_3d and position_3d:
            info_lines.append(f"XYZ: ({position_3d.get('x', 0):.1f}cm, {position_3d.get('y', 0):.1f}cm, {position_3d.get('z', 0):.1f}cm)")
        
        if self.show_orientation and orientation:
            info_lines.append(f"RPY: ({orientation.get('roll', 0):.1f}deg, {orientation.get('pitch', 0):.1f}deg, {orientation.get('yaw', 0):.1f}deg)")
        
        # Calcola dimensioni del box di testo
        max_width = 0
        total_height = 0
        line_heights = []
        
        for line in info_lines:
            text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 
                                       self.config.FONT_SCALE_INFO, 1)[0]
            max_width = max(max_width, text_size[0])
            line_heights.append(text_size[1])
            total_height += text_size[1] + 5
        
        # Posiziona testo sopra il marcatore
        text_x = max(10, center_int[0] - max_width // 2)
        text_y = max(total_height + 10, center_int[1] - 30)
        
        # Sfondo per il testo
        cv2.rectangle(frame, 
                     (text_x - 5, text_y - total_height - 5),
                     (text_x + max_width + 10, text_y + 5),
                     (0, 0, 0, 180), -1)
        
        # Disegna le linee di testo
        current_y = text_y - total_height + line_heights[0]
        
        for i, line in enumerate(info_lines):
            if i == 0:  # Distanza
                color = self.config.COLOR_TEXT_DISTANCE
            elif "XYZ" in line:  # Posizione 3D
                color = self.config.COLOR_TEXT_POSITION
            elif "RPY" in line:  # Orientamento
                color = self.config.COLOR_TEXT_ORIENTATION
            else:
                color = self.config.COLOR_TEXT_INFO
            
            cv2.putText(frame, line, (text_x, current_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, self.config.FONT_SCALE_INFO,
                       color, 1)
            
            current_y += line_heights[i] + 5
    
    def draw_marker_info_position_detail(self, frame, tag_data, is_right_frame=False):
        """
        Disegna informazioni dettagliate di posizione - Vista Posizione
        """
        tag_id = tag_data['id']
        
        # Seleziona i dati della camera appropriata
        if is_right_frame:
            corners = np.array(tag_data['corners_right'], dtype=np.float32)
            center = tag_data['center_right']
            position_3d = tag_data.get('right_camera', {}).get('position_3d_smooth', {})
            position_3d_raw = tag_data.get('right_camera', {}).get('position_3d', {})
            camera_label = "DESTRA"
        else:
            corners = np.array(tag_data['corners_left'], dtype=np.float32)
            center = tag_data['center_left']
            position_3d = tag_data.get('left_camera', {}).get('position_3d_smooth',
                                     tag_data.get('position_3d_smooth', tag_data.get('position_3d', {})))
            position_3d_raw = tag_data.get('left_camera', {}).get('position_3d',
                                         tag_data.get('position_3d', {}))
            camera_label = "SINISTRA"

        # Disegna riquadro del marcatore
        corners_int = corners.astype(int)
        cv2.polylines(frame, [corners_int], True, self.config.COLOR_MARKER_BOX, 2)
        cv2.circle(frame, (int(center[0]), int(center[1])), 3, self.config.COLOR_MARKER_CENTER, -1)
        
        # Informazioni dettagliate posizione
        info_lines = [
            f"TAG ID {tag_id} - POSIZIONE 3D ({camera_label})",
            f"X: {position_3d.get('x', 0):.2f}cm (raw: {position_3d_raw.get('x', 0):.2f})",
            f"Y: {position_3d.get('y', 0):.2f}cm (raw: {position_3d_raw.get('y', 0):.2f})",
            f"Z: {position_3d.get('z', 0):.2f}cm (raw: {position_3d_raw.get('z', 0):.2f})",
            f"Distanza: {tag_data['distance_smooth']:.2f}cm"
        ]
        
        # Posiziona il box informazioni in alto a sinistra del marcatore
        text_x = max(10, int(center[0]) - 200)
        text_y = max(120, int(center[1]) - 50)
        
        # Calcola dimensioni box
        max_width = 0
        total_height = len(info_lines) * 20 + 10
        
        for line in info_lines:
            text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 
                                       self.config.FONT_SCALE_DETAIL, 1)[0]
            max_width = max(max_width, text_size[0])
        
        # Sfondo
        cv2.rectangle(frame, 
                     (text_x - 5, text_y - total_height),
                     (text_x + max_width + 10, text_y + 5),
                     (0, 0, 0, 200), -1)
        
        # Disegna testo
        for i, line in enumerate(info_lines):
            y = text_y - total_height + 20 + i * 20
            if i == 0:  # Titolo
                color = self.config.COLOR_TEXT_DISTANCE
            else:
                color = self.config.COLOR_TEXT_POSITION
            
            cv2.putText(frame, line, (text_x, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, self.config.FONT_SCALE_DETAIL,
                       color, 1)
    
    def draw_marker_info_orientation_detail(self, frame, tag_data, is_right_frame=False):
        """
        Disegna informazioni dettagliate di orientamento - Vista Orientamento
        """
        tag_id = tag_data['id']
        
        # Seleziona i dati della camera appropriata
        if is_right_frame:
            corners = np.array(tag_data['corners_right'], dtype=np.float32)
            center = tag_data['center_right']
            orientation = tag_data.get('right_camera', {}).get('orientation_smooth', {})
            orientation_raw = tag_data.get('right_camera', {}).get('orientation', {})
            camera_label = "DESTRA"
        else:
            corners = np.array(tag_data['corners_left'], dtype=np.float32)
            center = tag_data['center_left']
            orientation = tag_data.get('left_camera', {}).get('orientation_smooth',
                                     tag_data.get('orientation_smooth', tag_data.get('orientation', {})))
            orientation_raw = tag_data.get('left_camera', {}).get('orientation',
                                         tag_data.get('orientation', {}))
            camera_label = "SINISTRA"
        
        # Disegna riquadro del marcatore
        corners_int = corners.astype(int)
        cv2.polylines(frame, [corners_int], True, self.config.COLOR_MARKER_BOX, 2)
        cv2.circle(frame, (int(center[0]), int(center[1])), 3, self.config.COLOR_MARKER_CENTER, -1)
        
        # Informazioni dettagliate orientamento
        info_lines = [
            f"TAG ID {tag_id} - ORIENTAMENTO ({camera_label})",
            f"Roll:  {orientation.get('roll', 0):.2f} deg (raw: {orientation_raw.get('roll', 0):.2f} deg)",
            f"Pitch: {orientation.get('pitch', 0):.2f} deg (raw: {orientation_raw.get('pitch', 0):.2f} deg)",
            f"Yaw:   {orientation.get('yaw', 0):.2f} deg (raw: {orientation_raw.get('yaw', 0):.2f}deg)",
            f"Distanza: {tag_data['distance_smooth']:.1f}cm"
        ]
        
        # Posiziona il box informazioni 
        text_x = max(10, int(center[0]) - 200)
        text_y = max(120, int(center[1]) - 50)
        
        # Calcola dimensioni box
        max_width = 0
        total_height = len(info_lines) * 20 + 10
        
        for line in info_lines:
            text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 
                                       self.config.FONT_SCALE_DETAIL, 1)[0]
            max_width = max(max_width, text_size[0])
        
        # Sfondo
        cv2.rectangle(frame, 
                     (text_x - 5, text_y - total_height),
                     (text_x + max_width + 10, text_y + 5),
                     (0, 0, 0, 200), -1)
        
        # Disegna testo
        for i, line in enumerate(info_lines):
            y = text_y - total_height + 20 + i * 20
            if i == 0:  # Titolo
                color = self.config.COLOR_TEXT_DISTANCE
            else:
                color = self.config.COLOR_TEXT_ORIENTATION
            
            cv2.putText(frame, line, (text_x, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, self.config.FONT_SCALE_DETAIL,
                       color, 1)
    
    def draw_marker_info(self, frame, tag_data, is_right_frame=False):
        """
        Disegna informazioni del marcatore in base alla modalità di visualizzazione
        """
        if self.view_mode == 1:
            self.draw_marker_info_standard(frame, tag_data, is_right_frame)
        elif self.view_mode == 2:
            self.draw_marker_info_position_detail(frame, tag_data, is_right_frame)
        elif self.view_mode == 3:
            self.draw_marker_info_orientation_detail(frame, tag_data, is_right_frame)
    
    def draw_overlay_info(self, frame):
        """
        Disegna informazioni generali overlay
        """
        height, width = frame.shape[:2]
        
        # Informazioni generali in alto a sinistra
        info_lines = []
        
        with self.data_lock:
            tags_count = len(self.detection_data.get('tags', {}))
            frame_count = self.detection_data.get('frame_count', 0)
        
        view_names = {1: "Standard", 2: "Posizione 3D", 3: "Orientamento"}
        
        info_lines.append(f"Tags rilevati: {tags_count}")
        info_lines.append(f"FPS: {self.stats['fps']:.1f}")
        info_lines.append(f"Vista: {view_names.get(self.view_mode, 'Sconosciuta')}")
        info_lines.append(f"Frames: {self.stats['frames_received']}")
        
        # Disegna sfondo per info generali
        info_height = len(info_lines) * 25 + 10
        cv2.rectangle(frame, (10, 10), (280, info_height), (0, 0, 0, 128), -1)
        
        for i, line in enumerate(info_lines):
            y = 30 + i * 25
            cv2.putText(frame, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX,
                       self.config.FONT_SCALE_INFO, self.config.COLOR_TEXT_INFO, 1)
        
        # Informazioni di calibrazione in basso a sinistra
        with self.data_lock:
            calib_info = self.detection_data.get('calibration_info', {})
        
        if calib_info:
            calib_lines = [
                f"RMS: {calib_info.get('rms_error', 0):.3f}px",
                f"Baseline: {calib_info.get('baseline', 0):.2f}cm"
            ]
            
            calib_height = len(calib_lines) * 20 + 10
            calib_y_start = height - calib_height - 10
            
            cv2.rectangle(frame, (10, calib_y_start), (220, height - 10), (0, 0, 0, 128), -1)
            
            for i, line in enumerate(calib_lines):
                y = calib_y_start + 20 + i * 20
                cv2.putText(frame, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX,
                           self.config.FONT_SCALE_DEBUG, self.config.COLOR_TEXT_INFO, 1)
        
        # Comandi in basso a destra
        commands = ["Q:Esci", "R:Reset", "S:Screen", "1/2/3:Viste", "P:Pos", "O:Ori", "D:Debug", "F:Full"]
        command_text = " | ".join(commands)
        
        text_size = cv2.getTextSize(command_text, cv2.FONT_HERSHEY_SIMPLEX,
                                   self.config.FONT_SCALE_DEBUG, 1)[0]
        
        cmd_x = width - text_size[0] - 15
        cmd_y = height - 15
        
        cv2.rectangle(frame, (cmd_x - 5, cmd_y - 20), (width - 10, height - 5), (0, 0, 0, 128), -1)
        cv2.putText(frame, command_text, (cmd_x, cmd_y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                   self.config.FONT_SCALE_DEBUG, self.config.COLOR_TEXT_INFO, 1)
        
        # Separatore stereo (se il frame è stereo)
        if width > height:  # Probabilmente stereo side-by-side
            separator_x = width // 2
            cv2.line(frame, (separator_x, 0), (separator_x, height), self.config.COLOR_SEPARATOR, 2)
            
            # Etichette sinistra/destra
            cv2.putText(frame, "SINISTRA", (50, height - 50), cv2.FONT_HERSHEY_SIMPLEX,
                       0.8, self.config.COLOR_SEPARATOR, 2)
            cv2.putText(frame, "DESTRA", (separator_x + 50, height - 50), cv2.FONT_HERSHEY_SIMPLEX,
                       0.8, self.config.COLOR_SEPARATOR, 2)
    
    def process_frame(self):
        """
        Elabora il frame corrente con le informazioni di rilevamento
        """
        with self.data_lock:
            if self.current_frame is None:
                return None
            
            frame = self.current_frame.copy()
            tags_data = self.detection_data.get('tags', {})
        
        height, width = frame.shape[:2]
        
        # Determina se è un frame stereo side-by-side
        is_stereo = width > height
        
        if is_stereo:
            # Frame stereo - separa le due metà
            separator_x = width // 2
            
            # Disegna informazioni per ogni tag rilevato
            for tag_id, tag_data in tags_data.items():
                # Solo se il tag ha dati per entrambe le telecamere
                if 'corners_left' in tag_data and 'corners_right' in tag_data:
                    # Disegna sulla parte sinistra
                    self.draw_marker_info(frame, tag_data, is_right_frame=False)
                    
                    # Disegna sulla parte destra - aggiusta le coordinate per la parte destra del frame
                    # Crea una copia dei dati del tag con coordinate aggiustate per la parte destra
                    right_tag_data = tag_data.copy()
                    if 'corners_right' in tag_data and 'center_right' in tag_data:
                        # Le coordinate della camera destra sono già corrette per la loro metà del frame
                        # Ma dobbiamo aggiustarle per la posizione nella metà destra del frame completo
                        corners_right_adjusted = np.array(tag_data['corners_right'])
                        corners_right_adjusted[:, 0] += separator_x  # Sposta X verso destra
                        
                        center_right_adjusted = [
                            tag_data['center_right'][0] + separator_x,
                            tag_data['center_right'][1]
                        ]
                        
                        # Aggiorna i dati per il disegno sulla parte destra
                        right_tag_data['corners_right'] = corners_right_adjusted.tolist()
                        right_tag_data['center_right'] = center_right_adjusted
                        
                        self.draw_marker_info(frame, right_tag_data, is_right_frame=True)
        else:
            # Frame singolo - disegna normalmente (solo camera sinistra)
            for tag_id, tag_data in tags_data.items():
                self.draw_marker_info(frame, tag_data, is_right_frame=False)
        
        # Disegna overlay con informazioni generali
        self.draw_overlay_info(frame)
        
        return frame
    
    def save_screenshot(self):
        """
        Salva uno screenshot
        """
        frame = self.process_frame()
        if frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"apriltag_screenshot_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            self.logger.info(f"Screenshot salvato: {filename}")
            return filename
        return None
    
    def data_update_thread(self):
        """
        Thread per aggiornare i dati dal server
        """
        while self.is_running:
            self.fetch_detection_data()
            time.sleep(0.1)  # 10 Hz per i dati
    
    def video_update_thread(self):
        """
        Thread per aggiornare il video dal server
        """
        while self.is_running:
            self.fetch_video_frame()
            time.sleep(self.config.UPDATE_INTERVAL)
    
    def run(self):
        """
        Esegue il client di visualizzazione
        """
        # Testa connessione
        if not self.test_connection():
            self.logger.error("Impossibile connettersi al server")
            return False
        
        self.is_running = True
        
        # Avvia thread per aggiornamento dati
        data_thread = threading.Thread(target=self.data_update_thread, daemon=True)
        data_thread.start()
        
        # Avvia thread per aggiornamento video
        video_thread = threading.Thread(target=self.video_update_thread, daemon=True)
        video_thread.start()
        
        # Aspetta il primo frame
        self.logger.info("Attendendo primo frame...")
        while self.is_running and self.current_frame is None:
            time.sleep(0.1)
        
        if self.current_frame is None:
            self.logger.error("Nessun frame ricevuto")
            return False
        
        # Crea finestra
        frame_height, frame_width = self.current_frame.shape[:2]
        cv2.namedWindow(self.config.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.config.WINDOW_NAME, frame_width, frame_height)
        
        self.logger.info("Client avviato - comandi disponibili:")
        self.logger.info("  Q: Esci, R: Reset filtri, S: Screenshot")
        self.logger.info("  1/2/3: Cambia vista, P: Toggle posizione, O: Toggle orientamento")
        self.logger.info("  D: Toggle debug, F: Toggle fullscreen")
        self.logger.info("  NOVITÀ: Visualizzazione dati camera destra sui frame stereo")
        
        try:
            while self.is_running:
                display_frame = self.process_frame()
                
                if display_frame is not None:
                    if self.fullscreen:
                        cv2.setWindowProperty(self.config.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    else:
                        cv2.setWindowProperty(self.config.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    
                    cv2.imshow(self.config.WINDOW_NAME, display_frame)
                
                # Gestione input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # Q o ESC
                    break
                elif key == ord('r'):  # Reset filtri
                    if self.reset_server_filters():
                        self.logger.info("Filtri resettati")
                elif key == ord('s'):  # Screenshot
                    screenshot_file = self.save_screenshot()
                elif key == ord('d'):  # Toggle debug
                    self.show_debug = not self.show_debug
                    self.logger.info(f"Debug info: {'ON' if self.show_debug else 'OFF'}")
                elif key == ord('f'):  # Toggle fullscreen
                    self.fullscreen = not self.fullscreen
                elif key == ord('p'):  # Toggle posizione 3D
                    self.show_position_3d = not self.show_position_3d
                    self.logger.info(f"Posizione 3D: {'ON' if self.show_position_3d else 'OFF'}")
                elif key == ord('o'):  # Toggle orientamento
                    self.show_orientation = not self.show_orientation
                    self.logger.info(f"Orientamento: {'ON' if self.show_orientation else 'OFF'}")
                elif key == ord('1'):  # Vista standard
                    self.view_mode = 1
                    self.logger.info("Vista: Standard")
                elif key == ord('2'):  # Vista posizione dettagliata
                    self.view_mode = 2
                    self.logger.info("Vista: Posizione 3D dettagliata")
                elif key == ord('3'):  # Vista orientamento dettagliato
                    self.view_mode = 3
                    self.logger.info("Vista: Orientamento dettagliato")
                
                time.sleep(0.01)
        
        except KeyboardInterrupt:
            self.logger.info("Interruzione da tastiera")
        
        finally:
            self.is_running = False
            cv2.destroyAllWindows()
            self.logger.info("Client terminato")
        
        return True

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Client per April Tag Distance Detection Web Service')
    config = ClientConfig()
    parser.add_argument('--server', default=config.DEFAULT_SERVER_IP,
                       help=f'Indirizzo IP del server (default: {config.DEFAULT_SERVER_IP})')
    parser.add_argument('--port', type=int, default=config.DEFAULT_SERVER_PORT,
                       help=f'Porta del server (default: {config.DEFAULT_SERVER_PORT})')
    
    args = parser.parse_args()
    
    # Crea e avvia client
    client = AprilTagClient(args.server, args.port)
    
    print(f"Connessione a {args.server}:{args.port}")
    print("Comandi disponibili:")
    print("  Q: Esci")
    print("  R: Reset filtri server") 
    print("  S: Screenshot")
    print("  D: Toggle informazioni debug")
    print("  F: Toggle fullscreen")
    print("  P: Toggle visualizzazione posizione 3D")
    print("  O: Toggle visualizzazione orientamento")
    print("  1: Vista standard")
    print("  2: Vista dettagliata posizione 3D")
    print("  3: Vista dettagliata orientamento")
    
    success = client.run()
    
    if success:
        print("Client terminato correttamente")
    else:
        print("Errore durante l'esecuzione del client")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())