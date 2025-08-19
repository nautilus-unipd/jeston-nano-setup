#!/usr/bin/env python3
"""
Client per visualizzazione Web Service April Tag Distance Detection
Versione: Client 1.0

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

# =============================================================================
# CONFIGURAZIONE
# =============================================================================

class ClientConfig:
    DEFAULT_SERVER_IP = "192.168.55.1"
    DEFAULT_SERVER_PORT = 10000
    
    # Parametri visualizzazione
    WINDOW_NAME = "April Tag Distance Detection - Client"
    UPDATE_INTERVAL = 0.033  # ~30 FPS
    
    # Colori per visualizzazione (BGR)
    COLOR_MARKER_BOX = (0, 255, 0)      # Verde per il riquadro
    COLOR_MARKER_CENTER = (0, 0, 255)   # Rosso per il centro
    COLOR_TEXT_DISTANCE = (0, 255, 255) # Giallo per la distanza
    COLOR_TEXT_INFO = (255, 255, 255)   # Bianco per info generali
    COLOR_TEXT_DEBUG = (128, 128, 255)  # Azzurro per debug
    COLOR_SEPARATOR = (255, 255, 255)   # Bianco per separatore
    
    # Parametri testo
    FONT_SCALE_DISTANCE = 0.8
    FONT_SCALE_INFO = 0.6
    FONT_SCALE_DEBUG = 0.5
    FONT_THICKNESS = 2

# =============================================================================
# CLIENT VISUALIZZATORE
# =============================================================================

class AprilTagClient:
    def __init__(self, server_ip, server_port):
        self.server_ip = server_ip
        self.server_port = server_port
        self.base_url = f"http://{server_ip}:{server_port}"
        
        # Variabili per i dati
        self.current_frame = None
        self.detection_data = {}
        self.server_status = {}
        self.is_running = False
        
        # Lock per thread safety
        self.data_lock = threading.Lock()
        
        # Parametri visualizzazione
        self.show_debug = False
        self.fullscreen = False
        
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
    
    def draw_marker_info(self, frame, tag_data):
        """
        Disegna informazioni del marcatore sul frame
        """
        tag_id = tag_data['id']
        distance = tag_data['distance_smooth']
        distance_raw = tag_data['distance_raw']
        corners_left = np.array(tag_data['corners_left'], dtype=np.float32)
        center_left = tag_data['center_left']
        
        # Disegna riquadro del marcatore
        corners_int = corners_left.astype(int)
        cv2.polylines(frame, [corners_int], True, ClientConfig.COLOR_MARKER_BOX, 3)
        
        # Disegna centro
        center_int = (int(center_left[0]), int(center_left[1]))
        cv2.circle(frame, center_int, 5, ClientConfig.COLOR_MARKER_CENTER, -1)
        
        # Disegna ID e distanza principale
        text_distance = f"ID {tag_id}: {distance:.1f}cm"
        text_size = cv2.getTextSize(text_distance, cv2.FONT_HERSHEY_SIMPLEX, 
                                   ClientConfig.FONT_SCALE_DISTANCE, ClientConfig.FONT_THICKNESS)[0]
        
        # Posiziona testo sopra il marcatore
        text_x = max(10, center_int[0] - text_size[0] // 2)
        text_y = max(30, center_int[1] - 20)
        
        # Sfondo per il testo
        cv2.rectangle(frame, 
                     (text_x - 5, text_y - text_size[1] - 5),
                     (text_x + text_size[0] + 5, text_y + 5),
                     (0, 0, 0), -1)
        
        cv2.putText(frame, text_distance, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, ClientConfig.FONT_SCALE_DISTANCE,
                   ClientConfig.COLOR_TEXT_DISTANCE, ClientConfig.FONT_THICKNESS)
        
        # Informazioni aggiuntive se debug è attivo
        if self.show_debug:
            debug_info = [
                f"Raw: {distance_raw:.1f}cm",
                f"Disp: {tag_data['disparity']:.1f}px",
                f"Center: ({center_left[0]:.0f}, {center_left[1]:.0f})"
            ]
            
            for i, info in enumerate(debug_info):
                debug_y = text_y + 25 + (i * 20)
                cv2.putText(frame, info, (text_x, debug_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, ClientConfig.FONT_SCALE_DEBUG,
                           ClientConfig.COLOR_TEXT_DEBUG, 1)
    
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
            
        info_lines.append(f"Tags rilevati: {tags_count}")
        info_lines.append(f"FPS: {self.stats['fps']:.1f}")
        info_lines.append(f"Frames: {self.stats['frames_received']}")
        
        # Disegna sfondo per info generali
        info_height = len(info_lines) * 25 + 10
        cv2.rectangle(frame, (10, 10), (250, info_height), (0, 0, 0, 128), -1)
        
        for i, line in enumerate(info_lines):
            y = 30 + i * 25
            cv2.putText(frame, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX,
                       ClientConfig.FONT_SCALE_INFO, ClientConfig.COLOR_TEXT_INFO, 1)
        
        # Informazioni di calibrazione in basso a sinistra
        with self.data_lock:
            calib_info = self.detection_data.get('calibration_info', {})
            
        if calib_info:
            calib_lines = [
                f"RMS: {calib_info.get('rms_error', 0):.2f}px",
                f"Baseline: {calib_info.get('baseline', 0):.1f}cm"
            ]
            
            calib_height = len(calib_lines) * 20 + 10
            calib_y_start = height - calib_height - 10
            
            cv2.rectangle(frame, (10, calib_y_start), (200, height - 10), (0, 0, 0, 128), -1)
            
            for i, line in enumerate(calib_lines):
                y = calib_y_start + 20 + i * 20
                cv2.putText(frame, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX,
                           ClientConfig.FONT_SCALE_DEBUG, ClientConfig.COLOR_TEXT_INFO, 1)
        
        # Comandi in basso a destra
        commands = ["Q:Esci", "R:Reset", "S:Screenshot", "D:Debug", "F:Fullscreen"]
        command_text = " | ".join(commands)
        
        text_size = cv2.getTextSize(command_text, cv2.FONT_HERSHEY_SIMPLEX,
                                   ClientConfig.FONT_SCALE_DEBUG, 1)[0]
        
        cmd_x = width - text_size[0] - 15
        cmd_y = height - 15
        
        cv2.rectangle(frame, (cmd_x - 5, cmd_y - 20), (width - 10, height - 5), (0, 0, 0, 128), -1)
        cv2.putText(frame, command_text, (cmd_x, cmd_y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                   ClientConfig.FONT_SCALE_DEBUG, ClientConfig.COLOR_TEXT_INFO, 1)
        
        # Separatore stereo (se il frame è stereo)
        if width > height:  # Probabilmente stereo side-by-side
            separator_x = width // 2
            cv2.line(frame, (separator_x, 0), (separator_x, height), ClientConfig.COLOR_SEPARATOR, 2)
            
            # Etichette sinistra/destra
            cv2.putText(frame, "SINISTRA", (50, height - 50), cv2.FONT_HERSHEY_SIMPLEX,
                       0.8, ClientConfig.COLOR_SEPARATOR, 2)
            cv2.putText(frame, "DESTRA", (separator_x + 50, height - 50), cv2.FONT_HERSHEY_SIMPLEX,
                       0.8, ClientConfig.COLOR_SEPARATOR, 2)
    
    def process_frame(self):
        """
        Elabora il frame corrente con le informazioni di rilevamento
        """
        with self.data_lock:
            if self.current_frame is None:
                return None
            
            frame = self.current_frame.copy()
            tags_data = self.detection_data.get('tags', {})
        
        # Disegna informazioni per ogni tag rilevato
        for tag_id, tag_data in tags_data.items():
            self.draw_marker_info(frame, tag_data)
        
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
            time.sleep(ClientConfig.UPDATE_INTERVAL)
    
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
        
        # Aspetta il primo frame per determinare le dimensioni della finestra
        self.logger.info("Attendendo primo frame per determinare dimensioni finestra...")
        while self.is_running and self.current_frame is None:
            time.sleep(0.1)
        
        if self.current_frame is None:
            self.logger.error("Nessun frame ricevuto")
            return False
        
        # Crea finestra con dimensioni fisse basate sul frame ricevuto
        frame_height, frame_width = self.current_frame.shape[:2]
        cv2.namedWindow(ClientConfig.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(ClientConfig.WINDOW_NAME, frame_width, frame_height)
        cv2.setWindowProperty(ClientConfig.WINDOW_NAME, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_FREERATIO)
        
        # Rende la finestra non ridimensionabile
        cv2.setWindowProperty(ClientConfig.WINDOW_NAME, cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_AUTOSIZE)
        
        self.logger.info(f"Finestra creata con dimensioni: {frame_width}x{frame_height}")
        
        self.logger.info("Client avviato - premere Q per uscire")
        
        try:
            while self.is_running:
                # Elabora frame corrente
                display_frame = self.process_frame()
                
                if display_frame is not None:
                    # Mostra frame
                    if self.fullscreen:
                        cv2.setWindowProperty(ClientConfig.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    else:
                        cv2.setWindowProperty(ClientConfig.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    
                    cv2.imshow(ClientConfig.WINDOW_NAME, display_frame)
                
                # Gestione input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # Q o ESC
                    break
                elif key == ord('r'):  # Reset filtri
                    if self.reset_server_filters():
                        self.logger.info("Filtri resettati")
                elif key == ord('s'):  # Screenshot
                    screenshot_file = self.save_screenshot()
                    if screenshot_file:
                        self.logger.info(f"Screenshot: {screenshot_file}")
                elif key == ord('d'):  # Toggle debug
                    self.show_debug = not self.show_debug
                    self.logger.info(f"Debug info: {'ON' if self.show_debug else 'OFF'}")
                elif key == ord('f'):  # Toggle fullscreen
                    self.fullscreen = not self.fullscreen
                    self.logger.info(f"Fullscreen: {'ON' if self.fullscreen else 'OFF'}")
                
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
    parser.add_argument('--server', default=ClientConfig.DEFAULT_SERVER_IP,
                       help=f'Indirizzo IP del server (default: {ClientConfig.DEFAULT_SERVER_IP})')
    parser.add_argument('--port', type=int, default=ClientConfig.DEFAULT_SERVER_PORT,
                       help=f'Porta del server (default: {ClientConfig.DEFAULT_SERVER_PORT})')
    
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
    print()
    
    success = client.run()
    
    if success:
        print("Client terminato correttamente")
    else:
        print("Errore durante l'esecuzione del client")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())