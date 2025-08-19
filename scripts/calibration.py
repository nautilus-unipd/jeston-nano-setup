#!/usr/bin/env python3
"""
Sistema di Calibrazione Stereo Semplificato
Versione: 1.0

Requisiti:
pip install opencv-python numpy

Uso:
python stereo_calibrator.py --calibrate           # Calibra da stream video
python stereo_calibrator.py --upload             # Carica calibrazione su Jetson
python stereo_calibrator.py --help               # Mostra aiuto
"""

import cv2
import numpy as np
import json
import os
import sys
import argparse
import time
from datetime import datetime
import pickle
import socket
import subprocess

# =============================================================================
# CONFIGURAZIONE
# =============================================================================

class Config:
    # Stream URL
    STREAM_URL = "http://10.70.64.50:8081/back/stereo/feed"
    
    # Risoluzione telecamere
    CAMERA_LEFT_WIDTH = 640
    CAMERA_LEFT_HEIGHT = 480
    CAMERA_RIGHT_WIDTH = 640
    CAMERA_RIGHT_HEIGHT = 480
    
    # Parametri scacchiera
    CHESSBOARD_SIZE = (10, 7)  # Numero di angoli interni (width, height)
    SQUARE_SIZE = 3.7  # cm - Misurare precisamente!
    
    # File di salvataggio
    CALIBRATION_FILE = "stereo_calibration_data.pkl"
    LOG_FILE = "calibration_log.txt"
    
    # Parametri di qualità
    MIN_CALIBRATION_IMAGES = 25
    MAX_REPROJECTION_ERROR = 1.0
    MIN_DISPARITY = 1.0
    
    # Server Jetson
    DEFAULT_SERVER_IP = "192.168.55.1"
    DEFAULT_SERVER_PORT = 10000
    JETSON_PATH = "/home/nautilus/jeston-nano-setup/server/"

# =============================================================================
# LOGGING
# =============================================================================

class Logger:
    def __init__(self, filename=Config.LOG_FILE):
        self.filename = filename
    
    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        
        with open(self.filename, 'a', encoding='utf-8') as f:
            f.write(log_msg + "\n")

logger = Logger()

# =============================================================================
# UTILITÀ
# =============================================================================

def split_stereo_frame(frame, config=Config):
    """Divide il frame stereo in frame sinistro e destro"""
    frame_h, frame_w = frame.shape[:2]
    expected_width = config.CAMERA_LEFT_WIDTH + config.CAMERA_RIGHT_WIDTH
    
    if frame_w != expected_width:
        logger.log(f"ATTENZIONE: Larghezza frame ({frame_w}) diversa da prevista ({expected_width})")
        half_w = frame_w // 2
        frame_l = frame[:, :half_w]
        frame_r = frame[:, half_w:]
    else:
        frame_l = frame[:, :config.CAMERA_LEFT_WIDTH]
        frame_r = frame[:, config.CAMERA_LEFT_WIDTH:config.CAMERA_LEFT_WIDTH + config.CAMERA_RIGHT_WIDTH]
    
    return frame_l, frame_r

def validate_chessboard_image(gray_l, gray_r, chessboard_size):
    """Valida la qualità dell'immagine della scacchiera"""
    ret_l, corners_l = cv2.findChessboardCorners(gray_l, chessboard_size, 
                                                cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                cv2.CALIB_CB_NORMALIZE_IMAGE +
                                                cv2.CALIB_CB_FILTER_QUADS)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, chessboard_size,
                                                cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                cv2.CALIB_CB_NORMALIZE_IMAGE +
                                                cv2.CALIB_CB_FILTER_QUADS)
    
    if not (ret_l and ret_r):
        return False, "Scacchiera non trovata"
    
    if len(corners_l) != chessboard_size[0] * chessboard_size[1]:
        return False, "Numero angoli non corretto"
    
    # Calcola area coperta
    corners_l_flat = corners_l.reshape(-1, 2)
    min_x_l, min_y_l = corners_l_flat.min(axis=0)
    max_x_l, max_y_l = corners_l_flat.max(axis=0)
    area_l = (max_x_l - min_x_l) * (max_y_l - min_y_l)
    
    min_area = 5500 * ((Config.CAMERA_LEFT_WIDTH * Config.CAMERA_LEFT_HEIGHT) / (640 * 480))
    
    if area_l < min_area:
        return False, f"Scacchiera troppo piccola"
    
    # Controlla bordi
    img_h, img_w = gray_l.shape
    border_margin = max(30, int(30 * (img_w / 640)))
    
    if (min_x_l < border_margin or min_y_l < border_margin or 
        max_x_l > img_w - border_margin or max_y_l > img_h - border_margin):
        return False, "Troppo vicina ai bordi"
    
    return True, "OK"

# =============================================================================
# CALIBRATORE STEREO
# =============================================================================

class StereoCalibrator:
    def __init__(self, config=Config):
        self.config = config
        self.logger = logger
        
        # Criteri di ottimizzazione
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7)
        self.criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-6)
        
        # Array per i punti
        self.objpoints = []
        self.imgpoints_l = []
        self.imgpoints_r = []
        
        # Punti 3D della scacchiera
        self.objp = np.zeros((config.CHESSBOARD_SIZE[0] * config.CHESSBOARD_SIZE[1], 3), np.float32)
        self.objp[:,:2] = np.mgrid[0:config.CHESSBOARD_SIZE[0], 0:config.CHESSBOARD_SIZE[1]].T.reshape(-1,2)
        self.objp *= config.SQUARE_SIZE
        
        # Parametri di calibrazione
        self.camera_matrix_l = None
        self.camera_matrix_r = None
        self.dist_coeffs_l = None
        self.dist_coeffs_r = None
        self.R = None
        self.T = None
        self.E = None
        self.F = None
        self.calibration_quality = {}
        
        # Mappe di rettificazione
        self.map1_l = None
        self.map2_l = None
        self.map1_r = None
        self.map2_r = None
        self.Q = None
        
        self.logger.log("Calibratore stereo inizializzato")
    
    def connect_camera(self):
        """Connessione alla telecamera"""
        for attempt in range(3):
            try:
                cap = cv2.VideoCapture(self.config.STREAM_URL)
                if cap.isOpened():
                    self.logger.log("Connesso alla telecamera")
                    return cap
                else:
                    time.sleep(2)
            except Exception as e:
                self.logger.log(f"Errore connessione: {e}")
        
        self.logger.log("ERRORE: Impossibile connettersi alla telecamera")
        return None
    
    def capture_calibration_images(self, target_images=15):
        """Cattura immagini di calibrazione"""
        cap = self.connect_camera()
        if not cap:
            return False
        
        self.logger.log(f"Inizio cattura {target_images} immagini")
        self.logger.log("Comandi: SPAZIO=cattura, ESC=esci")
        
        captured = 0
        rejected = 0
        
        while captured < target_images:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame_l, frame_r = split_stereo_frame(frame, self.config)
            gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
            
            # Valida immagine
            is_valid, message = validate_chessboard_image(gray_l, gray_r, self.config.CHESSBOARD_SIZE)
            
            # Trova angoli
            flags = (cv2.CALIB_CB_ADAPTIVE_THRESH + 
                    cv2.CALIB_CB_NORMALIZE_IMAGE + 
                    cv2.CALIB_CB_FILTER_QUADS)
            
            ret_l, corners_l = cv2.findChessboardCorners(gray_l, self.config.CHESSBOARD_SIZE, flags)
            ret_r, corners_r = cv2.findChessboardCorners(gray_r, self.config.CHESSBOARD_SIZE, flags)
            
            display_l = frame_l.copy()
            display_r = frame_r.copy()
            
            if ret_l and ret_r and is_valid:
                # Disegna angoli colorati
                cv2.drawChessboardCorners(display_l, self.config.CHESSBOARD_SIZE, corners_l, ret_l)
                cv2.drawChessboardCorners(display_r, self.config.CHESSBOARD_SIZE, corners_r, ret_r)
                status_color = (0, 255, 0)
                status_text = f"PRONTO - {message}"
            else:
                status_color = (0, 0, 255)
                status_text = f"NON VALIDO - {message}"
            
            # Prepara display
            combined = np.hstack([display_l, display_r])
            
            # Aggiungi informazioni
            cv2.putText(combined, f"Catturate: {captured}/{target_images} | Rifiutate: {rejected}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(combined, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            cv2.putText(combined, "SPAZIO=Cattura | ESC=Esci", 
                       (10, combined.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Separatore
            separator_x = self.config.CAMERA_LEFT_WIDTH
            cv2.line(combined, (separator_x, 0), (separator_x, combined.shape[0]), (255, 255, 255), 2)
            cv2.putText(combined, "SINISTRA", (50, combined.shape[0]-50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(combined, "DESTRA", (separator_x + 50, combined.shape[0]-50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            cv2.imshow('Calibrazione Stereo', combined)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Cattura
                if ret_l and ret_r and is_valid:
                    # Refina angoli
                    corners_l = cv2.cornerSubPix(gray_l, corners_l, (11,11), (-1,-1), self.criteria)
                    corners_r = cv2.cornerSubPix(gray_r, corners_r, (11,11), (-1,-1), self.criteria)
                    
                    # Salva punti
                    self.objpoints.append(self.objp)
                    self.imgpoints_l.append(corners_l)
                    self.imgpoints_r.append(corners_r)
                    
                    captured += 1
                    self.logger.log(f"Immagine {captured} catturata")
                    
                else:
                    rejected += 1
                    self.logger.log(f"Immagine rifiutata: {message}")
            
            elif key == 27:  # ESC
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        success = captured >= self.config.MIN_CALIBRATION_IMAGES
        self.logger.log(f"Cattura completata: {captured} immagini valide")
        return success
    
    def calibrate_individual_cameras(self, img_size):
        """Calibra le singole telecamere"""
        self.logger.log("Calibrazione telecamere individuali...")
        
        ret_l, self.camera_matrix_l, self.dist_coeffs_l, _, _ = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_l, img_size, None, None, criteria=self.criteria
        )
        
        ret_r, self.camera_matrix_r, self.dist_coeffs_r, _, _ = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_r, img_size, None, None, criteria=self.criteria
        )
        
        self.calibration_quality['left_rms'] = ret_l
        self.calibration_quality['right_rms'] = ret_r
        
        self.logger.log(f"RMS sinistra: {ret_l:.3f}, destra: {ret_r:.3f}")
        return ret_l < 1.0 and ret_r < 1.0
    
    def calibrate_stereo_system(self, img_size):
        """Calibra il sistema stereo"""
        self.logger.log("Calibrazione sistema stereo...")
        
        flags = (cv2.CALIB_FIX_INTRINSIC +
                cv2.CALIB_RATIONAL_MODEL +
                cv2.CALIB_FIX_PRINCIPAL_POINT)
        
        ret, _, _, _, _, self.R, self.T, self.E, self.F = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_l, self.imgpoints_r,
            self.camera_matrix_l, self.dist_coeffs_l,
            self.camera_matrix_r, self.dist_coeffs_r,
            img_size, criteria=self.criteria_stereo, flags=flags
        )
        
        self.calibration_quality['stereo_rms'] = ret
        baseline = np.linalg.norm(self.T)
        self.calibration_quality['baseline'] = baseline
        
        self.logger.log(f"RMS stereo: {ret:.3f}, Baseline: {baseline:.2f}cm")
        
        # Rettificazione stereo
        self.R1, self.R2, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(
            self.camera_matrix_l, self.dist_coeffs_l,
            self.camera_matrix_r, self.dist_coeffs_r,
            img_size, self.R, self.T, flags=cv2.CALIB_ZERO_DISPARITY
        )
        
        # Calcola mappe di rettificazione
        self.map1_l, self.map2_l = cv2.initUndistortRectifyMap(
            self.camera_matrix_l, self.dist_coeffs_l, self.R1, self.P1, img_size, cv2.CV_16SC2
        )
        
        self.map1_r, self.map2_r = cv2.initUndistortRectifyMap(
            self.camera_matrix_r, self.dist_coeffs_r, self.R2, self.P2, img_size, cv2.CV_16SC2
        )
        
        return ret < self.config.MAX_REPROJECTION_ERROR
    
    def save_calibration(self):
        """Salva parametri di calibrazione"""
        calibration_data = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'chessboard_size': self.config.CHESSBOARD_SIZE,
                'square_size': self.config.SQUARE_SIZE,
                'camera_left_width': self.config.CAMERA_LEFT_WIDTH,
                'camera_left_height': self.config.CAMERA_LEFT_HEIGHT,
                'camera_right_width': self.config.CAMERA_RIGHT_WIDTH,
                'camera_right_height': self.config.CAMERA_RIGHT_HEIGHT
            },
            'quality': self.calibration_quality,
            'camera_matrix_l': self.camera_matrix_l,
            'camera_matrix_r': self.camera_matrix_r,
            'dist_coeffs_l': self.dist_coeffs_l,
            'dist_coeffs_r': self.dist_coeffs_r,
            'R': self.R,
            'T': self.T,
            'E': self.E,
            'F': self.F,
            'R1': self.R1,
            'R2': self.R2,
            'P1': self.P1,
            'P2': self.P2,
            'Q': self.Q,
            'map1_l': self.map1_l,
            'map2_l': self.map2_l,
            'map1_r': self.map1_r,
            'map2_r': self.map2_r
        }
        
        with open(self.config.CALIBRATION_FILE, 'wb') as f:
            pickle.dump(calibration_data, f)
        
        self.logger.log(f"Calibrazione salvata in {self.config.CALIBRATION_FILE}")
    
    def run_calibration(self):
        """Esegue il processo completo di calibrazione"""
        self.logger.log("INIZIO CALIBRAZIONE")
        
        # Cattura immagini
        if not self.capture_calibration_images(self.config.MIN_CALIBRATION_IMAGES):
            self.logger.log("ERRORE: Immagini insufficienti")
            return False
        
        # Usa dimensioni configurate
        img_size = (self.config.CAMERA_LEFT_WIDTH, self.config.CAMERA_LEFT_HEIGHT)
        
        # Calibra telecamere individuali
        if not self.calibrate_individual_cameras(img_size):
            self.logger.log("ERRORE: Calibrazione telecamere fallita")
            return False
        
        # Calibra sistema stereo
        if not self.calibrate_stereo_system(img_size):
            self.logger.log("ERRORE: Calibrazione stereo fallita")
            return False
        
        # Salva risultati
        self.save_calibration()
        
        self.logger.log("CALIBRAZIONE COMPLETATA!")
        return True

# =============================================================================
# UPLOAD SU JETSON
# =============================================================================

def upload_calibration_to_jetson(server_ip=Config.DEFAULT_SERVER_IP, jetson_path=Config.JETSON_PATH):
    """Carica il file di calibrazione sulla Jetson Nano via SCP"""
    
    if not os.path.exists(Config.CALIBRATION_FILE):
        logger.log(f"ERRORE: File di calibrazione {Config.CALIBRATION_FILE} non trovato")
        return False
    
    try:
        # Costruisci comando SCP
        remote_path = f"nautilus@{server_ip}:{jetson_path}{Config.CALIBRATION_FILE}"
        cmd = ["scp", Config.CALIBRATION_FILE, remote_path]
        
        logger.log(f"Caricamento su {remote_path}...")
        
        # Esegui comando SCP
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            logger.log("File caricato con successo sulla Jetson!")
            return True
        else:
            logger.log(f"ERRORE SCP: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.log("ERRORE: Timeout durante upload")
        return False
    except Exception as e:
        logger.log(f"ERRORE: {e}")
        return False

# =============================================================================
# MAIN
# =============================================================================

def print_help():
    """Stampa istruzioni d'uso"""
    print("\n" + "="*60)
    print("SISTEMA DI CALIBRAZIONE STEREO")
    print("="*60)
    print()
    print("COMANDI:")
    print("  --calibrate    Esegui calibrazione stereo da stream video")
    print("  --upload       Carica calibrazione su Jetson Nano")
    print("  --help         Mostra questo aiuto")
    print()
    print("CONFIGURAZIONE ATTUALE:")
    print(f"  Stream URL: {Config.STREAM_URL}")
    print(f"  Risoluzione L: {Config.CAMERA_LEFT_WIDTH}x{Config.CAMERA_LEFT_HEIGHT}")
    print(f"  Risoluzione R: {Config.CAMERA_RIGHT_WIDTH}x{Config.CAMERA_RIGHT_HEIGHT}")
    print(f"  Scacchiera: {Config.CHESSBOARD_SIZE} angoli interni")
    print(f"  Quadrato: {Config.SQUARE_SIZE}cm")
    print(f"  Server Jetson: {Config.DEFAULT_SERVER_IP}")
    print()
    print("PROCESSO:")
    print("  1. python stereo_calibrator.py --calibrate")
    print("  2. python stereo_calibrator.py --upload")
    print()
    print("CALIBRAZIONE:")
    print("  - Muovi la scacchiera in varie posizioni")
    print("  - Inclina in vari angoli (±15°, ±30°)")
    print("  - Assicurati che sia ben illuminata e piatta")
    print("  - La scacchiera deve riempire almeno 70% dell'immagine")
    print("  - SPAZIO per catturare, ESC per uscire")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Sistema di Calibrazione Stereo', add_help=False)
    parser.add_argument('--calibrate', action='store_true', help='Esegui calibrazione stereo')
    parser.add_argument('--upload', action='store_true', help='Carica calibrazione su Jetson')
    parser.add_argument('--help', action='store_true', help='Mostra aiuto')
    
    args = parser.parse_args()
    
    if args.help or not any(vars(args).values()):
        print_help()
        return
    
    if args.calibrate:
        logger.log("AVVIO CALIBRAZIONE STEREO")
        calibrator = StereoCalibrator()
        
        print(f"\nCONFIGURAZIONE:")
        print(f"Scacchiera: {Config.CHESSBOARD_SIZE} angoli interni")
        print(f"Quadrato: {Config.SQUARE_SIZE}cm")
        print(f"Risoluzione L: {Config.CAMERA_LEFT_WIDTH}x{Config.CAMERA_LEFT_HEIGHT}")
        print(f"Risoluzione R: {Config.CAMERA_RIGHT_WIDTH}x{Config.CAMERA_RIGHT_HEIGHT}")
        print(f"Stream: {Config.STREAM_URL}")
        
        response = input("\nConfermi configurazione? (y/n): ")
        if response.lower() != 'y':
            print("Modifica parametri nella classe Config e riprova.")
            return
        
        success = calibrator.run_calibration()
        
        if success:
            print("\n✅ CALIBRAZIONE COMPLETATA!")
            print("Usa: python stereo_calibrator.py --upload")
        else:
            print("\n❌ CALIBRAZIONE FALLITA!")
    
    elif args.upload:
        logger.log("CARICAMENTO SU JETSON")
        
        print(f"\nCaricamento file su Jetson:")
        print(f"Server: {Config.DEFAULT_SERVER_IP}")
        print(f"Percorso: {Config.JETSON_PATH}")
        
        response = input("\nConfermi upload? (y/n): ")
        if response.lower() != 'y':
            print("Upload annullato.")
            return
        
        success = upload_calibration_to_jetson()
        
        if success:
            print("\n✅ FILE CARICATO CON SUCCESSO!")
        else:
            print("\n❌ ERRORE DURANTE UPLOAD!")
            print("Verifica:")
            print("- Connessione di rete alla Jetson")
            print("- Credenziali SSH configurate")
            print("- Percorso di destinazione esistente")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgramma interrotto dall'utente")
    except Exception as e:
        logger.log(f"ERRORE CRITICO: {e}")
        print(f"Errore critico: {e}")