#!/usr/bin/env python3
"""
Web Service per Rilevamento Distanza April Tag con Telecamera Stereo
Versione: Web Service 1.2 - Extended with Right Camera Processing

Requisiti:
pip install opencv-python numpy

Uso:
python stereo_apriltag_webservice.py

Endpoints:
- /video_feed : Stream video senza elaborazione
- /detection_data : Dati JSON dei marcatori rilevati
- /status : Stato del servizio
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import json
import os
import time
import pickle
import threading
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
import socketserver
from urllib.parse import urlparse, parse_qs
import logging
from config_loader import ConfigLoader

# =============================================================================
# CONFIGURAZIONE GLOBALE
# =============================================================================

class Config(ConfigLoader):
    """Configurazione per il web server ArUco Marker"""
    
    def __init__(self):
        # Parametri web service
        self.WEB_SERVICE_PORT = self.get_nested_value("network", "default_server_port")
        self.WEB_SERVICE_HOST = self.get_nested_value("network", "web_service_host")
        
        # Parametri stream video
        self.STREAM_URL = self.get_nested_value("camera", "stream_url")
        
        # Configurazione risoluzione telecamere
        self.CAMERA_LEFT_WIDTH = self.get_nested_value("camera", "left", "width")
        self.CAMERA_LEFT_HEIGHT = self.get_nested_value("camera", "left", "height")
        self.CAMERA_RIGHT_WIDTH = self.get_nested_value("camera", "right", "width")
        self.CAMERA_RIGHT_HEIGHT = self.get_nested_value("camera", "right", "height")
        
        # Parametri April Tag
        dict_name = self.get_nested_value("aruco", "dict_type")
        self.ARUCO_DICT = getattr(aruco, dict_name)
        self.TAG_SIZE_REAL = self.get_nested_value("aruco", "tag_size_real")
        
        # File di calibrazione
        self.CALIBRATION_FILE = self.get_nested_value("calibration", "files", "calibration_file")
        
        # Parametri di qualita
        self.MIN_DISPARITY = self.get_nested_value("calibration", "quality", "min_disparity")

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# UTILITA
# =============================================================================

def split_stereo_frame(frame, config=Config()):
    """
    Divide il frame stereo in frame sinistro e destro basandosi sulla configurazione
    """
    frame_h, frame_w = frame.shape[:2]
    expected_width = config.CAMERA_LEFT_WIDTH + config.CAMERA_RIGHT_WIDTH
    
    if frame_w != expected_width:
        logger.warning(f"Larghezza frame ({frame_w}) diversa da prevista ({expected_width})")
        half_w = frame_w // 2
        frame_l = frame[:, :half_w]
        frame_r = frame[:, half_w:]
    else:
        frame_l = frame[:, :config.CAMERA_LEFT_WIDTH]
        frame_r = frame[:, config.CAMERA_LEFT_WIDTH:config.CAMERA_LEFT_WIDTH + config.CAMERA_RIGHT_WIDTH]
    
    return frame_l, frame_r

def rotation_matrix_to_euler_angles(R):
    """
    Converte matrice di rotazione in angoli di Eulero (roll, pitch, yaw) in gradi
    """
    # Calcola gli angoli di Eulero dalla matrice di rotazione
    sy = np.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    singular = sy < 1e-6
    
    if not singular:
        x = np.arctan2(R[2,1], R[2,2])  # roll
        y = np.arctan2(-R[2,0], sy)     # pitch
        z = np.arctan2(R[1,0], R[0,0])  # yaw
    else:
        x = np.arctan2(-R[1,2], R[1,1]) # roll
        y = np.arctan2(-R[2,0], sy)     # pitch
        z = 0                           # yaw
    
    # Converti in gradi
    roll = np.degrees(x)
    pitch = np.degrees(y)
    yaw = np.degrees(z)
    
    return roll, pitch, yaw

# =============================================================================
# CALIBRATORE STEREO
# =============================================================================

class StereoCalibrator:
    def __init__(self, config=Config()):
        self.config = config
        
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
        
        logger.info("Calibratore stereo inizializzato")
    
    def load_calibration(self):
        """
        Carica parametri di calibrazione esistenti
        """
        if not os.path.exists(self.config.CALIBRATION_FILE):
            return False
        
        try:
            with open(self.config.CALIBRATION_FILE, 'rb') as f:
                data = pickle.load(f)
            
            self.camera_matrix_l = data['camera_matrix_l']
            self.camera_matrix_r = data['camera_matrix_r']
            self.dist_coeffs_l = data['dist_coeffs_l']
            self.dist_coeffs_r = data['dist_coeffs_r']
            self.R = data['R']
            self.T = data['T']
            self.E = data['E']
            self.F = data['F']
            self.R1 = data['R1']
            self.R2 = data['R2']
            self.P1 = data['P1']
            self.P2 = data['P2']
            self.Q = data['Q']
            self.map1_l = data['map1_l']
            self.map2_l = data['map2_l']
            self.map1_r = data['map1_r']
            self.map2_r = data['map2_r']
            self.calibration_quality = data['quality']
            
            logger.info(f"Calibrazione caricata da {self.config.CALIBRATION_FILE}")
            logger.info(f"RMS Stereo: {self.calibration_quality['stereo_rms']:.3f}")
            logger.info(f"Baseline: {self.calibration_quality['baseline']:.2f} cm")
            
            return True
            
        except Exception as e:
            logger.error(f"Errore caricamento calibrazione: {e}")
            return False

# =============================================================================
# RILEVATORE DISTANZE
# =============================================================================

class AprilTagDistanceDetector:
    def __init__(self, config=Config()):
        self.config = config
        self.calibrator = StereoCalibrator(config)
        
        # Setup ArUco - Gestione compatibilità versioni OpenCV
        self.aruco_dict = aruco.getPredefinedDictionary(config.ARUCO_DICT)
        
        # Prova prima la nuova API (OpenCV 4.7+)
        try:
            self.aruco_params = aruco.DetectorParameters()
            self.detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            self.use_new_api = True
            logger.info("Usando nuova API ArUco (OpenCV 4.7+)")
        except AttributeError:
            # Fallback alla vecchia API (OpenCV < 4.7)
            try:
                self.aruco_params = aruco.DetectorParameters_create()
                self.detector = None  # Non serve per la vecchia API
                self.use_new_api = False
                logger.info("Usando vecchia API ArUco (OpenCV < 4.7)")
            except AttributeError:
                # Versione ancora più vecchia
                self.aruco_params = None
                self.detector = None
                self.use_new_api = False
                logger.info("Usando API ArUco legacy (OpenCV molto vecchio)")
        
        # Filtri per smoothing - ESTESI PER CAMERA DESTRA
        self.distance_filters = {}
        self.position_filters = {}
        self.orientation_filters = {}
        # Nuovi filtri per camera destra
        self.position_filters_right = {}
        self.orientation_filters_right = {}
        
        # Variabili per il threading
        self.cap = None
        self.current_frame = None
        self.detection_data = {}
        self.is_running = False
        self.frame_lock = threading.Lock()
        self.data_lock = threading.Lock()
        
        logger.info("Rilevatore distanze inizializzato")
    
    def detect_markers(self, frame):
        """
        Rileva i marcatori compatibilmente con diverse versioni di OpenCV
        """
        if self.use_new_api and self.detector is not None:
            # Nuova API (OpenCV 4.7+)
            corners, ids, rejected = self.detector.detectMarkers(frame)
        else:
            # Vecchia API (OpenCV < 4.7)
            if self.aruco_params is not None:
                corners, ids, rejected = aruco.detectMarkers(frame, self.aruco_dict, parameters=self.aruco_params)
            else:
                # Versione molto vecchia senza parametri
                corners, ids, rejected = aruco.detectMarkers(frame, self.aruco_dict)
        
        return corners, ids, rejected
    
    def estimate_pose_single_marker(self, corners, camera_matrix, dist_coeffs, stereo_depth_cm):
        """
        Calcola la pose di un singolo marcatore con Z fissata alla profondità stereo
        """
        # Dimensione reale del marcatore in metri
        marker_size = self.config.TAG_SIZE_REAL / 100.0  # converti cm in metri
        
        # Punti 3D del marcatore (nel sistema di coordinate del marcatore)
        object_points = np.array([
            [-marker_size/2, marker_size/2, 0],
            [marker_size/2, marker_size/2, 0],
            [marker_size/2, -marker_size/2, 0],
            [-marker_size/2, -marker_size/2, 0]
        ], dtype=np.float32)
        
        # Usa solvePnP per calcolare pose
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            corners[0],
            camera_matrix,
            dist_coeffs
        )
        
        if success:
            # Converti vettore di rotazione in matrice di rotazione
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            
            # Calcola angoli di Eulero
            roll, pitch, yaw = rotation_matrix_to_euler_angles(rotation_matrix)
            
            # Converti posizione da metri a centimetri
            position_cm = tvec.flatten() * 100.0
            
            # CORREZIONE PRINCIPALE: Sostituisci Z con la profondità stereo
            corrected_position_cm = position_cm.copy()

            # correzione delle coordinate x,y,z
            scale_factor = stereo_depth_cm / position_cm[2]  # quanto è cambiata la profondità
            corrected_position_cm[0] = position_cm[0] * scale_factor  # X
            corrected_position_cm[1] = position_cm[1] * scale_factor  # Y
            corrected_position_cm[2] = stereo_depth_cm               # Z

            
            return {
                'success': True,
                'position': {
                    'x': float(corrected_position_cm[0]),
                    'y': float(corrected_position_cm[1]),
                    'z': float(corrected_position_cm[2])  # Ora corrisponde alla profondità stereo
                },
                'position_original_solvepnp': {
                    'x': float(position_cm[0]),
                    'y': float(position_cm[1]),
                    'z': float(position_cm[2])  # Z originale da solvePnP per confronto
                },
                'orientation': {
                    'roll': float(roll),
                    'pitch': float(pitch),
                    'yaw': float(yaw)
                },
                'rotation_vector': rvec.flatten().tolist(),
                'translation_vector': tvec.flatten().tolist(),
                'rotation_matrix': rotation_matrix.tolist(),
                'stereo_depth_used': float(stereo_depth_cm)
            }
        else:
            return {'success': False}
    
    def get_smoothed_values(self, tag_id, distance, position, orientation):
        """
        Applica filtri passa-basso per smooth delle misurazioni (camera sinistra)
        """
        # Filtro distanza
        if tag_id not in self.distance_filters:
            self.distance_filters[tag_id] = []
        self.distance_filters[tag_id].append(distance)
        if len(self.distance_filters[tag_id]) > 5:
            self.distance_filters[tag_id].pop(0)
        
        # Filtro posizione
        if tag_id not in self.position_filters:
            self.position_filters[tag_id] = {'x': [], 'y': [], 'z': []}
        for axis in ['x', 'y', 'z']:
            self.position_filters[tag_id][axis].append(position[axis])
            if len(self.position_filters[tag_id][axis]) > 5:
                self.position_filters[tag_id][axis].pop(0)
        
        # Filtro orientamento
        if tag_id not in self.orientation_filters:
            self.orientation_filters[tag_id] = {'roll': [], 'pitch': [], 'yaw': []}
        for angle in ['roll', 'pitch', 'yaw']:
            self.orientation_filters[tag_id][angle].append(orientation[angle])
            if len(self.orientation_filters[tag_id][angle]) > 5:
                self.orientation_filters[tag_id][angle].pop(0)
        
        # Calcola medie ponderate
        weights = np.linspace(0.5, 1.0, len(self.distance_filters[tag_id]))
        
        smoothed_distance = np.average(self.distance_filters[tag_id], weights=weights)
        
        smoothed_position = {}
        smoothed_orientation = {}
        
        for axis in ['x', 'y', 'z']:
            pos_weights = np.linspace(0.5, 1.0, len(self.position_filters[tag_id][axis]))
            smoothed_position[axis] = np.average(self.position_filters[tag_id][axis], weights=pos_weights)
        
        for angle in ['roll', 'pitch', 'yaw']:
            angle_weights = np.linspace(0.5, 1.0, len(self.orientation_filters[tag_id][angle]))
            smoothed_orientation[angle] = np.average(self.orientation_filters[tag_id][angle], weights=angle_weights)
        
        return smoothed_distance, smoothed_position, smoothed_orientation
    
    def get_smoothed_values_right(self, tag_id, position, orientation):
        """
        Applica filtri passa-basso per smooth delle misurazioni (camera destra)
        """
        # Filtro posizione camera destra
        if tag_id not in self.position_filters_right:
            self.position_filters_right[tag_id] = {'x': [], 'y': [], 'z': []}
        for axis in ['x', 'y', 'z']:
            self.position_filters_right[tag_id][axis].append(position[axis])
            if len(self.position_filters_right[tag_id][axis]) > 5:
                self.position_filters_right[tag_id][axis].pop(0)
        
        # Filtro orientamento camera destra
        if tag_id not in self.orientation_filters_right:
            self.orientation_filters_right[tag_id] = {'roll': [], 'pitch': [], 'yaw': []}
        for angle in ['roll', 'pitch', 'yaw']:
            self.orientation_filters_right[tag_id][angle].append(orientation[angle])
            if len(self.orientation_filters_right[tag_id][angle]) > 5:
                self.orientation_filters_right[tag_id][angle].pop(0)
        
        smoothed_position = {}
        smoothed_orientation = {}
        
        for axis in ['x', 'y', 'z']:
            pos_weights = np.linspace(0.5, 1.0, len(self.position_filters_right[tag_id][axis]))
            smoothed_position[axis] = np.average(self.position_filters_right[tag_id][axis], weights=pos_weights)
        
        for angle in ['roll', 'pitch', 'yaw']:
            angle_weights = np.linspace(0.5, 1.0, len(self.orientation_filters_right[tag_id][angle]))
            smoothed_orientation[angle] = np.average(self.orientation_filters_right[tag_id][angle], weights=angle_weights)
        
        return smoothed_position, smoothed_orientation
    
    def connect_camera(self):
        """Connessione alla telecamera"""
        try:
            cap = cv2.VideoCapture(self.config.STREAM_URL)
            if cap.isOpened():
                logger.info(f"Connesso alla telecamera: {self.config.STREAM_URL}")
                return cap
            else:
                logger.error("Impossibile aprire stream telecamera")
                return None
        except Exception as e:
            logger.error(f"Errore connessione: {e}")
            return None
    
    def start_detection(self):
        """
        Avvia il processo di rilevamento in background
        """
        # Carica calibrazione
        if not self.calibrator.load_calibration():
            logger.error("ERRORE: Nessuna calibrazione trovata!")
            return False
        
        # Connetti alla camera
        self.cap = self.connect_camera()
        if not self.cap:
            return False
        
        self.is_running = True
        
        # Avvia thread di rilevamento
        detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        detection_thread.start()
        
        logger.info("Sistema di rilevamento avviato")
        return True
    
    def stop_detection(self):
        """
        Ferma il processo di rilevamento
        """
        self.is_running = False
        if self.cap:
            self.cap.release()
        logger.info("Sistema di rilevamento fermato")
    
    def _detection_loop(self):
        """
        Loop principale di rilevamento (eseguito in thread separato)
        """
        frame_count = 0
        
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                frame_count += 1
                
                # Dividi frame stereo
                frame_l, frame_r = split_stereo_frame(frame, self.config)
                
                # Rettifica usando calibrazione
                rect_l = cv2.remap(frame_l, self.calibrator.map1_l, self.calibrator.map2_l, cv2.INTER_LINEAR)
                rect_r = cv2.remap(frame_r, self.calibrator.map1_r, self.calibrator.map2_r, cv2.INTER_LINEAR)
                
                # Ricomponi frame stereo rettificato per il video feed
                rectified_frame = np.hstack([rect_l, rect_r])
                
                # Aggiorna frame corrente per il video feed (ora con frame rettificati)
                with self.frame_lock:
                    self.current_frame = rectified_frame.copy()
                
                # Rileva April Tags sui frame rettificati
                corners_l, ids_l, _ = self.detect_markers(rect_l)
                corners_r, ids_r, _ = self.detect_markers(rect_r)
                
                # Calcola distanze per tag comuni
                detection_results = {}
                
                if ids_l is not None and ids_r is not None:
                    ids_l_flat = ids_l.flatten()
                    ids_r_flat = ids_r.flatten()
                    common_ids = set(ids_l_flat) & set(ids_r_flat)
                    
                    for tag_id in common_ids:
                        # Trova indici
                        idx_l = np.where(ids_l_flat == tag_id)[0][0]
                        idx_r = np.where(ids_r_flat == tag_id)[0][0]
                        
                        # Ottieni corners
                        corners_l_tag = corners_l[idx_l][0]
                        corners_r_tag = corners_r[idx_r][0]
                        
                        # Calcola centri
                        center_l = corners_l_tag.mean(axis=0)
                        center_r = corners_r_tag.mean(axis=0)
                        
                        # Calcola disparità
                        disparity = abs(center_l[0] - center_r[0])
                        
                        if disparity > self.config.MIN_DISPARITY:
                            # Usa parametri di calibrazione per calcolo preciso
                            focal_length = self.calibrator.P1[0, 0]
                            baseline = self.calibrator.calibration_quality['baseline']
                            
                            # Calcola distanza stereo
                            distance_raw = (focal_length * baseline) / disparity
                            
                            # Calcola pose del marcatore usando la camera SINISTRA
                            pose_result_left = self.estimate_pose_single_marker(
                                [corners_l_tag],
                                self.calibrator.camera_matrix_l,
                                self.calibrator.dist_coeffs_l,
                                distance_raw
                            )
                            
                            # Calcola pose del marcatore usando la camera DESTRA
                            pose_result_right = self.estimate_pose_single_marker(
                                [corners_r_tag],
                                self.calibrator.camera_matrix_r,
                                self.calibrator.dist_coeffs_r,
                                distance_raw
                            )
                            
                            if pose_result_left['success'] and pose_result_right['success']:
                                # Applica filtri per smoothing - CAMERA SINISTRA
                                distance_smooth, position_smooth_left, orientation_smooth_left = self.get_smoothed_values(
                                    tag_id, distance_raw, pose_result_left['position'], pose_result_left['orientation']
                                )
                                
                                # Applica filtri per smoothing - CAMERA DESTRA
                                position_smooth_right, orientation_smooth_right = self.get_smoothed_values_right(
                                    tag_id, pose_result_right['position'], pose_result_right['orientation']
                                )
                                
                                detection_results[int(tag_id)] = {
                                    'id': int(tag_id),
                                    'distance_raw': float(distance_raw),
                                    'distance_smooth': float(distance_smooth),
                                    'disparity': float(disparity),
                                    'center_left': [float(center_l[0]), float(center_l[1])],
                                    'center_right': [float(center_r[0]), float(center_r[1])],
                                    'corners_left': corners_l_tag.tolist(),
                                    'corners_right': corners_r_tag.tolist(),
                                    
                                    # DATI CAMERA SINISTRA
                                    'left_camera': {
                                        'position_3d': {
                                            'x': float(pose_result_left['position']['x']),
                                            'y': float(pose_result_left['position']['y']),
                                            'z': float(pose_result_left['position']['z'])  # Ora è = distance_raw
                                        },
                                        'position_3d_smooth': {
                                            'x': float(position_smooth_left['x']),
                                            'y': float(position_smooth_left['y']),
                                            'z': float(position_smooth_left['z'])  # Ora è = distance_smooth
                                        },
                                        'position_3d_original_solvepnp': pose_result_left.get('position_original_solvepnp', {}),
                                        'orientation': {
                                            'roll': float(pose_result_left['orientation']['roll']),
                                            'pitch': float(pose_result_left['orientation']['pitch']),
                                            'yaw': float(pose_result_left['orientation']['yaw'])
                                        },
                                        'orientation_smooth': {
                                            'roll': float(orientation_smooth_left['roll']),
                                            'pitch': float(orientation_smooth_left['pitch']),
                                            'yaw': float(orientation_smooth_left['yaw'])
                                        },
                                        'rotation_vector': pose_result_left['rotation_vector'],
                                        'translation_vector': pose_result_left['translation_vector'],
                                        'rotation_matrix': pose_result_left['rotation_matrix']
                                    },
                                    
                                    # DATI CAMERA DESTRA
                                    'right_camera': {
                                        'position_3d': {
                                            'x': float(pose_result_right['position']['x']),
                                            'y': float(pose_result_right['position']['y']),
                                            'z': float(pose_result_right['position']['z'])  # Ora è = distance_raw
                                        },
                                        'position_3d_smooth': {
                                            'x': float(position_smooth_right['x']),
                                            'y': float(position_smooth_right['y']),
                                            'z': float(position_smooth_right['z'])  # Ora è = distance_smooth
                                        },
                                        'position_3d_original_solvepnp': pose_result_right.get('position_original_solvepnp', {}),
                                        'orientation': {
                                            'roll': float(pose_result_right['orientation']['roll']),
                                            'pitch': float(pose_result_right['orientation']['pitch']),
                                            'yaw': float(pose_result_right['orientation']['yaw'])
                                        },
                                        'orientation_smooth': {
                                            'roll': float(orientation_smooth_right['roll']),
                                            'pitch': float(orientation_smooth_right['pitch']),
                                            'yaw': float(orientation_smooth_right['yaw'])
                                        },
                                        'rotation_vector': pose_result_right['rotation_vector'],
                                        'translation_vector': pose_result_right['translation_vector'],
                                        'rotation_matrix': pose_result_right['rotation_matrix']
                                    },
                                    
                                    # BACKWARD COMPATIBILITY - mantieni i vecchi campi per compatibilità
                                    'position_3d': {
                                        'x': float(pose_result_left['position']['x']),
                                        'y': float(pose_result_left['position']['y']),
                                        'z': float(pose_result_left['position']['z'])  # Ora è = distance_raw
                                    },
                                    'position_3d_smooth': {
                                        'x': float(position_smooth_left['x']),
                                        'y': float(position_smooth_left['y']),
                                        'z': float(position_smooth_left['z'])  # Ora è = distance_smooth
                                    },
                                    'position_3d_original_solvepnp': pose_result_left.get('position_original_solvepnp', {}),
                                    'orientation': {
                                        'roll': float(pose_result_left['orientation']['roll']),
                                        'pitch': float(pose_result_left['orientation']['pitch']),
                                        'yaw': float(pose_result_left['orientation']['yaw'])
                                    },
                                    'orientation_smooth': {
                                        'roll': float(orientation_smooth_left['roll']),
                                        'pitch': float(orientation_smooth_left['pitch']),
                                        'yaw': float(orientation_smooth_left['yaw'])
                                    },
                                    'rotation_vector': pose_result_left['rotation_vector'],
                                    'translation_vector': pose_result_left['translation_vector'],
                                    'rotation_matrix': pose_result_left['rotation_matrix'],
                                    'stereo_depth_confirmation': float(distance_raw),
                                    'timestamp': time.time()
                                }
                
                # Aggiorna dati di rilevamento
                with self.data_lock:
                    self.detection_data = {
                        'tags': detection_results,
                        'frame_count': frame_count,
                        'timestamp': time.time(),
                        'calibration_info': {
                            'rms_error': float(self.calibrator.calibration_quality.get('stereo_rms', 0)),
                            'baseline': float(self.calibrator.calibration_quality.get('baseline', 0))
                        },
                        'note': 'Z coordinate now matches stereo depth exactly - Extended with right camera processing'
                    }
                
                # Log periodico con conferma della correzione - ESTESO PER CAMERA DESTRA
                if frame_count % 30 == 0 and detection_results:
                    for tag_id, data in detection_results.items():
                        pos_l = data['left_camera']['position_3d_smooth']
                        ori_l = data['left_camera']['orientation_smooth']
                        pos_r = data['right_camera']['position_3d_smooth']
                        ori_r = data['right_camera']['orientation_smooth']
                        distance = data['distance_smooth']
                        logger.info(f"Tag {tag_id}: Dist={distance:.1f}cm")
                        logger.info(f"  Left  Cam: Pos({pos_l['x']:.1f}, {pos_l['y']:.1f}, {pos_l['z']:.1f})cm, "
                                  f"Ori({ori_l['roll']:.1f}, {ori_l['pitch']:.1f}, {ori_l['yaw']:.1f})°")
                        logger.info(f"  Right Cam: Pos({pos_r['x']:.1f}, {pos_r['y']:.1f}, {pos_r['z']:.1f})cm, "
                                  f"Ori({ori_r['roll']:.1f}, {ori_r['pitch']:.1f}, {ori_r['yaw']:.1f})°")
                
            except Exception as e:
                logger.error(f"Errore nel loop di rilevamento: {e}")
                continue
    
    def get_current_frame(self):
        """
        Ottiene il frame corrente per il video feed
        """
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def get_detection_data(self):
        """
        Ottiene i dati di rilevamento correnti
        """
        with self.data_lock:
            return self.detection_data.copy()

# =============================================================================
# WEB SERVICE CON HTTP.SERVER
# =============================================================================

class WebServiceHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        """Override per ridurre log verbosity"""
        pass
    
    def do_GET(self):
        """Gestisce le richieste GET"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        if path == '/video_feed':
            self.handle_video_feed()
        elif path == '/detection_data':
            self.handle_detection_data()
        elif path == '/status':
            self.handle_status()
        else:
            self.send_error(404, 'Endpoint not found')
    
    def do_POST(self):
        """Gestisce le richieste POST"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        if path == '/reset_filters':
            self.handle_reset_filters()
        elif path == '/shutdown':
            self.handle_shutdown()
        else:
            self.send_error(404, 'Endpoint not found')
    
    def handle_video_feed(self):
        """Handler per lo stream video"""
        try:
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            
            while True:
                frame = detector.get_current_frame()
                if frame is not None:
                    # Codifica frame in JPEG
                    ret, buffer = cv2.imencode('.jpg', frame)
                    if ret:
                        frame_bytes = buffer.tobytes()
                        self.wfile.write(b'--frame\r\n')
                        self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
                        self.wfile.write(frame_bytes)
                        self.wfile.write(b'\r\n')
                else:
                    time.sleep(0.1)
        except Exception as e:
            logger.error(f"Errore stream video: {e}")
    
    def handle_detection_data(self):
        """Handler per i dati di rilevamento"""
        try:
            data = detector.get_detection_data()
            json_data = json.dumps(data, indent=2)
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            self.wfile.write(json_data.encode('utf-8'))
        except Exception as e:
            logger.error(f"Errore dati rilevamento: {e}")
            self.send_error(500, 'Internal server error')
    
    def handle_status(self):
        """Handler per lo stato del servizio"""
        try:
            config = Config()
            status_data = {
                'service': 'April Tag Distance Detection',
                'version': '1.2 - Extended with Right Camera Processing',
                'status': 'running' if detector.is_running else 'stopped',
                'timestamp': time.time(),
                'opencv_version': cv2.__version__,
                'aruco_api': 'new' if detector.use_new_api else 'legacy',
                'features': [
                    'Left camera pose estimation',
                    'Right camera pose estimation',
                    'Stereo depth correction',
                    'Position and orientation smoothing for both cameras',
                    'Backward compatibility with existing API'
                ],
                'config': {
                    'camera_resolution_left': f"{config.CAMERA_LEFT_WIDTH}x{config.CAMERA_LEFT_HEIGHT}",
                    'camera_resolution_right': f"{config.CAMERA_RIGHT_WIDTH}x{config.CAMERA_RIGHT_HEIGHT}",
                    'aruco_dict': str(config.ARUCO_DICT),
                    'tag_size': config.TAG_SIZE_REAL
                }
            }
            
            json_data = json.dumps(status_data, indent=2)
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            self.wfile.write(json_data.encode('utf-8'))
        except Exception as e:
            logger.error(f"Errore stato servizio: {e}")
            self.send_error(500, 'Internal server error')
    
    def handle_reset_filters(self):
        """Handler per reset filtri - ESTESO PER CAMERA DESTRA"""
        try:
            detector.distance_filters.clear()
            detector.position_filters.clear()
            detector.orientation_filters.clear()
            # Reset anche filtri camera destra
            detector.position_filters_right.clear()
            detector.orientation_filters_right.clear()
            logger.info("Tutti i filtri resettati (inclusi filtri camera destra)")
            
            response_data = {
                'status': 'filters_reset',
                'filters_cleared': [
                    'distance_filters',
                    'position_filters (left camera)',
                    'orientation_filters (left camera)',
                    'position_filters_right (right camera)',
                    'orientation_filters_right (right camera)'
                ],
                'timestamp': time.time()
            }
            
            json_data = json.dumps(response_data, indent=2)
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            self.wfile.write(json_data.encode('utf-8'))
        except Exception as e:
            logger.error(f"Errore reset filtri: {e}")
            self.send_error(500, 'Internal server error')
    
    def handle_shutdown(self):
        """Handler per spegnimento servizio"""
        try:
            response_data = {
                'status': 'shutting_down',
                'timestamp': time.time()
            }
            
            json_data = json.dumps(response_data, indent=2)
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            self.wfile.write(json_data.encode('utf-8'))
            
            # Ferma il detector in un thread separato per permettere alla response di completarsi
            threading.Thread(target=detector.stop_detection, daemon=True).start()
            
        except Exception as e:
            logger.error(f"Errore shutdown: {e}")
            self.send_error(500, 'Internal server error')

class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    """HTTP Server con threading per gestire richieste multiple"""
    daemon_threads = True
    allow_reuse_address = True

# Istanza globale del detector
detector = AprilTagDistanceDetector()

def main():
    logger.info("Avvio Web Service April Tag Distance Detection v1.2")
    logger.info("Nuove funzionalità: Elaborazione camera destra con pose estimation")
    logger.info(f"OpenCV version: {cv2.__version__}")
    
    # Avvia il sistema di rilevamento
    if not detector.start_detection():
        logger.error("Impossibile avviare il sistema di rilevamento")
        return
    
    try:
        # Configura e avvia il server HTTP
        config = Config()
        server_address = (config.WEB_SERVICE_HOST, config.WEB_SERVICE_PORT)
        httpd = ThreadedHTTPServer(server_address, WebServiceHandler)
        
        logger.info(f"Web service avviato su {config.WEB_SERVICE_HOST}:{config.WEB_SERVICE_PORT}")
        logger.info("Endpoints disponibili:")
        logger.info(f"  - Video feed: http://{config.WEB_SERVICE_HOST}:{config.WEB_SERVICE_PORT}/video_feed")
        logger.info(f"  - Detection data: http://{config.WEB_SERVICE_HOST}:{config.WEB_SERVICE_PORT}/detection_data")
        logger.info(f"  - Status: http://{config.WEB_SERVICE_HOST}:{config.WEB_SERVICE_PORT}/status")
        logger.info(f"  - Reset filters (POST): http://{config.WEB_SERVICE_HOST}:{config.WEB_SERVICE_PORT}/reset_filters")
        logger.info(f"  - Shutdown (POST): http://{config.WEB_SERVICE_HOST}:{config.WEB_SERVICE_PORT}/shutdown")
        
        httpd.serve_forever()
                
    except KeyboardInterrupt:
        logger.info("Servizio interrotto dall'utente")
    except Exception as e:
        logger.error(f"Errore nel web service: {e}")
    finally:
        detector.stop_detection()
        logger.info("Web service terminato")

if __name__ == "__main__":
    main()