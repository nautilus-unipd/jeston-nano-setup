#!/usr/bin/env python3
"""
Rilevamento Distanza April Tag con Telecamera Stereo - Image Processor
Versione: Image Processor 1.0 - Elabora solo immagini singole

Requisiti:
pip install opencv-python numpy

Uso:
python stereo_apriltag_processor.py --image path.png --output /path/to/output/folder

Output:
- stereo_frame.png : Immagine stereo rettificata
- detection_data.json : Tutti i dati di rilevamento
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import json
import os
import time
import pickle
import argparse
from datetime import datetime
import logging

# =============================================================================
# CONFIGURAZIONE GLOBALE
# =============================================================================

class Config:
    # Configurazione risoluzione telecamere
    CAMERA_LEFT_WIDTH = 640
    CAMERA_LEFT_HEIGHT = 480
    CAMERA_RIGHT_WIDTH = 640
    CAMERA_RIGHT_HEIGHT = 480
    
    # Parametri April Tag
    ARUCO_DICT = aruco.DICT_7X7_250
    TAG_SIZE_REAL = 5.0  # cm
    
    # File di calibrazione
    CALIBRATION_FILE = "stereo_calibration_data.pkl"
    
    # Parametri di qualità
    MIN_DISPARITY = 1.0  # pixel

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# UTILITÀ
# =============================================================================

def split_stereo_frame(frame, config=Config):
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
    def __init__(self, config=Config):
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
# PROCESSORE IMMAGINI APRIL TAG
# =============================================================================

class AprilTagImageProcessor:
    def __init__(self, config=Config, image_path=None, output_folder=None):
        self.config = config
        self.calibrator = StereoCalibrator(config)
        self.image_path = image_path
        self.output_folder = output_folder
        
        # Crea cartella di output se non esiste
        if self.output_folder and not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            logger.info(f"Cartella di output creata: {self.output_folder}")
        
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
        
        logger.info(f"Processore immagini April Tag inizializzato")
        logger.info(f"Immagine da analizzare: {image_path}")
        if self.output_folder:
            logger.info(f"Output salvato in: {self.output_folder}")
    
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
    
    def save_results(self, rectified_frame, detection_data):
        """Salva l'immagine e i dati JSON nella cartella di output"""
        if not self.output_folder:
            logger.error("Cartella di output non specificata")
            return False
        
        try:
            # Salva immagine stereo rettificata
            image_path = os.path.join(self.output_folder, "stereo_frame.png")
            cv2.imwrite(image_path, rectified_frame)
            logger.info(f"Immagine salvata: {image_path}")
            
            # Salva dati JSON
            json_path = os.path.join(self.output_folder, "detection_data.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(detection_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Dati salvati: {json_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Errore salvataggio risultati: {e}")
            return False
    
    def process_image(self):
        """Carica e processa l'immagine stereo specificata"""
        try:
            if not os.path.exists(self.image_path):
                logger.error(f"Immagine non trovata: {self.image_path}")
                return False
            
            frame = cv2.imread(self.image_path)
            if frame is None:
                logger.error(f"Impossibile caricare immagine: {self.image_path}")
                return False
            
            logger.info(f"Immagine caricata: {frame.shape}")
            
            # Dividi frame stereo
            frame_l, frame_r = split_stereo_frame(frame, self.config)
            
            # Rettifica usando calibrazione
            rect_l = cv2.remap(frame_l, self.calibrator.map1_l, self.calibrator.map2_l, cv2.INTER_LINEAR)
            rect_r = cv2.remap(frame_r, self.calibrator.map1_r, self.calibrator.map2_r, cv2.INTER_LINEAR)
            
            # Ricomponi frame stereo rettificato
            rectified_frame = np.hstack([rect_l, rect_r])
            
            # Rileva April Tags sui frame rettificati
            corners_l, ids_l, _ = self.detect_markers(rect_l)
            corners_r, ids_r, _ = self.detect_markers(rect_r)
            
            # Calcola distanze per tag comuni
            detection_results = {}
            
            if ids_l is not None and ids_r is not None:
                ids_l_flat = ids_l.flatten()
                ids_r_flat = ids_r.flatten()
                common_ids = set(ids_l_flat) & set(ids_r_flat)
                
                logger.info(f"Tag rilevati - Sinistra: {ids_l_flat}, Destra: {ids_r_flat}, Comuni: {list(common_ids)}")
                
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
                        
                        # Calcola pose del marcatore usando la camera sinistra
                        pose_result = self.estimate_pose_single_marker(
                            [corners_l_tag],
                            self.calibrator.camera_matrix_l,
                            self.calibrator.dist_coeffs_l,
                            distance_raw
                        )
                        
                        if pose_result['success']:
                            detection_results[int(tag_id)] = {
                                'id': int(tag_id),
                                'distance_raw': float(distance_raw),
                                'distance_smooth': float(distance_raw),  # Senza smoothing per immagine singola
                                'disparity': float(disparity),
                                'center_left': [float(center_l[0]), float(center_l[1])],
                                'center_right': [float(center_r[0]), float(center_r[1])],
                                'corners_left': corners_l_tag.tolist(),
                                'corners_right': corners_r_tag.tolist(),
                                'position_3d': {
                                    'x': float(pose_result['position']['x']),
                                    'y': float(pose_result['position']['y']),
                                    'z': float(pose_result['position']['z'])
                                },
                                'position_3d_smooth': {
                                    'x': float(pose_result['position']['x']),  # Stessi valori per immagine singola
                                    'y': float(pose_result['position']['y']),
                                    'z': float(pose_result['position']['z'])
                                },
                                'position_3d_original_solvepnp': pose_result.get('position_original_solvepnp', {}),
                                'orientation': {
                                    'roll': float(pose_result['orientation']['roll']),
                                    'pitch': float(pose_result['orientation']['pitch']),
                                    'yaw': float(pose_result['orientation']['yaw'])
                                },
                                'orientation_smooth': {
                                    'roll': float(pose_result['orientation']['roll']),  # Stessi valori per immagine singola
                                    'pitch': float(pose_result['orientation']['pitch']),
                                    'yaw': float(pose_result['orientation']['yaw'])
                                },
                                'rotation_vector': pose_result['rotation_vector'],
                                'translation_vector': pose_result['translation_vector'],
                                'rotation_matrix': pose_result['rotation_matrix'],
                                'stereo_depth_confirmation': float(distance_raw),
                                'timestamp': time.time()
                            }
            
            # Prepara dati di rilevamento completi
            detection_data = {
                'tags': detection_results,
                'frame_count': 1,
                'timestamp': time.time(),
                'mode': 'single_image',
                'image_path': self.image_path,
                'calibration_info': {
                    'rms_error': float(self.calibrator.calibration_quality.get('stereo_rms', 0)),
                    'baseline': float(self.calibrator.calibration_quality.get('baseline', 0))
                },
                'note': 'Z coordinate matches stereo depth exactly - Single image analysis'
            }
            
            # Salva risultati
            self.save_results(rectified_frame, detection_data)
            
            # Log risultati
            if detection_results:
                logger.info(f"Rilevati {len(detection_results)} tag nell'immagine:")
                for tag_id, data in detection_results.items():
                    pos = data['position_3d']
                    ori = data['orientation']
                    distance = data['distance_raw']
                    logger.info(f"  Tag {tag_id}: Dist={distance:.1f}cm, "
                              f"Pos({pos['x']:.1f}, {pos['y']:.1f}, {pos['z']:.1f})cm, "
                              f"Ori({ori['roll']:.1f}, {ori['pitch']:.1f}, {ori['yaw']:.1f})°")
            else:
                logger.info("Nessun tag rilevato nell'immagine")
            
            return True
            
        except Exception as e:
            logger.error(f"Errore elaborazione immagine: {e}")
            return False
    
    def run(self):
        """Metodo principale per eseguire il rilevamento"""
        # Carica calibrazione
        if not self.calibrator.load_calibration():
            logger.error("ERRORE: Nessuna calibrazione trovata!")
            return False
        
        logger.info("Avvio elaborazione immagine...")
        return self.process_image()

def parse_arguments():
    """Parse degli argomenti da linea di comando"""
    parser = argparse.ArgumentParser(
        description='Rilevamento Distanza April Tag con Telecamera Stereo - Image Processor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempio di utilizzo:
  python stereo_apriltag_processor.py --image frame.png --output /path/to/output

Output generato:
  - stereo_frame.png : Immagine stereo rettificata
  - detection_data.json : Tutti i dati di rilevamento
        """
    )
    
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path dell\'immagine stereo PNG da analizzare (richiesto)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path della cartella dove salvare i risultati (richiesto)'
    )
    
    return parser.parse_args()

def main():
    # Parse degli argomenti
    args = parse_arguments()
    
    logger.info("Avvio April Tag Image Processor")
    logger.info(f"OpenCV version: {cv2.__version__}")
    
    # Verifica che l'immagine esista
    if not os.path.exists(args.image):
        logger.error(f"Immagine non trovata: {args.image}")
        return
    
    logger.info(f"Immagine di input: {args.image}")
    logger.info(f"Output salvato in: {args.output}")
    
    # Inizializza processore
    processor = AprilTagImageProcessor(
        image_path=args.image,
        output_folder=args.output
    )
    
    # Esegui elaborazione
    success = processor.run()
    
    if success:
        logger.info("Elaborazione completata con successo!")
        logger.info(f"Risultati salvati in: {args.output}")
        logger.info("  - stereo_frame.png : Immagine stereo rettificata")
        logger.info("  - detection_data.json : Dati di rilevamento completi")
    else:
        logger.error("Elaborazione fallita!")

if __name__ == "__main__":
    main()