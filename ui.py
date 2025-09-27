# src/ui.py
import os
import sys
import shutil
import tempfile
import uuid
import time
from typing import List, Optional

import numpy as np
import cv2

from PyQt5.QtCore import Qt, pyqtSignal, QThread, QSize, QLocale, QTimer
from PyQt5.QtGui import QPixmap, QIcon, QFont, QKeySequence, QPainter, QColor
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QTextEdit, QFileDialog,
    QComboBox, QHBoxLayout, QVBoxLayout, QScrollArea, QMessageBox,
    QProgressBar, QShortcut, QFrame, QSizePolicy
)

from utils import logger, ensure_dir, correct_face_orientation
from detector import load_image, detect_landmarks, extract_mouth_region
from mask_utils import create_mouth_mask
from generator import generate_variants

# ---- small helper: clickable QLabel ----
from PyQt5.QtCore import pyqtSignal as Signal
class ClickableLabel(QLabel):
    clicked = Signal()
    doubleClicked = Signal()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.doubleClicked.emit()


# ---- Status Message Widget ----
class StatusMessage(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setWordWrap(True)
        self.setMinimumHeight(50)  # Increased height
        self.setMaximumHeight(60)
        self.setObjectName("StatusMessage")
        self.setFont(QFont("Segoe UI", 14, QFont.Bold))  # Larger font
        self.hide()

        # Auto-hide timer
        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.fade_out)

    def show_message(self, text: str, message_type: str = "info", duration: int = 4000):
        """Show a status message with auto-hide"""
        self.setText(text)
        self.setProperty("messageType", message_type)
        self.style().polish(self)  # Refresh styling
        self.show()
        if duration > 0:
            self.timer.start(duration)

    def fade_out(self):
        self.hide()


# ---- Worker thread for generation ----
class GenerateWorker(QThread):
    finished = pyqtSignal(list)      # emits list[str] output file paths
    error = pyqtSignal(str)

    def __init__(self, image_path: str, doctor_text: str, patient_text: str, num_outputs: int, tmpdir: str):
        super().__init__()
        self.image_path = image_path
        self.doctor_text = doctor_text
        self.patient_text = patient_text
        self.num_outputs = num_outputs
        self.tmpdir = tmpdir
        ensure_dir(self.tmpdir)

    def _save_rgb_to_file(self, arr: np.ndarray, prefix: str) -> str:
        """Save RGB numpy array to a PNG file in tmpdir and return path."""
        # ensure uint8
        arr = np.asarray(arr, dtype=np.uint8)
        if arr.ndim == 2:  # single channel
            arr_bgr = arr
        else:
            # convert RGB -> BGR for OpenCV write
            arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        filename = f"{prefix}_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}.png"
        path = os.path.join(self.tmpdir, filename)
        try:
            cv2.imwrite(path, arr_bgr)
        except Exception as e:
            logger.exception("Failed to write temp image %s: %s", path, e)
            raise
        return path

    def run(self):
        try:
            logger.info("Worker: starting pipeline for %s", self.image_path)

            # 1) Load image (RGB)
            img = load_image(self.image_path)

            # 2) Find canonical aligned image (0 orientation) using robust detector
            try:
                landmarks, aligned_img = detect_landmarks(img)
                logger.debug("Worker: detected landmarks and aligned image via detect_landmarks.")
            except Exception as e:
                logger.debug("Worker: detect_landmarks failed: %s. Trying correct_face_orientation fallback.", e)
                try:
                    aligned_img = correct_face_orientation(img)
                    # try detect again to ensure landmarks exist
                    landmarks, _ = detect_landmarks(aligned_img)
                    logger.debug("Worker: fallback align succeeded and landmarks found.")
                except Exception as e2:
                    logger.warning("Worker: fallback detection failed: %s. Proceeding with original image (no alignment).", e2)
                    aligned_img = img
                    try:
                        landmarks, _ = detect_landmarks(aligned_img)
                    except Exception:
                        landmarks = None

            # Save corrected aligned client image to tmp and use that as canonical "client_image_path"
            corrected_client_path = self._save_rgb_to_file(aligned_img, "client_aligned")
            logger.debug("Worker: saved aligned client image to %s", corrected_client_path)

            # 3) Create mask if we have landmarks
            if landmarks is not None:
                try:
                    mask = create_mouth_mask(aligned_img.shape, landmarks)
                except Exception as e:
                    logger.exception("Failed to create mask: %s", e)
                    mask = np.zeros(aligned_img.shape[:2], dtype=np.uint8)
            else:
                # safe fallback mask (empty)
                mask = np.zeros(aligned_img.shape[:2], dtype=np.uint8)

            # 4) Build prompt
            prompt = f"enhanced smile, {self.doctor_text}, {self.patient_text}".strip(", ")

            # 5) Call generator (we call generate_variants directly so we reuse aligned_img/mask)
            try:
                variants = generate_variants(aligned_img, mask, prompt, num_variants=self.num_outputs)
            except Exception as e:
                logger.exception("Generator failed: %s", e)
                raise RuntimeError(f"Generation error: {e}")

            if not variants:
                raise RuntimeError("Generator returned no outputs.")

            # 6) Save each generated image to tmp files and return file paths
            saved_paths: List[str] = []
            for i, item in enumerate(variants):
                if isinstance(item, dict):
                    arr = item.get("image")
                    if arr is None:
                        arr = item.get("image_rgb")
                    if arr is None:
                        arr = item.get("image_array")

                    if arr is None:
                        # If dict has no image array, skip
                        logger.warning("Variant %d: dict has no image entry, skipping", i)
                        continue
                elif isinstance(item, np.ndarray):
                    arr = item
                else:
                    logger.warning("Variant %d: unexpected variant type %s, skipping", i, type(item))
                    continue

                # Ensure shape and dtype
                try:
                    arr = np.asarray(arr, dtype=np.uint8)
                    if arr.ndim == 2:
                        # grayscale -> convert to RGB
                        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
                    elif arr.shape[2] == 4:
                        # RGBA -> RGB
                        arr = arr[:, :, :3]
                except Exception as e:
                    logger.exception("Failed to normalize variant array: %s", e)
                    continue

                out_path = self._save_rgb_to_file(arr, f"gen_{i+1}")
                saved_paths.append(out_path)
                logger.debug("Saved variant %d to %s", i+1, out_path)

            # Prepend the corrected client path as first item is original in some flows
            # But the enhanced UI expects on_generation_finished to build [client] + outputs itself.
            # We will emit only generated file paths; the UI will use self.client_image_path (which we update on upload).
            logger.info("Worker: finished saving %d generated outputs", len(saved_paths))
            # Emit the list of generated file paths
            self.finished.emit(saved_paths)

        except Exception as e:
            logger.exception("Worker failed: %s", e)
            self.error.emit(str(e))


# ---- Main UI ----
class SmileGeneratorUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("😊 Smile Generator")
        # icon (optional)
        icon_path = os.path.join("assets", "icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        # Enhanced sizing for maximized display - much larger defaults
        self.setMinimumSize(1600, 1000)
        self.resize(1920, 1080)  # Full HD default

        # state
        self.original_client_path: Optional[str] = None  # real original path user uploaded
        self.client_image_path: Optional[str] = None     # canonical aligned path saved in tmp
        self.current_main_image: Optional[str] = None
        self.worker: Optional[GenerateWorker] = None
        self.tmpdir = os.path.join(tempfile.gettempdir(), "smilegen_tmp")
        ensure_dir(self.tmpdir)

        # build UI
        self._create_widgets()
        self._create_layout()
        self._apply_stylesheet()

        # ESC toggles fullscreen
        self._shortcut_exit = QShortcut(QKeySequence("Esc"), self)
        self._shortcut_exit.activated.connect(self._toggle_fullscreen)

    # ---- widgets ----
    def _create_widgets(self):
        # header with larger styling for maximized display
        self.header = QLabel("AI Smile Generator")
        self.header.setObjectName("HeaderLabel")
        self.header.setAlignment(Qt.AlignCenter)
        self.header.setFont(QFont("Segoe UI", 28, QFont.Bold))  # Much larger font
        self.header.setFixedHeight(120)  # Increased height

        # upload section with enhanced sizing
        self.upload_section = QFrame()
        self.upload_section.setObjectName("SectionFrame")

        self.upload_btn = QPushButton("Upload Image")
        self.upload_btn.clicked.connect(self.upload_image)
        self.upload_btn.setObjectName("PrimaryButton")
        self.upload_btn.setMinimumHeight(65)  # Much larger button
        self.upload_btn.setFont(QFont("Segoe UI", 16, QFont.Bold))  # Larger font

        self.num_label = QLabel("Variants")
        self.num_label.setObjectName("InlineLabel")
        self.num_label.setFont(QFont("Segoe UI", 15, QFont.DemiBold))  # Larger font
        
        self.num_dropdown = QComboBox()
        self.num_dropdown.addItems([str(i) for i in range(1, 7)])
        # safe default index
        self.num_dropdown.setCurrentIndex(3 if self.num_dropdown.count() > 3 else 0)
        self.num_dropdown.setFixedWidth(120)  # Increased width
        self.num_dropdown.setMinimumHeight(65)  # Match button height
        self.num_dropdown.setFont(QFont("Segoe UI", 15, QFont.Bold))  # Larger font

        # Status message widget with enhanced sizing
        self.status_message = StatusMessage()

        # main preview section with larger sizing
        self.preview_section = QFrame()
        self.preview_section.setObjectName("SectionFrame")

        self.main_caption = QLabel("")
        self.main_caption.setAlignment(Qt.AlignCenter)
        self.main_caption.setFont(QFont("Segoe UI", 18, QFont.DemiBold))  # Larger font
        self.main_caption.setObjectName("CaptionLabel")
        self.main_caption.setMinimumHeight(60)  # Increased height

        self.main_preview = QLabel("Drop your client image here to get started")
        self.main_preview.setAlignment(Qt.AlignCenter)
        self.main_preview.setFrameShape(QFrame.StyledPanel)
        self.main_preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.main_preview.setMinimumSize(1000, 700)  # Much larger preview area
        self.main_preview.setObjectName("MainPreview")
        self.main_preview.setFont(QFont("Segoe UI", 20, QFont.Normal))  # Larger placeholder font

        # text input sections with enhanced sizing
        self.doctor_section = QFrame()
        self.doctor_section.setObjectName("SectionFrame")

        self.label_doctor = QLabel("Doctor's Recommendations")
        self.label_doctor.setObjectName("SectionLabel")
        self.label_doctor.setFixedHeight(70)  # Increased height
        self.label_doctor.setFont(QFont("Segoe UI", 18, QFont.Bold))  # Larger font

        self.doctor_text = QTextEdit()
        self.doctor_text.setPlaceholderText("(e.g., whiten teeth, align upper right, fix gap...)")
        self.doctor_text.setMinimumHeight(300)  # Increased to show more lines (was 150)
        self.doctor_text.setMaximumHeight(300)  # Increased max height (was 220)
        self.doctor_text.setFont(QFont("Segoe UI", 12))

        self.patient_section = QFrame()
        self.patient_section.setObjectName("SectionFrame")

        self.label_patient = QLabel("Patient's Needs")
        self.label_patient.setObjectName("SectionLabel")
        self.label_patient.setFixedHeight(70)  # Increased height
        self.label_patient.setFont(QFont("Segoe UI", 18, QFont.Bold))  # Larger font

        self.patient_text = QTextEdit()
        self.patient_text.setPlaceholderText("(e.g., natural look, Hollywood smile, subtle changes...)")
        self.patient_text.setMinimumHeight(300)  # Increased to show more lines (was 150)
        self.patient_text.setMaximumHeight(300)  # Increased max height (was 220)
        self.patient_text.setFont(QFont("Segoe UI", 12))

        # generation section with enhanced sizing
        self.generation_section = QFrame()
        self.generation_section.setObjectName("SectionFrame")

        self.generate_btn = QPushButton("✨ Generate Smile Suggestions")
        self.generate_btn.clicked.connect(self.on_generate_clicked)
        self.generate_btn.setObjectName("GenerateButton")
        self.generate_btn.setMinimumHeight(70)  # Much larger generate button
        self.generate_btn.setFont(QFont("Segoe UI", 18, QFont.Bold))  # Larger font

        self.progress = QProgressBar()
        self.progress.setFixedHeight(50)  # Increased height
        self.progress.setTextVisible(True)
        self.progress.setObjectName("ProgressBar")
        self.progress.setFont(QFont("Segoe UI", 14, QFont.Bold))  # Larger font
        self.progress.hide()  # Hidden by default

        # completion message with enhanced sizing
        self.completion_message = QLabel()
        self.completion_message.setAlignment(Qt.AlignCenter)
        self.completion_message.setObjectName("CompletionMessage")
        self.completion_message.setFont(QFont("Segoe UI", 16, QFont.Bold))  # Larger font
        self.completion_message.setMinimumHeight(60)  # Increased height
        self.completion_message.hide()

        # thumbnails section with enhanced sizing
        self.thumbs_section = QFrame()
        self.thumbs_section.setObjectName("SectionFrame")

        self.thumbs_label = QLabel("Generated Results")
        self.thumbs_label.setObjectName("SectionLabel")
        self.thumbs_label.setFont(QFont("Segoe UI", 18, QFont.Bold))  # Larger font
        self.thumbs_label.setFixedHeight(50)  # Increased height
        self.thumbs_label.hide()  # Show only when we have results

        self.thumb_scroll = QScrollArea()
        self.thumb_scroll.setWidgetResizable(True)
        self.thumb_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.thumb_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.thumb_scroll.setMaximumHeight(300)  # Increased height
        self.thumb_scroll.setMinimumHeight(250)  # Increased minimum

        self.thumbs_container = QWidget()
        self.thumbs_layout = QHBoxLayout()
        self.thumbs_layout.setSpacing(25)  # Increased spacing
        self.thumbs_layout.setContentsMargins(20, 15, 20, 15)  # Increased margins
        self.thumbs_container.setLayout(self.thumbs_layout)
        self.thumb_scroll.setWidget(self.thumbs_container)
        self.thumb_scroll.hide()  # Hidden initially

    # ---- layout with enhanced sizing ----
    def _create_layout(self):
        # Left panel - controls with enhanced spacing for maximized display
        left_layout = QVBoxLayout()
        left_layout.setSpacing(40)  # Much larger spacing between sections (was 30)
        left_layout.setContentsMargins(30, 30, 30, 30)
        left_layout.setAlignment(Qt.AlignTop)

        # Header
        left_layout.addWidget(self.header)
        left_layout.addSpacing(30)  # Increased spacing after header

        # Upload section with enhanced padding
        upload_layout = QVBoxLayout()
        upload_layout.setContentsMargins(25, 25, 25, 25)
        upload_layout.setSpacing(18)

        # Upload row with variants
        upload_row = QHBoxLayout()
        upload_row.setSpacing(20)
        upload_row.addWidget(self.upload_btn, stretch=3)
        upload_row.addWidget(self.num_label, alignment=Qt.AlignCenter)
        upload_row.addWidget(self.num_dropdown)
        upload_layout.addLayout(upload_row)
        upload_layout.addWidget(self.status_message)
        self.upload_section.setLayout(upload_layout)
        left_layout.addWidget(self.upload_section)

        left_layout.addSpacing(25)  # Extra spacing between upload and doctor sections

        # Doctor section with enhanced padding
        doctor_layout = QVBoxLayout()
        doctor_layout.setContentsMargins(25, 25, 25, 25)
        doctor_layout.setSpacing(15)
        doctor_layout.addWidget(self.label_doctor)
        doctor_layout.addWidget(self.doctor_text)
        self.doctor_section.setLayout(doctor_layout)
        left_layout.addWidget(self.doctor_section)

        left_layout.addSpacing(25)  # Extra spacing between doctor and patient sections

        # Patient section with enhanced padding
        patient_layout = QVBoxLayout()
        patient_layout.setContentsMargins(25, 25, 25, 25)
        patient_layout.setSpacing(15)
        patient_layout.addWidget(self.label_patient)
        patient_layout.addWidget(self.patient_text)
        self.patient_section.setLayout(patient_layout)
        left_layout.addWidget(self.patient_section)

        left_layout.addSpacing(25)  # Extra spacing between patient and generation sections

        # Generation section with enhanced padding
        gen_layout = QVBoxLayout()
        gen_layout.setContentsMargins(25, 25, 25, 25)
        gen_layout.setSpacing(18)
        gen_layout.addWidget(self.generate_btn)
        gen_layout.addWidget(self.progress)
        gen_layout.addWidget(self.completion_message)
        self.generation_section.setLayout(gen_layout)
        left_layout.addWidget(self.generation_section)

        # Add stretch to push everything up
        left_layout.addStretch(1)

        left_panel = QWidget()
        left_panel.setLayout(left_layout)
        left_panel.setObjectName("LeftPanel")
        left_panel.setMinimumWidth(550)  # Increased minimum width

        # Right panel - preview and results with enhanced sizing
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(30, 30, 30, 30)  # Increased margins
        right_layout.setSpacing(30)  # Increased spacing
        right_layout.setAlignment(Qt.AlignTop)

        # Preview section with enhanced padding
        preview_layout = QVBoxLayout()
        preview_layout.setContentsMargins(25, 25, 25, 25)  # Increased margins
        preview_layout.setSpacing(20)  # Increased spacing
        preview_layout.addWidget(self.main_caption)
        preview_layout.addWidget(self.main_preview, stretch=1)
        self.preview_section.setLayout(preview_layout)
        right_layout.addWidget(self.preview_section, stretch=3)

        # Thumbnails section with enhanced padding
        thumbs_layout = QVBoxLayout()
        thumbs_layout.setContentsMargins(25, 25, 25, 25)  # Increased margins
        thumbs_layout.setSpacing(20)  # Increased spacing
        thumbs_layout.addWidget(self.thumbs_label)
        thumbs_layout.addWidget(self.thumb_scroll)
        self.thumbs_section.setLayout(thumbs_layout)
        right_layout.addWidget(self.thumbs_section, stretch=1)

        right_panel = QWidget()
        right_panel.setLayout(right_layout)
        right_panel.setObjectName("RightPanel")
        right_panel.setMinimumWidth(1000)  # Increased minimum width

        # Main horizontal split with enhanced spacing
        main_layout = QHBoxLayout()
        main_layout.setSpacing(30)  # Increased spacing
        main_layout.setContentsMargins(20, 20, 20, 20)  # Reasonable outer margins
        main_layout.addWidget(left_panel, stretch=4)
        main_layout.addWidget(right_panel, stretch=6)
        self.setLayout(main_layout)

    # ---- Fixed stylesheet - removes size constraints, only handles appearance ----
    def _apply_stylesheet(self):
        """Stylesheet focused on appearance only - no size constraints"""
        style = """
        /* Main application styling - NO size constraints */
        QWidget {
            background-color: #0a1929;
            color: #ffffff;
            font-family: "Segoe UI", "Roboto", "Arial", sans-serif;
        }

        /* Header styling - appearance only */
        QLabel#HeaderLabel {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                        stop:0 #1976d2, stop:1 #42a5f5);
            color: white;
            border-radius: 25px;
            border: 3px solid rgba(255, 255, 255, 0.1);
        }

        /* Section frames - appearance only */
        QFrame#SectionFrame {
            background-color: #132f4c;
            border: 2px solid rgba(66, 165, 245, 0.2);
            border-radius: 20px;
        }

        /* Section labels - appearance only */
        QLabel#SectionLabel {
            color: #90caf9;
            background-color: rgba(25, 118, 210, 0.1);
            border-radius: 12px;
            border: 2px solid rgba(144, 202, 249, 0.2);
        }

        /* Inline labels - appearance only */
        QLabel#InlineLabel {
            color: #e3f2fd;
            font-weight: 600;
        }

        /* Caption labels - appearance only */
        QLabel#CaptionLabel {
            color: #90caf9;
            background-color: rgba(25, 118, 210, 0.1);
            border-radius: 18px;
            border: 2px solid rgba(144, 202, 249, 0.2);
        }

        /* Main preview area - appearance only */
        QLabel#MainPreview {
            background-color: #1e3a5f;
            border: 4px dashed rgba(66, 165, 245, 0.4);
            border-radius: 22px;
            color: #90caf9;
        }

        /* Primary button (Upload) - appearance only */
        QPushButton#PrimaryButton {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                        stop:0 #1976d2, stop:1 #1565c0);
            color: white;
            border: none;
            border-radius: 18px;
            font-weight: 700;
        }
        QPushButton#PrimaryButton:hover { 
            background: qlineargradient(x1:0,y1:0,x2:1,y2:1, 
                        stop:0 #2196f3, stop:1 #1976d2); 
        }
        QPushButton#PrimaryButton:pressed { 
            background: qlineargradient(x1:0,y1:0,x2:1,y2:1, 
                        stop:0 #1565c0, stop:1 #0d47a1); 
        }

        /* Generate button - appearance only */
        QPushButton#GenerateButton {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                        stop:0 #f57c00, stop:1 #ff9800);
            color: white;
            border: none;
            border-radius: 22px;
            font-weight: 700;
            border: 3px solid rgba(255, 152, 0, 0.3);
        }
        QPushButton#GenerateButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                        stop:0 #ff9800, stop:1 #ffb74d);
            border: 3px solid rgba(255, 152, 0, 0.5);
        }
        QPushButton#GenerateButton:pressed {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                        stop:0 #f57c00, stop:1 #e65100);
        }
        QPushButton#GenerateButton:disabled {
            background: #37474f;
            color: #78909c;
            border: 3px solid rgba(120, 144, 156, 0.2);
        }

        /* Text inputs - appearance only */
        QTextEdit, QLineEdit {
            background-color: #1e3a5f;
            border: 3px solid rgba(66, 165, 245, 0.3);
            border-radius: 18px;
            color: white;
            selection-background-color: #1976d2;
        }
        QTextEdit:focus, QLineEdit:focus { 
            border: 3px solid #42a5f5; 
            background-color: #2c5282; 
        }

        /* Combo box - appearance only */
        QComboBox {
            background-color: #1e3a5f;
            border: 3px solid rgba(66, 165, 245, 0.3);
            border-radius: 18px;
            color: white;
            font-weight: 600;
        }
        QComboBox:hover {
            border: 3px solid #42a5f5;
        }
        QComboBox::drop-down {
            border: none;
        }
        QComboBox::down-arrow {
            image: none;
            border-left: 6px solid transparent;
            border-right: 6px solid transparent;
            border-top: 10px solid #90caf9;
        }

        /* Progress bar - appearance only */
        QProgressBar#ProgressBar {
            background-color: #1e3a5f;
            border: 3px solid rgba(66, 165, 245, 0.3);
            border-radius: 18px;
            text-align: center;
            color: white;
            font-weight: 600;
        }
        QProgressBar#ProgressBar::chunk {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #FFFFFF, stop:1 #FFFFFF);
            border-radius: 15px;
        }

        /* Status messages - appearance only */
        QLabel#StatusMessage { 
            border-radius: 18px; 
            font-weight: 600; 
        }
        QLabel#StatusMessage[messageType="success"] { 
            background-color: rgba(76, 175, 80, 0.2); 
            color: #a5d6a7; 
            border: 2px solid rgba(76, 175, 80, 0.4); 
        }
        QLabel#StatusMessage[messageType="error"] { 
            background-color: rgba(244, 67, 54, 0.2); 
            color: #ef9a9a; 
            border: 2px solid rgba(244, 67, 54, 0.4); 
        }
        QLabel#StatusMessage[messageType="info"] { 
            background-color: rgba(33, 150, 243, 0.2); 
            color: #90caf9; 
            border: 2px solid rgba(33, 150, 243, 0.4); 
        }

        /* Completion message - appearance only */
        QLabel#CompletionMessage {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                        stop:0 rgba(76, 175, 80, 0.2), stop:1 rgba(139, 195, 74, 0.2));
            color: #a5d6a7;
            border: 3px solid rgba(76, 175, 80, 0.4);
            border-radius: 18px;
            font-weight: 700;
        }

        /* Scroll area - appearance only */
        QScrollArea { 
            background-color: transparent; 
            border: none; 
            border-radius: 18px; 
        }
        QScrollBar:horizontal {
            background-color: #1e3a5f;
            border-radius: 8px;
        }
        QScrollBar::handle:horizontal {
            background-color: #42a5f5;
            border-radius: 8px;
        }
        QScrollBar::handle:horizontal:hover {
            background-color: #64b5f6;
        }

        /* Thumbnail frames - appearance only */
        QFrame#ThumbnailFrame {
            background-color: #1e3a5f;
            border: 3px solid rgba(66, 165, 245, 0.3);
            border-radius: 15px;
        }
        QFrame#ThumbnailFrame:hover {
            border: 3px solid #42a5f5;
            background-color: #2c5282;
        }

        /* Panel backgrounds */
        QWidget#LeftPanel, QWidget#RightPanel { 
            background-color: transparent; 
        }
        """

        # Skip external stylesheet completely for now to avoid conflicts
        self.setStyleSheet(style)

    # ---- utility methods ----
    def set_busy(self, busy: bool):
        """Enhanced busy state management"""
        if busy:
            self.progress.show()
            self.progress.setRange(0, 0)  # Indeterminate
            self.completion_message.hide()
            self.generate_btn.setEnabled(False)
            self.upload_btn.setEnabled(False)
        else:
            self.progress.hide()
            self.generate_btn.setEnabled(True)
            self.upload_btn.setEnabled(True)

    def show_completion_message(self, success: bool, count: int = 0):
        """Show generation completion message"""
        if success:
            self.completion_message.setText(f"✅ Generation Complete! Created {count} variations.")
            self.completion_message.show()
            # Auto-hide after 5 seconds
            QTimer.singleShot(5000, self.completion_message.hide)
        else:
            self.completion_message.setText("❌ Generation failed. Please try again.")
            self.completion_message.show()
            QTimer.singleShot(5000, self.completion_message.hide)

    # create composite pixmap: generated with inset original for comparison
    def _make_comparison_pixmap(self, generated_path: str, original_path: str) -> Optional[QPixmap]:
        if not (generated_path and original_path):
            return None
        if not os.path.exists(generated_path) or not os.path.exists(original_path):
            return None
        g_pix = QPixmap(generated_path)
        o_pix = QPixmap(original_path)
        if g_pix.isNull() or o_pix.isNull():
            return None

        base_size = self.main_preview.size()
        if base_size.width() <= 0 or base_size.height() <= 0:
            base_size = QSize(1200, 800)  # Larger fallback size

        base = g_pix.scaled(base_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        painter = QPainter(base)

        # Inset original: 24% of base width
        inset_w = max(150, int(base.width() * 0.24))  # Larger minimum
        inset_h = int(o_pix.height() * (inset_w / o_pix.width())) if o_pix.width() > 0 else int(inset_w * 0.75)
        inset = o_pix.scaled(inset_w, inset_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # Position top-right with margin
        margin = 20  # Increased margin
        x = base.width() - inset.width() - margin
        y = margin

        # Draw rounded background for inset
        painter.setRenderHint(QPainter.Antialiasing)
        border_rect = QColor(0, 0, 0, 200)
        painter.fillRect(x - 10, y - 10, inset.width() + 20, inset.height() + 20, border_rect)
        painter.drawPixmap(x, y, inset)

        # Label "Original" under inset with larger font
        painter.setPen(QColor(255, 255, 255, 240))
        painter.setFont(QFont("Segoe UI", 12, QFont.Bold))  # Larger font
        painter.drawText(x + 10, y + inset.height() + 25, "Original")
        painter.end()
        return base

    # show image in the main preview
    def _set_main_image(self, image_path: str, is_generated: bool = False):
        if not image_path or not os.path.exists(image_path):
            self.main_caption.setText("Preview unavailable")
            self.main_preview.setText("Preview unavailable")
            return

        if is_generated and self.client_image_path:
            comp = self._make_comparison_pixmap(image_path, self.client_image_path)
            if comp:
                self.current_main_image = image_path
                scaled = comp.scaled(self.main_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.main_preview.setPixmap(scaled)
                self.main_caption.setText(f"Generated — {os.path.basename(image_path).split('_')[1]}")
                return

        pix = QPixmap(image_path)
        if pix.isNull():
            self.main_preview.setText("Could not load image")
            return

        self.current_main_image = image_path
        scaled = pix.scaled(self.main_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.main_preview.setPixmap(scaled)

        # Set caption
        if image_path == self.client_image_path:
            self.main_caption.setText("Original Image")
        else:
            self.main_caption.setText(f"✨ {os.path.basename(image_path)}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if getattr(self, "current_main_image", None):
            try:
                is_generated = (self.current_main_image != self.client_image_path)
                self._set_main_image(self.current_main_image, is_generated=is_generated)
            except Exception:
                pass

    # ---- Event handlers ----
    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Client Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        if not file_path:
            return

        # Show processing message
        self.status_message.show_message("📤 Processing image...", "info", 0)

        try:
            # Keep original path for downloads or reference
            self.original_client_path = file_path

            # Load image as RGB numpy
            img = load_image(file_path)

            # Use detect_landmarks to get aligned image (0 orientation). If fails, fallback to correct_face_orientation.
            try:
                landmarks, aligned_img = detect_landmarks(img)
                logger.debug("upload_image: aligned via detect_landmarks.")
            except Exception as e:
                logger.debug("upload_image: detect_landmarks failed: %s. Falling back to correct_face_orientation.", e)
                try:
                    aligned_img = correct_face_orientation(img)
                    # try detect again (optional)
                    try:
                        landmarks, _ = detect_landmarks(aligned_img)
                    except Exception:
                        landmarks = None
                except Exception as e2:
                    logger.warning("upload_image: correct_face_orientation failed: %s. Using original image.", e2)
                    aligned_img = img
                    landmarks = None

            # Save aligned image to tmp and set as client_image_path
            aligned_path = os.path.join(self.tmpdir, f"client_aligned_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}.png")
            try:
                # convert RGB -> BGR for saving
                cv2.imwrite(aligned_path, cv2.cvtColor(np.asarray(aligned_img, dtype=np.uint8), cv2.COLOR_RGB2BGR))
                self.client_image_path = aligned_path
                logger.debug("Saved aligned client image to %s", aligned_path)
            except Exception as e:
                logger.exception("Failed to save aligned preview: %s", e)
                # fallback to original file path
                self.client_image_path = file_path

            # Update main preview to aligned image
            self._set_main_image(self.client_image_path, is_generated=False)

            # Create mask preview for developer debugging (optional)
            if landmarks is not None:
                try:
                    mask = create_mouth_mask(aligned_img.shape, landmarks)
                    mask_tmp = os.path.join(self.tmpdir, f"mask_preview_{uuid.uuid4().hex[:6]}.png")
                    cv2.imwrite(mask_tmp, mask)
                    logger.debug("Saved mask preview to %s", mask_tmp)
                except Exception as e:
                    logger.debug("Could not write mask preview: %s", e)

            # Show success message
            filename = os.path.basename(file_path)
            self.status_message.show_message(f"✅ Successfully uploaded: {filename}", "success", 3000)

        except Exception as e:
            logger.exception("Upload processing failed: %s", e)
            self.status_message.show_message(f"⚠️ Upload warning: {str(e)}", "error", 4000)
            # fallback: show raw file
            try:
                self.client_image_path = file_path
                self._set_main_image(file_path, is_generated=False)
            except Exception:
                pass

    def on_generate_clicked(self):
        if not self.client_image_path:
            self.status_message.show_message("⚠️ Please upload a client image first", "error", 3000)
            return

        doctor_notes = self.doctor_text.toPlainText().strip()
        patient_needs = self.patient_text.toPlainText().strip()
        num_outputs = int(self.num_dropdown.currentText())

        # Enhanced busy state
        self.set_busy(True)
        self.clear_thumbnails()

        # Show generation message
        self.status_message.show_message("🎨 Generating smile variations...", "info", 0)

        # Launch worker; pass tmpdir so worker saves generated files there
        self.worker = GenerateWorker(self.client_image_path, doctor_notes, patient_needs, num_outputs, tmpdir=self.tmpdir)
        self.worker.finished.connect(self.on_generation_finished)
        self.worker.error.connect(self.on_generation_error)
        self.worker.start()

    def on_generation_finished(self, outputs: List[str]):
        """
        outputs: list of file paths for generated images (PNG)
        UI will use self.client_image_path (aligned client) as the overlay original.
        """
        self.set_busy(False)
        self.status_message.hide()

        if outputs:
            # Show completion message
            self.show_completion_message(True, len(outputs))

            # Show first generated result with overlay of aligned original
            first_generated = outputs[0]
            self._set_main_image(first_generated, is_generated=True)

            # Populate thumbnails
            all_images = [self.client_image_path] + outputs  # client (aligned) first
            self._populate_thumbnails(all_images)

            # Show results section
            self.thumbs_label.show()
            self.thumb_scroll.show()
        else:
            self.show_completion_message(False)

    def on_generation_error(self, message: str):
        self.set_busy(False)
        self.status_message.hide()
        self.show_completion_message(False)
        QMessageBox.critical(self, "Generation Error", f"Failed to generate smile variations:\n\n{message}")

    def clear_thumbnails(self):
        """Clear thumbnail widgets"""
        while self.thumbs_layout.count():
            item = self.thumbs_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        # Hide thumbnails section when empty
        self.thumbs_label.hide()
        self.thumb_scroll.hide()

    def _make_thumbnail_widget(self, path: str, caption: str, is_original: bool):
        """Create enhanced thumbnail widget with larger sizing"""
        container = QFrame()
        container.setObjectName("ThumbnailFrame")
        container.setFixedSize(220, 200)  # Much larger thumbnails

        # Apply thumbnail-specific styling
        container.setStyleSheet("""
            QFrame#ThumbnailFrame {
                background-color: #1e3a5f;
                border: 3px solid rgba(66, 165, 245, 0.3);
                border-radius: 15px;
                margin: 8px;
            }
            QFrame#ThumbnailFrame:hover {
                border: 3px solid #42a5f5;
                background-color: #2c5282;
                transform: scale(1.02);
            }
        """)

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)  # Increased margins
        layout.setSpacing(8)

        # Image label with larger sizing
        img_label = ClickableLabel()
        img_label.setFixedSize(195, 150)  # Much larger image area
        img_label.setAlignment(Qt.AlignCenter)
        img_label.setStyleSheet("""
            border: 2px solid rgba(66, 165, 245, 0.2);
            border-radius: 12px;
            background-color: #132f4c;
        """)

        if os.path.exists(path):
            pix = QPixmap(path).scaled(img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            img_label.setPixmap(pix)
        else:
            img_label.setText("Missing")
            img_label.setStyleSheet(img_label.styleSheet() + "color: #ef5350;")

        # Caption with larger font
        caption_label = QLabel(caption)
        caption_label.setAlignment(Qt.AlignCenter)
        caption_label.setFont(QFont("Segoe UI", 13, QFont.Bold))  # Larger font
        caption_label.setStyleSheet("""
            background-color: rgba(25, 118, 210, 0.1);
            color: #90caf9;
            border-radius: 8px;
            padding: 6px 12px;
            font-size: 13px;
            font-weight: 600;
            border: 2px solid rgba(144, 202, 249, 0.2);
        """)

        # Connect events
        if is_original:
            img_label.clicked.connect(lambda p=path: self._set_main_image(p, is_generated=False))
        else:
            img_label.clicked.connect(lambda p=path: self._set_main_image(p, is_generated=True))

        img_label.doubleClicked.connect(lambda p=path: self.open_file(p))

        layout.addWidget(img_label)
        layout.addWidget(caption_label)
        container.setLayout(layout)
        return container

    def _populate_thumbnails(self, file_paths: List[str]):
        """Populate thumbnail gallery with enhanced styling"""
        self.clear_thumbnails()

        for i, path in enumerate(file_paths):
            if i == 0:
                caption = "📸 Original"
                is_orig = True
            else:
                caption = f"✨ Variant {i}"
                is_orig = False

            widget = self._make_thumbnail_widget(path, caption, is_original=is_orig)
            self.thumbs_layout.addWidget(widget)

        # Add stretch to center thumbnails
        self.thumbs_layout.addStretch(1)

    def download_result(self, src_path: str):
        """Enhanced download with better user feedback"""
        if not os.path.exists(src_path):
            self.status_message.show_message("⚠️ Generated file no longer exists", "error", 3000)
            return

        suggested = f"smile_result_{os.path.basename(src_path)}"
        dest_path, _ = QFileDialog.getSaveFileName(
            self, "Save Generated Smile", suggested,
            "Images (*.png *.jpg *.jpeg)"
        )

        if dest_path:
            try:
                shutil.copyfile(src_path, dest_path)
                filename = os.path.basename(dest_path)
                self.status_message.show_message(f"💾 Saved: {filename}", "success", 3000)
            except Exception as e:
                logger.exception("Save failed: %s", e)
                self.status_message.show_message(f"❌ Save failed: {str(e)}", "error", 4000)

    def open_file(self, path: str):
        """Open file in default system application"""
        if not os.path.exists(path):
            self.status_message.show_message("⚠️ File not found", "error", 2000)
            return

        try:
            if sys.platform.startswith("win"):
                os.startfile(path)
            elif sys.platform == "darwin":
                os.system(f"open '{path}'")
            else:
                os.system(f"xdg-open '{path}'")
        except Exception as e:
            logger.warning("Failed to open file: %s", e)
            self.status_message.show_message("⚠️ Could not open file", "error", 2000)

    def _toggle_fullscreen(self):
        """Enhanced fullscreen toggle with status message"""
        if self.isFullScreen():
            self.showMaximized()
            self.status_message.show_message("🪟 Exited fullscreen mode", "info", 2000)
        else:
            self.showFullScreen()
            self.status_message.show_message("🔳 Entered fullscreen mode", "info", 2000)


# ---- Enhanced entrypoint ----
def main():
    """Enhanced application startup optimized for maximized display"""
    # Force English locale for consistent UI
    QLocale.setDefault(QLocale(QLocale.English, QLocale.UnitedStates))

    app = QApplication(sys.argv)

    # Set application properties
    app.setApplicationName("AI Smile Generator")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("SmileAI")

    # Enhanced font settings for larger displays
    app.setFont(QFont("Segoe UI", 14))  # Larger base font

    # Create and show window
    window = SmileGeneratorUI()
    window.showMaximized()  # Always start maximized

    # Add application icon to taskbar (if available)
    icon_path = os.path.join("assets", "icon.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())