# src/ui.py
import os
import sys
import tempfile
from typing import List, Optional

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QSize, QLocale
from PyQt5.QtGui import QPixmap, QIcon, QFont, QKeySequence
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QTextEdit, QFileDialog,
    QComboBox, QHBoxLayout, QVBoxLayout, QScrollArea, QMessageBox,
    QProgressBar, QShortcut, QFrame, QSizePolicy, QSplitter
)

from utils import logger
from pipeline import run_pipeline
from generator import numpy_to_qpixmap
from detector import load_image, detect_landmarks
from utils import correct_face_orientation

# ---- small helper: clickable QLabel ----
from PyQt5.QtCore import pyqtSignal as Signal
class ClickableLabel(QLabel):
    clicked = Signal()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()


# ---- Worker thread for generation ----
class GenerateWorker(QThread):
    # emits (list[QPixmap], corrected QPixmap)
    finished = pyqtSignal(list, QPixmap)
    error = pyqtSignal(str)

    def __init__(self, image_path: str, doctor_text: str, patient_text: str, num_outputs: int):
        super().__init__()
        self.image_path = image_path
        self.doctor_text = doctor_text
        self.patient_text = patient_text
        self.num_outputs = num_outputs

    def run(self):
        try:
            logger.info("Worker: starting pipeline for %s", self.image_path)

            # Load image (RGB)
            img = load_image(self.image_path)

            # Try to get an aligned (upright) image for preview using detect_landmarks.
            # detect_landmarks tries 0/90/180/270 and returns the rotated image on which detection succeeded.
            try:
                landmarks, aligned_img = detect_landmarks(img)
                logger.debug("Worker: got aligned image from detect_landmarks.")
            except Exception as e:
                logger.debug("Worker: detect_landmarks failed for preview: %s. Falling back to correct_face_orientation.", e)
                try:
                    aligned_img = correct_face_orientation(img)
                except Exception as e2:
                    logger.warning("Worker: correct_face_orientation also failed: %s. Using original image for preview.", e2)
                    aligned_img = img

            # Ensure contiguous before creating QPixmap
            aligned_img = np.ascontiguousarray(aligned_img)
            corrected_pixmap = numpy_to_qpixmap(aligned_img)

            # Run the pipeline (this will do detection+masking+generation using the aligned image)
            outputs = run_pipeline(
                self.image_path,
                doctor_recommendations=self.doctor_text,
                patient_needs=self.patient_text,
                num_outputs=self.num_outputs
            )

            if not outputs:
                raise RuntimeError("Pipeline returned no outputs.")

            # outputs may be list[dict] or list[np.ndarray]; handle both
            qpixmaps = []
            for item in outputs:
                if isinstance(item, dict):
                    if "image" not in item:
                        logger.warning("Generator output dict missing 'image' key; skipping item.")
                        continue
                    arr = item["image"]
                elif isinstance(item, np.ndarray):
                    arr = item
                else:
                    logger.warning("Unexpected generator output type: %s. Skipping.", type(item))
                    continue

                # ensure contiguous and RGB uint8
                try:
                    arr = np.asarray(arr, dtype=np.uint8)
                    arr = np.ascontiguousarray(arr)
                    qpix = numpy_to_qpixmap(arr)
                    qpixmaps.append(qpix)
                except Exception as e:
                    logger.exception("Failed to convert generated image to QPixmap: %s", e)

            logger.info("Worker: finished, %d outputs", len(qpixmaps))
            self.finished.emit(qpixmaps, corrected_pixmap)

        except Exception as e:
            logger.exception("Worker failed: %s", e)
            self.error.emit(str(e))


# ---- Main UI ----
class SmileGeneratorUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("😊 Smile Generator")
        icon_path = os.path.join("assets", "icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self.setMinimumSize(1400, 900)

        # state
        self.client_image_path: Optional[str] = None
        self.current_main_pixmap: Optional[QPixmap] = None
        self.generated_pixmaps: List[QPixmap] = []
        self.corrected_pixmap: Optional[QPixmap] = None
        self.worker: Optional[GenerateWorker] = None
        self.tmpdir = tempfile.gettempdir()

        # build UI
        self._create_widgets()
        self._create_layout()
        self._apply_stylesheet()

        self._shortcut_exit = QShortcut(QKeySequence("Esc"), self)
        self._shortcut_exit.activated.connect(self._toggle_fullscreen)

    # ---- widgets ----
    def _create_widgets(self):
        self.header = QLabel("AI Smile Generator")
        self.header.setObjectName("HeaderLabel")
        self.header.setAlignment(Qt.AlignCenter)
        self.header.setFont(QFont("Segoe UI", 20, QFont.Bold))
        self.header.setFixedHeight(80)

        self.upload_btn = QPushButton("📤 Upload Client Image")
        self.upload_btn.clicked.connect(self.upload_image)

        self.num_dropdown = QComboBox()
        self.num_dropdown.addItems([str(i) for i in range(1, 7)])
        self.num_dropdown.setCurrentIndex(3)
        self.num_dropdown.setFixedWidth(70)

        self.label_doctor = QLabel("Doctor's Recommendations:")
        self.label_doctor.setObjectName("FieldLabel")
        self.doctor_text = QTextEdit()
        self.doctor_text.setMinimumHeight(120)

        self.label_patient = QLabel("Patient's Needs:")
        self.label_patient.setObjectName("FieldLabel")
        self.patient_text = QTextEdit()
        self.patient_text.setMinimumHeight(120)

        self.generate_btn = QPushButton("✨ Generate Smile Suggestions")
        self.generate_btn.clicked.connect(self.on_generate_clicked)

        self.progress = QProgressBar()
        self.progress.setFixedHeight(30)
        self.progress.setTextVisible(True)

        # Preview
        self.main_caption = QLabel("")
        self.main_caption.setAlignment(Qt.AlignCenter)
        self.main_caption.setFont(QFont("Segoe UI", 12, QFont.DemiBold))

        self.main_preview = QLabel("No image loaded")
        self.main_preview.setAlignment(Qt.AlignCenter)
        self.main_preview.setFrameShape(QFrame.StyledPanel)
        self.main_preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.main_preview.setMinimumSize(800, 600)
        self.main_preview.setObjectName("MainPreview")

        # Thumbnails
        self.thumb_scroll = QScrollArea()
        self.thumb_scroll.setWidgetResizable(True)
        self.thumbs_container = QWidget()
        self.thumbs_layout = QHBoxLayout()
        self.thumbs_layout.setSpacing(20)
        self.thumbs_layout.setContentsMargins(10, 10, 10, 10)
        self.thumbs_container.setLayout(self.thumbs_layout)
        self.thumb_scroll.setWidget(self.thumbs_container)
        self.thumb_scroll.setFixedHeight(400)

    # ---- layout ----
    def _create_layout(self):
        # Left panel
        left_v = QVBoxLayout()
        left_v.setSpacing(15)
        left_v.addWidget(self.header)
        top_row = QHBoxLayout()
        top_row.addWidget(self.upload_btn)
        top_row.addStretch(1)
        top_row.addWidget(QLabel("Variants:"))
        top_row.addWidget(self.num_dropdown)
        left_v.addLayout(top_row)
        left_v.addWidget(self.label_doctor)
        left_v.addWidget(self.doctor_text)
        left_v.addWidget(self.label_patient)
        left_v.addWidget(self.patient_text)
        left_v.addWidget(self.generate_btn)
        left_v.addWidget(self.progress)
        left_v.addStretch(1)

        left_panel = QWidget()
        left_panel.setLayout(left_v)

        # Right panel
        right_v = QVBoxLayout()
        right_v.setSpacing(10)
        right_v.addWidget(self.main_caption)
        right_v.addWidget(self.main_preview, stretch=1)
        right_v.addWidget(self.thumb_scroll, stretch=0)

        right_panel = QWidget()
        right_panel.setLayout(right_v)

        # Splitter for balance
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 1000])  # left smaller, right dominant

        main_layout = QHBoxLayout(self)
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

    # ---- stylesheet ----
    def _apply_stylesheet(self):
        override = """
        QWidget { background-color: #003459; color: #FFFFFF; font-size: 13pt; }
        QLabel#HeaderLabel { font-size: 20pt; font-weight: 700; margin-bottom: 10px; }
        QLabel#MainPreview { border: 2px solid #007ea7; background-color: #002b45; }
        QTextEdit, QComboBox {
            background-color: #002b45; border: 1px solid #007ea7; border-radius: 8px;
        }
        QPushButton { background-color: #007ea7; border-radius: 8px; padding: 8px; }
        QProgressBar { border: 1px solid #007ea7; border-radius: 8px; text-align: center; }
        QProgressBar::chunk { background-color: #FFFFFF; }
        """
        self.setStyleSheet(override)

    # ---- utilities ----
    def set_busy(self, busy: bool):
        self.progress.setRange(0, 0 if busy else 100)
        if not busy:
            self.progress.setValue(100)

    def _set_main_pixmap(self, pix: QPixmap, caption: str):
        if not pix or pix.isNull():
            self.main_preview.setText("Preview unavailable")
            return
        self.current_main_pixmap = pix
        scaled = pix.scaled(self.main_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.main_preview.setPixmap(scaled)
        self.main_caption.setText(caption)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_main_pixmap:
            self._set_main_pixmap(self.current_main_pixmap, self.main_caption.text())

    # ---- interactions ----
    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Client Image", "", "Images (*.png *.jpg *.jpeg)")
        if not file_path:
            return
        self.client_image_path = file_path

        try:
            img = load_image(file_path)  # RGB numpy
            try:
                # prefer detect_landmarks to get the canonical aligned image (0 orientation)
                landmarks, aligned_img = detect_landmarks(img)
                logger.debug("upload_image: aligned via detect_landmarks.")
            except Exception as e:
                logger.debug("upload_image: detect_landmarks failed: %s. Falling back to correct_face_orientation.", e)
                aligned_img = correct_face_orientation(img)

            aligned_img = np.ascontiguousarray(aligned_img)
            pix = numpy_to_qpixmap(aligned_img)
            self._set_main_pixmap(pix, "Corrected Image")
            # store corrected pixmap for later thumbnail use
            self.corrected_pixmap = pix

        except Exception as e:
            logger.exception("Failed to load/align image for preview: %s", e)
            # fallback: show raw file
            pix = QPixmap(file_path)
            self._set_main_pixmap(pix, "Original Image (no face detected)")
            self.corrected_pixmap = None

    def on_generate_clicked(self):
        if not self.client_image_path:
            QMessageBox.warning(self, "Missing Image", "Please upload a client image first.")
            return

        doctor_notes = self.doctor_text.toPlainText().strip()
        patient_needs = self.patient_text.toPlainText().strip()
        num_outputs = int(self.num_dropdown.currentText())

        self.generate_btn.setEnabled(False)
        self.upload_btn.setEnabled(False)
        self.set_busy(True)
        self.clear_thumbnails()

        self.worker = GenerateWorker(self.client_image_path, doctor_notes, patient_needs, num_outputs)
        self.worker.finished.connect(self.on_generation_finished)
        self.worker.error.connect(self.on_generation_error)
        self.worker.start()

    def on_generation_finished(self, pixmaps: List[QPixmap], corrected_pixmap: Optional[QPixmap]):
        self.generate_btn.setEnabled(True)
        self.upload_btn.setEnabled(True)
        self.set_busy(False)
        self.generated_pixmaps = pixmaps
        self.corrected_pixmap = corrected_pixmap

        # Show corrected (upright) image in main preview first if available,
        # otherwise fall back to first generated result.
        if corrected_pixmap:
            self._set_main_pixmap(corrected_pixmap, "Corrected Image")
        elif pixmaps:
            self._set_main_pixmap(pixmaps[0], "Generated Result 1")
        else:
            self._set_main_pixmap(QPixmap(), "No preview available")

        # Populate thumbnails: corrected first (if present), then generated variants
        thumbs = []
        if corrected_pixmap:
            thumbs.append(corrected_pixmap)
        thumbs.extend(pixmaps)
        self._populate_thumbnails(thumbs)

    def on_generation_error(self, message: str):
        self.generate_btn.setEnabled(True)
        self.upload_btn.setEnabled(True)
        self.set_busy(False)
        QMessageBox.critical(self, "Generation Error", message)

    def clear_thumbnails(self):
        while self.thumbs_layout.count():
            item = self.thumbs_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _make_thumbnail_widget(self, pix: QPixmap, caption: str, is_original: bool):
        container = QWidget()
        v = QVBoxLayout(container)
        v.setSpacing(5)
        lbl = ClickableLabel()
        lbl.setFixedSize(QSize(200, 150))
        lbl.setPixmap(pix.scaled(lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        caption_lbl = QLabel(caption)
        caption_lbl.setAlignment(Qt.AlignCenter)

        # capture pix and caption at definition time to avoid late-binding issues
        lbl.clicked.connect(lambda pix=pix, caption=caption, is_original=is_original: self._set_main_pixmap(pix, caption if not is_original else "Corrected Image"))

        v.addWidget(lbl)
        v.addWidget(caption_lbl)
        return container

    def _populate_thumbnails(self, pixmaps: List[QPixmap]):
        self.clear_thumbnails()
        for i, pix in enumerate(pixmaps):
            if i == 0 and self.corrected_pixmap is not None:
                caption = "Corrected"
                is_orig = True
            else:
                # if corrected exists, generated start from index 1
                gen_index = i if self.corrected_pixmap is None else (i)
                caption = f"Gen {gen_index}" if not (i == 0 and self.corrected_pixmap is None) else "Original"
                is_orig = (i == 0 and self.corrected_pixmap is None)
            w = self._make_thumbnail_widget(pix, caption, is_original=is_orig)
            self.thumbs_layout.addWidget(w)
        self.thumbs_layout.addStretch(1)

    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showMaximized()
        else:
            self.showFullScreen()


# ---- entrypoint ----
def main():
    QLocale.setDefault(QLocale(QLocale.English, QLocale.UnitedStates))
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 12))
    win = SmileGeneratorUI()
    win.showMaximized()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
