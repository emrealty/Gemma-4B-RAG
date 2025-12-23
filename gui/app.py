from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets

from embeddings.embedder import (
    DEFAULT_EMBED_MODEL,
    embed_document,
)
from generator.gemma import MISSING_ANSWER, build_prompt, generate_with_gemma
from retriever import INDEX_VERSION
from retriever.faiss_store import FaissStore
from retriever.query import has_relevant_context, retrieve_context


INDEX_PATH = Path("embeddings/index.faiss")
METADATA_PATH = Path("embeddings/chunks.json")


class TaskThread(QtCore.QThread):
    finished_with_result = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def run(self) -> None:
        try:
            result = self._fn(*self._args, **self._kwargs)
        except Exception as e:  # noqa: BLE001 - show error to user
            self.failed.emit(str(e))
            return
        self.finished_with_result.emit(result)


def build_store_from_document(doc_path: Path, embed_model: str = DEFAULT_EMBED_MODEL) -> FaissStore:
    vectors, chunks = embed_document(doc_path, model=embed_model)
    store = FaissStore(dimension=vectors.shape[1])
    doc_hash = hashlib.sha256(doc_path.read_bytes()).hexdigest()
    store.add(vectors, [
        {
            "text": c.text,
            "source": c.source,
            "chunk_id": c.chunk_id,
            "doc_hash": doc_hash,
            "index_version": INDEX_VERSION,
            "embed_model": embed_model,
        }
        for c in chunks
    ])
    # Persist to disk for parity with CLI usage
    store.save(INDEX_PATH, METADATA_PATH)
    return store


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Gemma 3 4B RAG — PySide")
        self.resize(900, 600)

        self._store: Optional[FaissStore] = None
        self._current_doc: Optional[Path] = None
        self._embed_model = DEFAULT_EMBED_MODEL
        self._busy = False

        self._build_ui()

    # UI
    def _build_ui(self) -> None:
        cw = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(cw)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # Top: File controls
        file_row = QtWidgets.QHBoxLayout()
        self.file_label = QtWidgets.QLabel("Seçili dosya: (yok)")
        self.file_label.setMinimumWidth(400)
        self.btn_browse = QtWidgets.QPushButton("TXT Yükle…")
        self.btn_browse.clicked.connect(self.on_browse_clicked)
        self.status_label = QtWidgets.QLabel("Durum: indeks yok")
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)

        file_row.addWidget(self.file_label, 3)
        file_row.addWidget(self.btn_browse, 0)
        file_row.addWidget(self.status_label, 1)
        file_row.addWidget(self.progress, 1)
        root.addLayout(file_row)

        # Chat area
        self.chat_view = QtWidgets.QPlainTextEdit()
        self.chat_view.setReadOnly(True)
        self.chat_view.setPlaceholderText("Sohbet burada görünecek…")
        root.addWidget(self.chat_view, 1)

        # Input row
        input_row = QtWidgets.QHBoxLayout()
        self.input_edit = QtWidgets.QLineEdit()
        self.input_edit.setPlaceholderText("Sorunuzu yazın ve Enter'a basın…")
        self.input_edit.returnPressed.connect(self.on_send_clicked)
        self.btn_send = QtWidgets.QPushButton("Gönder")
        self.btn_send.clicked.connect(self.on_send_clicked)
        input_row.addWidget(self.input_edit, 1)
        input_row.addWidget(self.btn_send, 0)
        root.addLayout(input_row)

        self.setCentralWidget(cw)

    # Helpers
    def _set_busy(self, busy: bool, message: str | None = None) -> None:
        self._busy = busy
        self.btn_browse.setEnabled(not busy)
        self.btn_send.setEnabled(not busy)
        self.input_edit.setEnabled(not busy)
        self.progress.setVisible(busy)
        if message:
            self.status_label.setText(f"Durum: {message}")

    def append_chat(self, who: str, text: str) -> None:
        fmt = QtGui.QTextCharFormat()
        if who == "Kullanıcı":
            fmt.setForeground(QtGui.QBrush(QtGui.QColor("#1565c0")))
        else:
            fmt.setForeground(QtGui.QBrush(QtGui.QColor("#2e7d32")))

        cursor = self.chat_view.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(f"{who}: ", fmt)
        cursor.insertText(text + "\n")
        self.chat_view.ensureCursorVisible()

    # Slots
    def on_browse_clicked(self) -> None:
        if self._busy:
            return
        path_str, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "TXT belgesi seçin",
            str(Path.cwd() / "data"),
            "Metin Dosyaları (*.txt);;Tüm Dosyalar (*)",
        )
        if not path_str:
            return

        self._current_doc = Path(path_str)
        self.file_label.setText(f"Seçili dosya: {self._current_doc}")
        self._set_busy(True, "indeks oluşturuluyor…")

        self._index_thread = TaskThread(build_store_from_document, self._current_doc, self._embed_model)
        self._index_thread.finished_with_result.connect(self._on_index_ready)
        self._index_thread.failed.connect(self._on_error)
        self._index_thread.start()

    def _on_index_ready(self, store: FaissStore) -> None:
        self._store = store
        self._set_busy(False, "indeks hazır")
        self.status_label.setText("Durum: indeks hazır")
        self.append_chat("Sistem", "Belge yüklendi ve indeks hazır.")

    def _on_error(self, message: str) -> None:
        self._set_busy(False, "hata")
        QtWidgets.QMessageBox.critical(self, "Hata", message)

    def on_send_clicked(self) -> None:
        if self._busy:
            return
        question = self.input_edit.text().strip()
        if not question:
            return
        if self._store is None:
            QtWidgets.QMessageBox.information(self, "Bilgi", "Lütfen önce bir TXT dosyası yükleyin.")
            return

        self.append_chat("Kullanıcı", question)
        self.input_edit.clear()
        self._set_busy(True, "yanıt üretiliyor…")

        def _answer() -> str:
            context_chunks = retrieve_context(self._store, question, embed_model=self._embed_model)
            if not context_chunks or not has_relevant_context(question, context_chunks):
                return MISSING_ANSWER
            prompt = build_prompt(question, context_chunks)
            answer = generate_with_gemma(prompt)
            return answer

        self._gen_thread = TaskThread(_answer)
        self._gen_thread.finished_with_result.connect(self._on_answer_ready)
        self._gen_thread.failed.connect(self._on_error)
        self._gen_thread.start()

    def _on_answer_ready(self, answer: str) -> None:
        self._set_busy(False, "hazır")
        self.append_chat("Asistan", answer or "belgede böyle bir bilgi yok")


def run() -> None:
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run()
