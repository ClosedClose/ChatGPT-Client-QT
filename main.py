# -*- coding: utf-8 -*-

import os
import sys
import asyncio
import json
import httpx
from dataclasses import dataclass
from typing import List, Optional, Any, Iterator, Dict

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QTextEdit, QPushButton, QLabel, QComboBox, QSpinBox,
    QDoubleSpinBox, QListWidget, QListWidgetItem, QCheckBox, QMessageBox,
    QTabWidget, QFileDialog, QProgressBar
)

from qasync import QEventLoop, asyncSlot  # Qt + asyncio интеграция
from openai import OpenAI  # официальный SDK

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

PROFILES_FILE = "profiles.json"

# ---------------- Общие структуры ----------------

ROLES = ["system", "user", "assistant"]

@dataclass
class ChatMessage:
    role: str
    content: str

def extract_chat_text(resp: Any) -> str:
    # Универсальный парсер текста из ответов OpenAI/совместимых
    if isinstance(resp, dict) and "choices" in resp and resp.get("choices"):
        ch0 = resp["choices"][0] or {}
        msg = ch0.get("message") or {}
        content = msg.get("content")
        if content:
            return content
    if isinstance(resp, list) and resp:
        first = resp[0] or {}
        msg = first.get("message") or {}
        content = msg.get("content")
        if content:
            return content
    chs = getattr(resp, "choices", None)
    if chs and len(chs) > 0:
        msg = getattr(chs[0], "message", None)
        if msg:
            content = getattr(msg, "content", None)
            if content:
                return content
    output_text = getattr(resp, "output_text", None)
    if isinstance(output_text, str) and output_text:
        return output_text
    return ""

# ---------------- Хранилище профилей ----------------

def load_profiles() -> dict:
    if os.path.exists(PROFILES_FILE):
        try:
            with open(PROFILES_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"openai": {}, "genapi": {}}

def save_profiles(storage: dict) -> None:
    with open(PROFILES_FILE, "w", encoding="utf-8") as f:
        json.dump(storage, f, ensure_ascii=False, indent=2)

# ---------------- Клиент OpenAI Legacy ----------------

class OpenAIClient:
    def __init__(self, api_key: str, base_url: Optional[str] = None):
        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)  # поддерживает base_url для прокси

    async def chat_completion_stream(
        self,
        model: str,
        messages: List[ChatMessage],
        temperature: float,
        top_p: float,
        max_tokens: Optional[int],
    ) -> Iterator[str]:
        stream = self.client.chat.completions.create(
            model=model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens or None,
            stream=True,
        )
        for event in stream:
            try:
                delta = getattr(event.choices[0], "delta", None)
                if delta:
                    piece = getattr(delta, "content", None)
                    if piece:
                        yield piece
                        continue
            except Exception:
                pass
            try:
                ch0 = getattr(event, "choices", [None])[0]
                msg = getattr(ch0, "message", None)
                if msg:
                    piece = getattr(msg, "content", None)
                    if piece:
                        yield piece
                        continue
            except Exception:
                pass
            if isinstance(event, dict) and "choices" in event and event["choices"]:
                ch0 = event["choices"][0] or {}
                delta = ch0.get("delta") or {}
                piece = delta.get("content")
                if piece:
                    yield piece
                    continue
                msg = ch0.get("message") or {}
                piece = msg.get("content")
                if piece:
                    yield piece
                    continue
            msg = event.get("message") if isinstance(event, dict) else None
            if isinstance(msg, dict) and "content" in msg:
                yield msg["content"]

    async def chat_completion_once(
        self,
        model: str,
        messages: List[ChatMessage],
        temperature: float,
        top_p: float,
        max_tokens: Optional[int],
    ) -> str:
        comp = self.client.chat.completions.create(
            model=model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens or None,
        )
        try:
            text = extract_chat_text(comp)
            if text:
                return text
        except Exception:
            pass
        try:
            as_dict = comp.to_dict()
            return extract_chat_text(as_dict)
        except Exception:
            return extract_chat_text(comp)

    async def responses_once(
        self,
        model: str,
        instructions: str,
        input_text: str,
        temperature: float,
        top_p: float,
        max_tokens: Optional[int],
    ) -> str:
        resp = self.client.responses.create(
            model=model,
            instructions=instructions or None,
            input=input_text,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens or None,
        )
        return extract_chat_text(resp)

# ---------------- Клиент GenAPI (gpt-5) ----------------

class GenAPIClient:
    """
    Прямая интеграция с GenAPI: авторизация Bearer, запуск генерации и получение результата.
    Поддерживает:
    - Sync (is_sync=true)
    - Long-polling (start -> task/status/{id})
    Документация “Схема работы” (авторизация, long-poll, сразу ответ). 
    """
    def __init__(self, api_key: str, base_url: str = "https://api.gen-api.ru"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.network_url = f"{self.base_url}/api/v1/networks/gpt-5"

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def generate_sync(self, messages: List[Dict[str, str]], parameters: Dict[str, Any]) -> Dict[str, Any]:
        payload = {"messages": messages, "parameters": parameters, "is_sync": True}
        timeout = httpx.Timeout(60.0, read=120.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(self.network_url, headers=self._headers(), json=payload)
            if resp.status_code != 200:
                raise RuntimeError(f"GenAPI sync error: {resp.status_code} {resp.text}")
            try:
                return resp.json()
            except Exception:
                return {"result": resp.text}

    async def generate_long_poll(
        self,
        messages: List[Dict[str, str]],
        parameters: Dict[str, Any],
        poll_interval_sec: float = 2.0,
        max_wait_sec: float = 120.0,
        progress_cb=None,
        log_cb=None,
    ) -> Dict[str, Any]:
        start_payload = {"messages": messages, "parameters": parameters}
        timeout = httpx.Timeout(60.0, read=120.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            if log_cb:
                log_cb("GenAPI start", {"url": self.network_url})
            start = await client.post(self.network_url, headers=self._headers(), json=start_payload)
            if start.status_code != 200:
                raise RuntimeError(f"GenAPI start error: {start.status_code} {start.text}")

            data = start.json() if start.headers.get("content-type", "").startswith("application/json") else {}
            request_id = data.get("request_id")
            if not request_id:
                if "result" in data:
                    return data
                raise RuntimeError("GenAPI: no request_id in start response")

            status_url = f"{self.base_url}/task/status/{request_id}"
            elapsed = 0.0
            while elapsed < max_wait_sec:
                await asyncio.sleep(poll_interval_sec)
                elapsed += poll_interval_sec
                if progress_cb:
                    # прогресс неизвестен — показываем спиннер (индикатор занятости)
                    progress_cb(True)
                if log_cb:
                    log_cb("GenAPI poll", {"url": status_url, "elapsed": elapsed})
                s = await client.get(status_url, headers=self._headers())
                if s.status_code != 200:
                    raise RuntimeError(f"GenAPI status error: {s.status_code} {s.text}")
                status_data = s.json() if s.headers.get("content-type", "").startswith("application/json") else {}
                st = status_data.get("status")
                if st == "SUCCESS":
                    if progress_cb:
                        progress_cb(False)
                    return status_data.get("result") if "result" in status_data else status_data
                if st in ("FAILURE", "CANCELED"):
                    if progress_cb:
                        progress_cb(False)
                    raise RuntimeError(f"GenAPI task failed: {st}")
            if progress_cb:
                progress_cb(False)
            raise TimeoutError("GenAPI long-poll timeout")

# ---------------- UI с вкладками ----------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ChatGPT-Client-QT (GenAPI/OpenAI)")
        self.resize(1180, 820)

        self.tabs = QTabWidget()
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.addWidget(self.tabs)

        # Нижняя панель: лог
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("Лог событий...")
        log_buttons = QHBoxLayout()
        self.btn_log_clear = QPushButton("Очистить лог")
        self.btn_log_save = QPushButton("Сохранить лог")
        log_buttons.addWidget(self.btn_log_clear)
        log_buttons.addWidget(self.btn_log_save)

        center_layout.addLayout(log_buttons)
        center_layout.addWidget(self.log_view, 2)

        self.setCentralWidget(center)

        # Прогресс-бар в статус-баре
        self.status_progress = QProgressBar()
        self.status_progress.setRange(0, 0)  # режим "занято"
        self.status_progress.setVisible(False)
        self.statusBar().addPermanentWidget(self.status_progress)

        # Вкладки
        self.legacy_tab = self._build_openai_legacy_tab()
        self.genapi_tab = self._build_genapi_tab()
        self.tabs.addTab(self.legacy_tab, "OpenAI Legacy")
        self.tabs.addTab(self.genapi_tab, "GenAPI (gpt-5)")

        # Лог-кнопки
        self.btn_log_clear.clicked.connect(lambda: self.log_view.clear())
        self.btn_log_save.clicked.connect(self.on_save_log)

    # ---------- лог-хелперы ----------
    def log_info(self, msg: str, data: Optional[dict] = None):
        line = {"level": "INFO", "msg": msg, "data": data or {}}
        self.log_view.append(json.dumps(line, ensure_ascii=False))

    def log_error(self, msg: str, data: Optional[dict] = None):
        line = {"level": "ERROR", "msg": msg, "data": data or {}}
        self.log_view.append(json.dumps(line, ensure_ascii=False))

    def on_save_log(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Сохранить лог", "", "Text (*.txt);;JSON (*.json);;All (*.*)")
        if not fn:
            return
        with open(fn, "w", encoding="utf-8") as f:
            f.write(self.log_view.toPlainText())

    # ---------- вкладка OpenAI Legacy ----------
    def _build_openai_legacy_tab(self) -> QWidget:
        root = QWidget()
        layout = QVBoxLayout(root)

        top = QWidget()
        top_layout = QHBoxLayout(top)

        # Профили
        prof_row = QHBoxLayout()
        self.oa_profiles_store = load_profiles()
        self.oa_profile_box = QComboBox()
        self.oa_profile_box.setEditable(True)
        self.oa_profile_box.addItems(sorted(self.oa_profiles_store["openai"].keys()))
        btn_prof_save = QPushButton("Save")
        btn_prof_del = QPushButton("Delete")
        btn_prof_apply = QPushButton("Load")
        prof_row.addWidget(QLabel("Profile:"))
        prof_row.addWidget(self.oa_profile_box, 1)
        prof_row.addWidget(btn_prof_apply)
        prof_row.addWidget(btn_prof_save)
        prof_row.addWidget(btn_prof_del)

        self.api_key_edit = QLineEdit(os.getenv("OPENAI_API_KEY", ""))
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        self.api_key_edit.setPlaceholderText("OPENAI_API_KEY")

        self.base_url_edit = QLineEdit(os.getenv("OPENAI_BASE_URL", ""))
        self.base_url_edit.setPlaceholderText("https://api.openai.com/v1 или совместимый URL")

        self.model_box = QComboBox()
        self.model_box.addItems(["gpt-4o", "gpt-4o-mini", "gpt-4.1-mini"])

        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.0, 2.0)
        self.temperature_spin.setSingleStep(0.1)
        self.temperature_spin.setValue(0.7)

        self.top_p_spin = QDoubleSpinBox()
        self.top_p_spin.setRange(0.0, 1.0)
        self.top_p_spin.setSingleStep(0.05)
        self.top_p_spin.setValue(1.0)

        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(0, 32768)
        self.max_tokens_spin.setValue(0)

        self.stream_check = QCheckBox("Stream")
        self.stream_check.setChecked(True)

        top_layout.addLayout(prof_row, 1)
        top_layout.addWidget(QLabel("API Key:"))
        top_layout.addWidget(self.api_key_edit, 2)
        top_layout.addWidget(QLabel("Base URL:"))
        top_layout.addWidget(self.base_url_edit, 2)
        top_layout.addWidget(QLabel("Model:"))
        top_layout.addWidget(self.model_box, 1)
        top_layout.addWidget(QLabel("Temperature:"))
        top_layout.addWidget(self.temperature_spin, 1)
        top_layout.addWidget(QLabel("Top-p:"))
        top_layout.addWidget(self.top_p_spin, 1)
        top_layout.addWidget(QLabel("Max tokens:"))
        top_layout.addWidget(self.max_tokens_spin, 1)
        top_layout.addWidget(self.stream_check)

        layout.addWidget(top)

        # Тело диалога
        self.system_edit = QTextEdit()
        self.system_edit.setPlaceholderText("Системная инструкция (system role)")

        self.history_list = QListWidget()
        self.user_edit = QTextEdit()
        self.user_edit.setPlaceholderText("Введите сообщение пользователя")

        # Вложения
        attach_bar = QHBoxLayout()
        self.attach_list = QListWidget()
        btn_attach = QPushButton("Прикрепить файлы")
        btn_attach_clear = QPushButton("Очистить вложения")
        attach_bar.addWidget(btn_attach)
        attach_bar.addWidget(btn_attach_clear)
        self.attach_files: List[str] = []

        layout.addWidget(QLabel("System:"))
        layout.addWidget(self.system_edit)
        layout.addWidget(QLabel("История сообщений (двойной клик — редактирование):"))
        layout.addWidget(self.history_list, 3)
        layout.addWidget(QLabel("User message:"))
        layout.addWidget(self.user_edit, 1)
        layout.addLayout(attach_bar)
        layout.addWidget(QLabel("Вложения:"))
        layout.addWidget(self.attach_list, 1)

        btn_row = QHBoxLayout()
        self.add_user_btn = QPushButton("Добавить в историю как user")
        self.add_assistant_btn = QPushButton("Добавить как assistant")
        self.clear_btn = QPushButton("Очистить историю")
        self.copy_btn = QPushButton("Копировать историю")
        self.export_btn = QPushButton("Экспорт истории")
        self.import_btn = QPushButton("Импорт истории")
        btn_row.addWidget(self.add_user_btn)
        btn_row.addWidget(self.add_assistant_btn)
        btn_row.addWidget(self.clear_btn)
        btn_row.addWidget(self.copy_btn)
        btn_row.addWidget(self.export_btn)
        btn_row.addWidget(self.import_btn)
        layout.addLayout(btn_row)

        send_row = QHBoxLayout()
        self.api_mode = QComboBox()
        self.api_mode.addItems(["Chat Completions", "Responses"])
        self.send_btn = QPushButton("Отправить")
        send_row.addWidget(QLabel("API:"))
        send_row.addWidget(self.api_mode)
        send_row.addWidget(self.send_btn)
        layout.addLayout(send_row)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        layout.addWidget(QLabel("Ответ ассистента:"))
        layout.addWidget(self.output, 3)

        # Сигналы
        self.add_user_btn.clicked.connect(self.on_add_user)
        self.add_assistant_btn.clicked.connect(self.on_add_assistant)
        self.clear_btn.clicked.connect(self.history_list.clear)
        self.copy_btn.clicked.connect(lambda: self.copy_history(self.history_list))
        self.export_btn.clicked.connect(lambda: self.export_history(self.history_list))
        self.import_btn.clicked.connect(lambda: self.import_history(self.history_list))
        self.history_list.itemDoubleClicked.connect(self.on_edit_item)
        self.send_btn.clicked.connect(self.on_send_clicked)
        btn_attach.clicked.connect(self.on_attach_files)
        btn_attach_clear.clicked.connect(lambda: self._set_attachments([]))

        # Профили — обработчики
        btn_prof_save.clicked.connect(self.on_oa_save_profile)
        btn_prof_del.clicked.connect(self.on_oa_delete_profile)
        btn_prof_apply.clicked.connect(self.on_oa_load_profile)

        return root

    # ---------- вкладка GenAPI ----------
    def _build_genapi_tab(self) -> QWidget:
        root = QWidget()
        layout = QVBoxLayout(root)

        top = QWidget()
        top_layout = QHBoxLayout(top)

        # Профили
        prof_row = QHBoxLayout()
        self.ga_profiles_store = load_profiles()
        self.ga_profile_box = QComboBox()
        self.ga_profile_box.setEditable(True)
        self.ga_profile_box.addItems(sorted(self.ga_profiles_store["genapi"].keys()))
        ga_btn_apply = QPushButton("Load")
        ga_btn_save = QPushButton("Save")
        ga_btn_del = QPushButton("Delete")
        prof_row.addWidget(QLabel("Profile:"))
        prof_row.addWidget(self.ga_profile_box, 1)
        prof_row.addWidget(ga_btn_apply)
        prof_row.addWidget(ga_btn_save)
        prof_row.addWidget(ga_btn_del)

        self.genapi_key_edit = QLineEdit(os.getenv("GENAPI_KEY", ""))
        self.genapi_key_edit.setEchoMode(QLineEdit.Password)
        self.genapi_key_edit.setPlaceholderText("GENAPI API KEY (Bearer)")

        self.genapi_base_edit = QLineEdit(os.getenv("GENAPI_BASE_URL", "https://api.gen-api.ru"))
        self.genapi_base_edit.setPlaceholderText("https://api.gen-api.ru")

        self.genapi_model_box = QComboBox()
        self.genapi_model_box.addItems(["gpt-5"])  # фиксированный выбор

        self.genapi_temperature = QDoubleSpinBox()
        self.genapi_temperature.setRange(0.0, 2.0)
        self.genapi_temperature.setSingleStep(0.1)
        self.genapi_temperature.setValue(0.7)

        self.genapi_top_p = QDoubleSpinBox()
        self.genapi_top_p.setRange(0.0, 1.0)
        self.genapi_top_p.setSingleStep(0.05)
        self.genapi_top_p.setValue(1.0)

        self.genapi_max_tokens = QSpinBox()
        self.genapi_max_tokens.setRange(0, 32768)
        self.genapi_max_tokens.setValue(0)

        self.genapi_sync = QCheckBox("Sync (is_sync=true)")
        self.genapi_sync.setChecked(True)

        top_layout.addLayout(prof_row, 1)
        top_layout.addWidget(QLabel("API Key:"))
        top_layout.addWidget(self.genapi_key_edit, 2)
        top_layout.addWidget(QLabel("Base URL:"))
        top_layout.addWidget(self.genapi_base_edit, 2)
        top_layout.addWidget(QLabel("Model:"))
        top_layout.addWidget(self.genapi_model_box, 1)
        top_layout.addWidget(QLabel("Temperature:"))
        top_layout.addWidget(self.genapi_temperature, 1)
        top_layout.addWidget(QLabel("Top-p:"))
        top_layout.addWidget(self.genapi_top_p, 1)
        top_layout.addWidget(QLabel("Max tokens:"))
        top_layout.addWidget(self.genapi_max_tokens, 1)
        top_layout.addWidget(self.genapi_sync)

        layout.addWidget(top)

        # Сообщения и вложения
        self.genapi_system = QTextEdit()
        self.genapi_system.setPlaceholderText("System (опционально)")
        self.genapi_history = QListWidget()
        self.genapi_user = QTextEdit()
        self.genapi_user.setPlaceholderText("User message")

        ga_attach_bar = QHBoxLayout()
        self.ga_attach_list = QListWidget()
        self.ga_attach_files: List[str] = []
        ga_btn_attach = QPushButton("Прикрепить файлы")
        ga_btn_attach_clear = QPushButton("Очистить вложения")
        ga_attach_bar.addWidget(ga_btn_attach)
        ga_attach_bar.addWidget(ga_btn_attach_clear)

        layout.addWidget(QLabel("System:"))
        layout.addWidget(self.genapi_system)
        layout.addWidget(QLabel("История сообщений:"))
        layout.addWidget(self.genapi_history, 3)
        layout.addWidget(QLabel("User message:"))
        layout.addWidget(self.genapi_user, 1)
        layout.addLayout(ga_attach_bar)
        layout.addWidget(QLabel("Вложения:"))
        layout.addWidget(self.ga_attach_list, 1)

        row = QHBoxLayout()
        self.genapi_add_user = QPushButton("Добавить user")
        self.genapi_add_assistant = QPushButton("Добавить assistant")
        self.genapi_clear = QPushButton("Очистить историю")
        self.genapi_copy = QPushButton("Копировать историю")
        self.genapi_export = QPushButton("Экспорт истории")
        self.genapi_import = QPushButton("Импорт истории")
        row.addWidget(self.genapi_add_user)
        row.addWidget(self.genapi_add_assistant)
        row.addWidget(self.genapi_clear)
        row.addWidget(self.genapi_copy)
        row.addWidget(self.genapi_export)
        row.addWidget(self.genapi_import)
        layout.addLayout(row)

        send_row = QHBoxLayout()
        self.genapi_send = QPushButton("Отправить в GenAPI (gpt-5)")
        send_row.addWidget(self.genapi_send)
        layout.addLayout(send_row)

        self.genapi_output = QTextEdit()
        self.genapi_output.setReadOnly(True)
        layout.addWidget(QLabel("Ответ (GenAPI):"))
        layout.addWidget(self.genapi_output, 3)

        # Сигналы
        self.genapi_add_user.clicked.connect(lambda: self._genapi_add("user"))
        self.genapi_add_assistant.clicked.connect(lambda: self._genapi_add("assistant"))
        self.genapi_clear.clicked.connect(self.genapi_history.clear)
        self.genapi_copy.clicked.connect(lambda: self.copy_history(self.genapi_history))
        self.genapi_export.clicked.connect(lambda: self.export_history(self.genapi_history))
        self.genapi_import.clicked.connect(lambda: self.import_history(self.genapi_history))
        self.genapi_send.clicked.connect(self.on_genapi_send)
        ga_btn_attach.clicked.connect(self.on_genapi_attach_files)
        ga_btn_attach_clear.clicked.connect(lambda: self._ga_set_attachments([]))

        ga_btn_save.clicked.connect(self.on_ga_save_profile)
        ga_btn_del.clicked.connect(self.on_ga_delete_profile)
        ga_btn_apply.clicked.connect(self.on_ga_load_profile)

        return root

    # ---------- общие вспомогательные ----------
    def _add_history_item(self, list_widget: QListWidget, role: str, content: str):
        item = QListWidgetItem(f"{role}: {content[:80]}")
        item.setData(Qt.UserRole, (role, content))
        list_widget.addItem(item)

    def _prompt_multiline(self, title: str, default: str):
        dlg = QTextEdit()
        dlg.setPlainText(default)
        box = QMessageBox(self)
        box.setWindowTitle(title)
        box.setIcon(QMessageBox.Question)
        box.setText("Измените текст и нажмите OK")
        box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        box.layout().addWidget(dlg, 1, 0, 1, box.layout().columnCount())
        res = box.exec()
        return dlg.toPlainText(), (res == QMessageBox.Ok)

    # ---------- обработчики вкладки OpenAI Legacy ----------
    def build_messages(self) -> List[ChatMessage]:
        messages: List[ChatMessage] = []
        system_text = self.system_edit.toPlainText().strip()
        if system_text:
            messages.append(ChatMessage("system", system_text))
        for i in range(self.history_list.count()):
            item = self.history_list.item(i)
            role, content = item.data(Qt.UserRole)
            messages.append(ChatMessage(role, content))
        user_now = self.user_edit.toPlainText().strip()
        if user_now:
            messages.append(ChatMessage("user", user_now))
        # Вложения — дописываем в конец user-сообщения
        if self.attach_files:
            note = "\n\nAttachments:\n" + "\n".join(f"- {p}" for p in self.attach_files)
            if messages and messages[-1].role == "user":
                messages[-1].content += note
            else:
                messages.append(ChatMessage("user", note))
        return messages

    def on_add_user(self):
        txt = self.user_edit.toPlainText().strip()
        if not txt:
            return
        self._add_history_item(self.history_list, "user", txt)
        self.user_edit.clear()

    def on_add_assistant(self):
        txt = self.user_edit.toPlainText().strip()
        if not txt:
            return
        self._add_history_item(self.history_list, "assistant", txt)
        self.user_edit.clear()

    def on_edit_item(self, item: QListWidgetItem):
        role, content = item.data(Qt.UserRole)
        new_text, ok = self._prompt_multiline(f"Редактировать ({role})", content)
        if ok:
            item.setData(Qt.UserRole, (role, new_text))
            item.setText(f"{role}: {new_text[:80]}")

    def _build_openai_client(self) -> OpenAIClient:
        api_key = self.api_key_edit.text().strip() or os.getenv("OPENAI_API_KEY", "")
        base_url = self.base_url_edit.text().strip() or os.getenv("OPENAI_BASE_URL", "")
        if not api_key:
            QMessageBox.warning(self, "Внимание", "Укажите API Key")
        return OpenAIClient(api_key=api_key, base_url=base_url if base_url else None)

    @asyncSlot()
    async def on_send_clicked(self):
        self.output.clear()
        client = self._build_openai_client()
        model = self.model_box.currentText()
        temperature = float(self.temperature_spin.value())
        top_p = float(self.top_p_spin.value())
        max_tokens = int(self.max_tokens_spin.value()) or None
        api_mode = self.api_mode.currentText()

        try:
            self.status_progress.setVisible(True)
            if api_mode == "Chat Completions":
                messages = self.build_messages()
                if self.stream_check.isChecked():
                    async for token in self._stream_chat(client, model, messages, temperature, top_p, max_tokens):
                        self.output.moveCursor(self.output.textCursor().End)
                        self.output.insertPlainText(token)
                        QApplication.processEvents()
                else:
                    text = await client.chat_completion_once(model, messages, temperature, top_p, max_tokens)
                    self.output.setPlainText(text)
                self._append_assistant_to_history(self.output, self.history_list)
            else:
                system_text = self.system_edit.toPlainText().strip()
                ctx = []
                for i in range(self.history_list.count()):
                    role, content = self.history_list.item(i).data(Qt.UserRole)
                    ctx.append(f"{role.upper()}: {content}")
                current_user = self.user_edit.toPlainText().strip()
                if current_user:
                    ctx.append(f"USER: {current_user}")
                input_text = "\n\n".join(ctx)
                text = await client.responses_once(
                    model=model,
                    instructions=system_text,
                    input_text=input_text,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
                self.output.setPlainText(text)
                self._append_assistant_to_history(self.output, self.history_list)
            self.log_info("OpenAI request done", {"mode": api_mode})
            self.on_oa_save_profile(auto=True)
        except Exception as e:
            self.log_error("OpenAI error", {"err": str(e)})
            QMessageBox.critical(self, "Ошибка", str(e))
        finally:
            self.status_progress.setVisible(False)

    async def _stream_chat(self, client: OpenAIClient, model, messages, temperature, top_p, max_tokens):
        async for token in client.chat_completion_stream(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        ):
            yield token

    # Вложения (Legacy)
    def on_attach_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Прикрепить файлы", "", 
            "All Files (*);;Images (*.png *.jpg *.jpeg *.gif);;Text (*.txt *.md)")
        if files:
            self.attach_files.extend(files)
            self._refresh_attach_list()

    def _set_attachments(self, files: List[str]):
        self.attach_files = files
        self._refresh_attach_list()

    def _refresh_attach_list(self):
        self.attach_list.clear()
        for f in self.attach_files:
            self.attach_list.addItem(f)

    # Профили OpenAI
    def on_oa_load_profile(self):
        name = self.oa_profile_box.currentText().strip()
        store = load_profiles()
        prof = store["openai"].get(name)
        if not prof:
            return
        self.api_key_edit.setText(prof.get("api_key", ""))
        self.base_url_edit.setText(prof.get("base_url", ""))
        self.model_box.setCurrentText(prof.get("model", self.model_box.currentText()))
        self.temperature_spin.setValue(float(prof.get("temperature", self.temperature_spin.value())))
        self.top_p_spin.setValue(float(prof.get("top_p", self.top_p_spin.value())))
        self.max_tokens_spin.setValue(int(prof.get("max_tokens", self.max_tokens_spin.value())))
        self.stream_check.setChecked(bool(prof.get("stream", self.stream_check.isChecked())))
        self.log_info("Profile loaded", {"name": name, "scope": "openai"})

    def on_oa_save_profile(self, auto: bool = False):
        name = self.oa_profile_box.currentText().strip() or "default"
        p = {
            "api_key": self.api_key_edit.text(),
            "base_url": self.base_url_edit.text(),
            "model": self.model_box.currentText(),
            "temperature": self.temperature_spin.value(),
            "top_p": self.top_p_spin.value(),
            "max_tokens": self.max_tokens_spin.value(),
            "stream": self.stream_check.isChecked(),
        }
        store = load_profiles()
        store["openai"][name] = p
        save_profiles(store)
        if self.oa_profile_box.findText(name) < 0:
            self.oa_profile_box.addItem(name)
        if not auto:
            self.log_info("Profile saved", {"name": name, "scope": "openai"})

    def on_oa_delete_profile(self):
        name = self.oa_profile_box.currentText().strip()
        store = load_profiles()
        if name in store["openai"]:
            del store["openai"][name]
            save_profiles(store)
            idx = self.oa_profile_box.findText(name)
            if idx >= 0:
                self.oa_profile_box.removeItem(idx)
            self.log_info("Profile deleted", {"name": name, "scope": "openai"})

    # Экспорт/импорт/копирование истории
    def export_history(self, list_widget: QListWidget):
        fn, _ = QFileDialog.getSaveFileName(self, "Сохранить историю", "", "JSON (*.json)")
        if not fn:
            return
        data = []
        for i in range(list_widget.count()):
            role, content = list_widget.item(i).data(Qt.UserRole)
            data.append({"role": role, "content": content})
        with open(fn, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self.log_info("History exported", {"file": fn})

    def import_history(self, list_widget: QListWidget):
        fn, _ = QFileDialog.getOpenFileName(self, "Импорт истории", "", "JSON (*.json)")
        if not fn:
            return
        with open(fn, "r", encoding="utf-8") as f:
            data = json.load(f)
        list_widget.clear()
        for m in data:
            role = m.get("role", "user")
            content = m.get("content", "")
            self._add_history_item(list_widget, role, content)
        self.log_info("History imported", {"file": fn})

    def copy_history(self, list_widget: QListWidget):
        lines = []
        for i in range(list_widget.count()):
            role, content = list_widget.item(i).data(Qt.UserRole)
            lines.append(f"{role.upper()}: {content}")
        QApplication.clipboard().setText("\n\n".join(lines))
        self.log_info("History copied", {"lines": len(lines)})

    # ---------- обработчики вкладки GenAPI ----------
    def _genapi_collect_messages(self) -> List[Dict[str, str]]:
        msgs: List[Dict[str, str]] = []
        sys_text = self.genapi_system.toPlainText().strip()
        if sys_text:
            msgs.append({"role": "system", "content": sys_text})
        for i in range(self.genapi_history.count()):
            role, content = self.genapi_history.item(i).data(Qt.UserRole)
            msgs.append({"role": role, "content": content})
        user_text = self.genapi_user.toPlainText().strip()
        if user_text:
            msgs.append({"role": "user", "content": user_text})
        # Вложения (примеры — как список путей)
        if self.ga_attach_files:
            note = "\n\nAttachments:\n" + "\n".join(f"- {p}" for p in self.ga_attach_files)
            if msgs and msgs[-1]["role"] == "user":
                msgs[-1]["content"] += note
            else:
                msgs.append({"role": "user", "content": note})
        return msgs

    def _genapi_add(self, role: str):
        src = self.genapi_user
        txt = src.toPlainText().strip()
        if not txt:
            return
        self._add_history_item(self.genapi_history, role, txt)
        src.clear()

    @asyncSlot()
    async def on_genapi_send(self):
        self.genapi_output.clear()
        api_key = self.genapi_key_edit.text().strip()
        if not api_key:
            QMessageBox.warning(self, "Внимание", "Укажите GenAPI API Key (Bearer)")
            return
        base_url = self.genapi_base_edit.text().strip() or "https://api.gen-api.ru"
        client = GenAPIClient(api_key=api_key, base_url=base_url)

        parameters = {}
        if self.genapi_max_tokens.value() > 0:
            parameters["max_tokens"] = int(self.genapi_max_tokens.value())
        parameters["temperature"] = float(self.genapi_temperature.value())
        parameters["top_p"] = float(self.genapi_top_p.value())

        messages = self._genapi_collect_messages()
        if not messages:
            QMessageBox.warning(self, "Внимание", "Пустые messages")
            return

        try:
            self.status_progress.setVisible(True)
            if self.genapi_sync.isChecked():
                data = await client.generate_sync(messages, parameters)
            else:
                data = await client.generate_long_poll(
                    messages, parameters,
                    progress_cb=lambda busy: self.status_progress.setVisible(busy),
                    log_cb=lambda msg, d=None: self.log_info(msg, d),
                )
            text = self._genapi_extract_text(data)
            self.genapi_output.setPlainText(text)
            self._append_assistant_to_history(self.genapi_output, self.genapi_history)
            self.log_info("GenAPI request done", {"sync": self.genapi_sync.isChecked()})
            self.on_ga_save_profile(auto=True)
        except Exception as e:
            self.log_error("GenAPI error", {"err": str(e)})
            QMessageBox.critical(self, "GenAPI ошибка", str(e))
        finally:
            self.status_progress.setVisible(False)

    def _genapi_extract_text(self, data: Any) -> str:
        if isinstance(data, dict):
            if isinstance(data.get("result"), str):
                return data["result"]
            if isinstance(data.get("choices"), list) and data["choices"]:
                ch0 = data["choices"][0] or {}
                msg = ch0.get("message") or {}
                if isinstance(msg.get("content"), str):
                    return msg["content"]
        if isinstance(data, list) and data:
            first = data[0] or {}
            msg = first.get("message") or {}
            if isinstance(msg.get("content"), str):
                return msg["content"]
        try:
            return json.dumps(data, ensure_ascii=False, indent=2)
        except Exception:
            return str(data)

    # Вложения (GenAPI)
    def on_genapi_attach_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Прикрепить файлы", "", 
            "All Files (*);;Images (*.png *.jpg *.jpeg *.gif);;Text (*.txt *.md)")
        if files:
            self.ga_attach_files.extend(files)
            self._ga_refresh_attach_list()

    def _ga_set_attachments(self, files: List[str]):
        self.ga_attach_files = files
        self._ga_refresh_attach_list()

    def _ga_refresh_attach_list(self):
        self.ga_attach_list.clear()
        for f in self.ga_attach_files:
            self.ga_attach_list.addItem(f)

    # Профили GenAPI
    def on_ga_load_profile(self):
        name = self.ga_profile_box.currentText().strip()
        store = load_profiles()
        prof = store["genapi"].get(name)
        if not prof:
            return
        self.genapi_key_edit.setText(prof.get("api_key", ""))
        self.genapi_base_edit.setText(prof.get("base_url", "https://api.gen-api.ru"))
        self.genapi_model_box.setCurrentText(prof.get("model", "gpt-5"))
        self.genapi_temperature.setValue(float(prof.get("temperature", self.genapi_temperature.value())))
        self.genapi_top_p.setValue(float(prof.get("top_p", self.genapi_top_p.value())))
        self.genapi_max_tokens.setValue(int(prof.get("max_tokens", self.genapi_max_tokens.value())))
        self.genapi_sync.setChecked(bool(prof.get("sync", True)))
        self.log_info("Profile loaded", {"name": name, "scope": "genapi"})

    def on_ga_save_profile(self, auto: bool = False):
        name = self.ga_profile_box.currentText().strip() or "default"
        p = {
            "api_key": self.genapi_key_edit.text(),
            "base_url": self.genapi_base_edit.text(),
            "model": self.genapi_model_box.currentText(),
            "temperature": self.genapi_temperature.value(),
            "top_p": self.genapi_top_p.value(),
            "max_tokens": self.genapi_max_tokens.value(),
            "sync": self.genapi_sync.isChecked(),
        }
        store = load_profiles()
        store["genapi"][name] = p
        save_profiles(store)
        if self.ga_profile_box.findText(name) < 0:
            self.ga_profile_box.addItem(name)
        if not auto:
            self.log_info("Profile saved", {"name": name, "scope": "genapi"})

    def on_ga_delete_profile(self):
        name = self.ga_profile_box.currentText().strip()
        store = load_profiles()
        if name in store["genapi"]:
            del store["genapi"][name]
            save_profiles(store)
            idx = self.ga_profile_box.findText(name)
            if idx >= 0:
                self.ga_profile_box.removeItem(idx)
            self.log_info("Profile deleted", {"name": name, "scope": "genapi"})

    # ---------- общие ----------
    def _append_assistant_to_history(self, output_edit: QTextEdit, list_widget: QListWidget):
        txt = output_edit.toPlainText().strip()
        if txt:
            self._add_history_item(list_widget, "assistant", txt)

# ---------------- Точка входа ----------------

def main():
    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)
    w = MainWindow()
    w.show()
    with loop:
        loop.run_forever()

if __name__ == "__main__":
    main()
