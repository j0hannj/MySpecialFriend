"""
agent/notifications.py — Système de notifications multi-backend.
Console, fichier, Telegram, Discord.

Usage:
    from agent.notifications import get_notifier, notify
    
    notifier = get_notifier()  # Uses config
    notifier.send("Training complete!", event="milestone")
    
    # Or use the global notify function
    notify("New fact learned!", event="learning")
"""
import json
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import LOG_DIR


class NotifConfig:
    """Configuration for notifications."""
    backend: str = "console"
    telegram_token: str = ""
    telegram_chat_id: str = ""
    discord_webhook: str = ""
    min_importance: int = 1  # 1=info, 2=milestone, 3=error
    
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)


NOTIF_CFG = NotifConfig()

EVENT_IMPORTANCE = {
    "info": 1,
    "learning": 1,
    "progress": 1,
    "milestone": 2,
    "checkpoint": 2,
    "warning": 2,
    "error": 3,
    "critical": 3,
    "session_end": 2,
}


class BaseNotifier(ABC):
    """Abstract base class for notification backends."""
    
    def __init__(self, config: NotifConfig = None):
        self.config = config or NOTIF_CFG
    
    @abstractmethod
    def send(self, message: str, event: str = "info", extra: Dict[str, Any] = None) -> bool:
        """Send a notification. Returns True if successful."""
        pass
    
    def should_send(self, event: str) -> bool:
        """Check if this event meets minimum importance threshold."""
        importance = EVENT_IMPORTANCE.get(event, 1)
        return importance >= self.config.min_importance


class ConsoleNotifier(BaseNotifier):
    """Print notifications to console."""
    
    ICONS = {
        "info": "ℹ️",
        "learning": "💡",
        "progress": "📊",
        "milestone": "🎯",
        "checkpoint": "💾",
        "warning": "⚠️",
        "error": "❌",
        "critical": "🚨",
        "session_end": "🏁",
    }
    
    def send(self, message: str, event: str = "info", extra: Dict[str, Any] = None) -> bool:
        if not self.should_send(event):
            return False
        
        icon = self.ICONS.get(event, "📢")
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {icon} [{event.upper()}] {message}")
        
        if extra:
            for k, v in extra.items():
                print(f"    {k}: {v}")
        
        return True


class FileNotifier(BaseNotifier):
    """Log notifications to a JSONL file."""
    
    def __init__(self, config: NotifConfig = None, path: Path = None):
        super().__init__(config)
        self.path = path or (LOG_DIR / "notifications.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
    
    def send(self, message: str, event: str = "info", extra: Dict[str, Any] = None) -> bool:
        if not self.should_send(event):
            return False
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "message": message,
        }
        if extra:
            entry["extra"] = extra
        
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
        return True


class TelegramNotifier(BaseNotifier):
    """Send notifications via Telegram bot."""
    
    def __init__(self, config: NotifConfig = None):
        super().__init__(config)
        self.token = self.config.telegram_token
        self.chat_id = self.config.telegram_chat_id
        self._session = None
    
    @property
    def session(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
        return self._session
    
    def send(self, message: str, event: str = "info", extra: Dict[str, Any] = None) -> bool:
        if not self.token or not self.chat_id:
            return False
        
        if not self.should_send(event):
            return False
        
        icon = ConsoleNotifier.ICONS.get(event, "📢")
        text = f"{icon} *{event.upper()}*\n{message}"
        
        if extra:
            text += "\n\n" + "\n".join(f"• {k}: {v}" for k, v in extra.items())
        
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            resp = self.session.post(url, json={
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "Markdown"
            }, timeout=10)
            return resp.status_code == 200
        except Exception as e:
            print(f"[NOTIF] Telegram error: {e}")
            return False


class DiscordNotifier(BaseNotifier):
    """Send notifications via Discord webhook."""
    
    COLORS = {
        "info": 0x3498db,
        "learning": 0x9b59b6,
        "progress": 0x2ecc71,
        "milestone": 0xf1c40f,
        "checkpoint": 0x1abc9c,
        "warning": 0xe67e22,
        "error": 0xe74c3c,
        "critical": 0x992d22,
        "session_end": 0x7f8c8d,
    }
    
    def __init__(self, config: NotifConfig = None):
        super().__init__(config)
        self.webhook_url = self.config.discord_webhook
        self._session = None
    
    @property
    def session(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
        return self._session
    
    def send(self, message: str, event: str = "info", extra: Dict[str, Any] = None) -> bool:
        if not self.webhook_url:
            return False
        
        if not self.should_send(event):
            return False
        
        color = self.COLORS.get(event, 0x3498db)
        
        embed = {
            "title": f"🤖 LLM Maison — {event.upper()}",
            "description": message,
            "color": color,
            "timestamp": datetime.now().isoformat(),
        }
        
        if extra:
            embed["fields"] = [
                {"name": k, "value": str(v), "inline": True}
                for k, v in extra.items()
            ]
        
        try:
            resp = self.session.post(
                self.webhook_url,
                json={"embeds": [embed]},
                timeout=10
            )
            return resp.status_code in (200, 204)
        except Exception as e:
            print(f"[NOTIF] Discord error: {e}")
            return False


class MultiNotifier(BaseNotifier):
    """Send to multiple backends."""
    
    def __init__(self, notifiers: list):
        super().__init__()
        self.notifiers = notifiers
    
    def send(self, message: str, event: str = "info", extra: Dict[str, Any] = None) -> bool:
        results = [n.send(message, event, extra) for n in self.notifiers]
        return any(results)


_global_notifier: Optional[BaseNotifier] = None


def get_notifier(config: NotifConfig = None) -> BaseNotifier:
    """Get or create the global notifier based on config."""
    global _global_notifier
    
    if _global_notifier is not None and config is None:
        return _global_notifier
    
    cfg = config or NOTIF_CFG
    
    if cfg.backend == "console":
        _global_notifier = ConsoleNotifier(cfg)
    elif cfg.backend == "file":
        _global_notifier = FileNotifier(cfg)
    elif cfg.backend == "telegram":
        _global_notifier = TelegramNotifier(cfg)
    elif cfg.backend == "discord":
        _global_notifier = DiscordNotifier(cfg)
    elif cfg.backend == "multi" or cfg.backend == "all":
        notifiers = [ConsoleNotifier(cfg), FileNotifier(cfg)]
        if cfg.telegram_token and cfg.telegram_chat_id:
            notifiers.append(TelegramNotifier(cfg))
        if cfg.discord_webhook:
            notifiers.append(DiscordNotifier(cfg))
        _global_notifier = MultiNotifier(notifiers)
    else:
        _global_notifier = ConsoleNotifier(cfg)
    
    return _global_notifier


def notify(message: str, event: str = "info", extra: Dict[str, Any] = None) -> bool:
    """Global notify function using default notifier."""
    return get_notifier().send(message, event, extra)


def configure_notifications(
    backend: str = "console",
    telegram_token: str = "",
    telegram_chat_id: str = "",
    discord_webhook: str = "",
    min_importance: int = 1
):
    """Configure the global notification settings."""
    global _global_notifier, NOTIF_CFG
    
    NOTIF_CFG = NotifConfig(
        backend=backend,
        telegram_token=telegram_token,
        telegram_chat_id=telegram_chat_id,
        discord_webhook=discord_webhook,
        min_importance=min_importance
    )
    _global_notifier = None
    return get_notifier()


if __name__ == "__main__":
    print("Testing notification system...")
    print("="*50)
    
    notify("This is an info message", event="info")
    notify("A new fact was learned!", event="learning", extra={"fact": "The sky is blue"})
    notify("Training reached milestone", event="milestone", extra={"step": 10000, "loss": 0.5})
    notify("Something went wrong", event="error", extra={"error": "OutOfMemory"})
    notify("Session complete", event="session_end", extra={"duration": "2h 30m"})
    
    print("\n" + "="*50)
    print("Testing file notifier...")
    file_notifier = FileNotifier()
    file_notifier.send("Test log entry", event="info")
    print(f"Check: {file_notifier.path}")
    
    print("\n✅ Notification system test complete")
