"""
monitoring/alerts.py
Alerting system for ML model monitoring.

Supports multiple alert channels:
- Logging
- Email (SMTP)
- Slack
- Microsoft Teams
- Azure Monitor (Application Insights)
"""

import os
import logging
import smtplib
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)


@dataclass
class AlertConfig:
    """
    Configuration for alerting system.
    
    Attributes:
        enabled: Whether alerts are enabled
        channels: List of enabled channels (log, email, slack, teams, azure)
        email_recipients: List of email addresses for alerts
        slack_webhook_url: Slack webhook URL
        teams_webhook_url: Microsoft Teams webhook URL
        azure_connection_string: Application Insights connection string
        min_severity: Minimum severity level (info, warning, error, critical)
    """
    
    enabled: bool = True
    channels: List[str] = field(default_factory=lambda: ["log"])
    email_recipients: List[str] = field(default_factory=list)
    slack_webhook_url: str = ""
    teams_webhook_url: str = ""
    azure_connection_string: str = ""
    min_severity: str = "warning"  # info, warning, error, critical


class AlertSeverity:
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    
    # Severity order for filtering
    ORDER = {
        INFO: 0,
        WARNING: 1,
        ERROR: 2,
        CRITICAL: 3
    }
    
    @classmethod
    def is_above(cls, severity: str, min_severity: str) -> bool:
        """Check if severity is above minimum threshold."""
        return cls.ORDER.get(severity, 0) >= cls.ORDER.get(min_severity, 0)


@dataclass
class Alert:
    """
    Alert message structure.
    
    Attributes:
        severity: Alert severity (info, warning, error, critical)
        title: Alert title
        message: Alert message
        timestamp: When the alert was created
        source: Source of the alert (e.g., "drift_detector", "model_performance")
        data: Additional data for the alert
    """
    
    severity: str
    title: str
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    source: str = "system"
    data: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "severity": self.severity,
            "title": self.title,
            "message": self.message,
            "timestamp": self.timestamp,
            "source": self.source,
            "data": self.data
        }
    
    def to_json(self) -> str:
        """Convert alert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class AlertSender:
    """
    Alert sender that supports multiple channels.
    """
    
    def __init__(self, config: Optional[AlertConfig] = None):
        """
        Initialize the alert sender.
        
        Args:
            config: Alert configuration. If None, uses environment variables.
        """
        self.config = config or self._load_from_env()
        self._last_alerts: List[Alert] = []
        self._max_history = 1000
    
    def _load_from_env(self) -> AlertConfig:
        """Load configuration from environment variables."""
        return AlertConfig(
            enabled=os.getenv("ALERTS_ENABLED", "true").lower() == "true",
            channels=os.getenv("ALERT_CHANNELS", "log").split(","),
            email_recipients=os.getenv("ALERT_EMAIL_RECIPIENTS", "").split(",") if os.getenv("ALERT_EMAIL_RECIPIENTS") else [],
            slack_webhook_url=os.getenv("SLACK_WEBHOOK_URL", ""),
            teams_webhook_url=os.getenv("TEAMS_WEBHOOK_URL", ""),
            azure_connection_string=os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING", ""),
            min_severity=os.getenv("ALERT_MIN_SEVERITY", "warning")
        )
    
    def send_alert(self, alert: Alert) -> bool:
        """
        Send an alert through all enabled channels.
        
        Args:
            alert: Alert to send
        
        Returns:
            bool: True if at least one channel succeeded
        """
        if not self.config.enabled:
            return False
        
        # Check severity threshold
        if not AlertSeverity.is_above(alert.severity, self.config.min_severity):
            logger.debug(f"Alert severity '{alert.severity}' below threshold '{self.config.min_severity}'")
            return False
        
        # Store alert history
        self._last_alerts.append(alert)
        if len(self._last_alerts) > self._max_history:
            self._last_alerts = self._last_alerts[-self._max_history:]
        
        success = False
        
        # Send through each channel
        if "log" in self.config.channels:
            success = self._send_to_log(alert) or success
        
        if "email" in self.config.channels and self.config.email_recipients:
            success = self._send_to_email(alert) or success
        
        if "slack" in self.config.channels and self.config.slack_webhook_url:
            success = self._send_to_slack(alert) or success
        
        if "teams" in self.config.channels and self.config.teams_webhook_url:
            success = self._send_to_teams(alert) or success
        
        if "azure" in self.config.channels and self.config.azure_connection_string:
            success = self._send_to_azure(alert) or success
        
        return success
    
    def _send_to_log(self, alert: Alert) -> bool:
        """Send alert to log file."""
        try:
            if alert.severity == AlertSeverity.INFO:
                logger.info(f"📊 ALERT: {alert.title} - {alert.message}")
            elif alert.severity == AlertSeverity.WARNING:
                logger.warning(f"⚠️ ALERT: {alert.title} - {alert.message}")
            elif alert.severity == AlertSeverity.ERROR:
                logger.error(f"❌ ALERT: {alert.title} - {alert.message}")
            elif alert.severity == AlertSeverity.CRITICAL:
                logger.critical(f"🚨 ALERT: {alert.title} - {alert.message}")
            
            # Also write to alerts.log
            log_file = Path("logs/alerts.log")
            log_file.parent.mkdir(exist_ok=True)
            with open(log_file, "a") as f:
                f.write(alert.to_json() + "\n")
            
            return True
        except Exception as e:
            logger.error(f"Failed to send alert to log: {e}")
            return False
    
    def _send_to_email(self, alert: Alert) -> bool:
        """Send alert via email."""
        try:
            smtp_host = os.getenv("SMTP_HOST", "")
            smtp_port = int(os.getenv("SMTP_PORT", "587"))
            smtp_user = os.getenv("SMTP_USER", "")
            smtp_password = os.getenv("SMTP_PASSWORD", "")
            smtp_from = os.getenv("SMTP_FROM", "alerts@health-predictor.com")
            
            if not smtp_host or not smtp_user:
                logger.warning("Email configuration incomplete, skipping")
                return False
            
            # Build email
            msg = MIMEMultipart()
            msg["From"] = smtp_from
            msg["To"] = ", ".join(self.config.email_recipients)
            msg["Subject"] = f"[{alert.severity.upper()}] {alert.title}"
            
            body = f"""
            Alert: {alert.title}
            Severity: {alert.severity.upper()}
            Source: {alert.source}
            Time: {alert.timestamp}
            
            Message: {alert.message}
            
            Additional Data:
            {json.dumps(alert.data, indent=2) if alert.data else "None"}
            """
            
            msg.attach(MIMEText(body, "plain"))
            
            # Send email
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.starttls()
                server.login(smtp_user, smtp_password)
                server.send_message(msg)
            
            logger.info(f"✅ Alert email sent to {len(self.config.email_recipients)} recipients")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send alert email: {e}")
            return False
    
    def _send_to_slack(self, alert: Alert) -> bool:
        """Send alert to Slack."""
        try:
            import requests
            
            # Determine color based on severity
            colors = {
                AlertSeverity.INFO: "#36a64f",      # Green
                AlertSeverity.WARNING: "#ecb22e",    # Yellow
                AlertSeverity.ERROR: "#e01e5a",      # Red
                AlertSeverity.CRITICAL: "#911014"    # Dark Red
            }
            
            # Build Slack message
            payload = {
                "attachments": [{
                    "color": colors.get(alert.severity, "#36a64f"),
                    "title": f"[{alert.severity.upper()}] {alert.title}",
                    "text": alert.message,
                    "fields": [
                        {
                            "title": "Source",
                            "value": alert.source,
                            "short": True
                        },
                        {
                            "title": "Time",
                            "value": alert.timestamp,
                            "short": True
                        }
                    ],
                    "footer": "Health Predictor Monitoring",
                    "ts": int(datetime.now().timestamp())
                }]
            }
            
            # Add data if present
            if alert.data:
                payload["attachments"][0]["fields"].append({
                    "title": "Data",
                    "value": json.dumps(alert.data, indent=2)[:500],
                    "short": False
                })
            
            response = requests.post(
                self.config.slack_webhook_url,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info("✅ Alert sent to Slack")
                return True
            else:
                logger.error(f"Slack webhook failed: {response.status_code}")
                return False
                
        except ImportError:
            logger.warning("requests not installed, skipping Slack alert")
            return False
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False
    
    def _send_to_teams(self, alert: Alert) -> bool:
        """Send alert to Microsoft Teams."""
        try:
            import requests
            
            # Build Teams message
            payload = {
                "@type": "MessageCard",
                "@context": "https://schema.org/extensions",
                "summary": alert.title,
                "themeColor": "FF0000" if alert.severity == AlertSeverity.CRITICAL else "FFA500",
                "title": f"[{alert.severity.upper()}] {alert.title}",
                "text": alert.message,
                "sections": [
                    {
                        "facts": [
                            {"name": "Source", "value": alert.source},
                            {"name": "Time", "value": alert.timestamp}
                        ]
                    }
                ]
            }
            
            response = requests.post(
                self.config.teams_webhook_url,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info("✅ Alert sent to Teams")
                return True
            else:
                logger.error(f"Teams webhook failed: {response.status_code}")
                return False
                
        except ImportError:
            logger.warning("requests not installed, skipping Teams alert")
            return False
        except Exception as e:
            logger.error(f"Failed to send Teams alert: {e}")
            return False
    
    def _send_to_azure(self, alert: Alert) -> bool:
        """Send alert to Azure Application Insights."""
        try:
            from azure.monitor.opentelemetry import configure_azure_monitor
            from opentelemetry import trace
            
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span("alert") as span:
                span.set_attribute("alert.severity", alert.severity)
                span.set_attribute("alert.title", alert.title)
                span.set_attribute("alert.message", alert.message)
                span.set_attribute("alert.source", alert.source)
                span.set_attribute("alert.timestamp", alert.timestamp)
                
                if alert.data:
                    for key, value in alert.data.items():
                        span.set_attribute(f"alert.data.{key}", str(value))
            
            logger.info("✅ Alert sent to Azure Application Insights")
            return True
            
        except ImportError:
            logger.warning("Azure monitoring packages not installed, skipping Azure alert")
            return False
        except Exception as e:
            logger.error(f"Failed to send alert to Azure: {e}")
            return False
    
    def get_alert_history(self, severity: Optional[str] = None, limit: int = 10) -> List[Alert]:
        """
        Get recent alert history.
        
        Args:
            severity: Filter by severity (info, warning, error, critical)
            limit: Maximum number of alerts to return
        
        Returns:
            List of recent alerts
        """
        alerts = self._last_alerts
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        return alerts[-limit:]


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

# Global alert sender instance
_alert_sender: Optional[AlertSender] = None


def get_alert_sender() -> AlertSender:
    """Get or create the global alert sender."""
    global _alert_sender
    if _alert_sender is None:
        _alert_sender = AlertSender()
    return _alert_sender


def send_alert(
    severity: str,
    title: str,
    message: str,
    source: str = "system",
    data: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Send an alert using the global alert sender.
    
    Args:
        severity: Alert severity (info, warning, error, critical)
        title: Alert title
        message: Alert message
        source: Source of the alert
        data: Additional data
    
    Returns:
        bool: True if alert was sent successfully
    """
    alert = Alert(
        severity=severity,
        title=title,
        message=message,
        source=source,
        data=data
    )
    sender = get_alert_sender()
    return sender.send_alert(alert)


def send_info(title: str, message: str, source: str = "system", data: Optional[Dict] = None) -> bool:
    """Send an info alert."""
    return send_alert(AlertSeverity.INFO, title, message, source, data)


def send_warning(title: str, message: str, source: str = "system", data: Optional[Dict] = None) -> bool:
    """Send a warning alert."""
    return send_alert(AlertSeverity.WARNING, title, message, source, data)


def send_error(title: str, message: str, source: str = "system", data: Optional[Dict] = None) -> bool:
    """Send an error alert."""
    return send_alert(AlertSeverity.ERROR, title, message, source, data)


def send_critical(title: str, message: str, source: str = "system", data: Optional[Dict] = None) -> bool:
    """Send a critical alert."""
    return send_alert(AlertSeverity.CRITICAL, title, message, source, data)


def get_alert_history(severity: Optional[str] = None, limit: int = 10) -> List[Alert]:
    """Get recent alert history."""
    return get_alert_sender().get_alert_history(severity, limit)


# ============================================================
# PRE-DEFINED ALERT TEMPLATES
# ============================================================

def alert_drift_detected(feature: str, message: str, data: Optional[Dict] = None) -> bool:
    """Alert when drift is detected in a feature."""
    return send_warning(
        title=f"Data Drift Detected: {feature}",
        message=message,
        source="drift_detector",
        data=data
    )


def alert_model_performance_drop(metric: str, old_value: float, new_value: float) -> bool:
    """Alert when model performance drops."""
    return send_error(
        title=f"Model Performance Drop: {metric}",
        message=f"{metric} dropped from {old_value:.4f} to {new_value:.4f}",
        source="model_performance",
        data={
            "metric": metric,
            "old_value": old_value,
            "new_value": new_value,
            "drop_percent": ((old_value - new_value) / old_value) * 100
        }
    )


def alert_anomaly(prediction: float, expected_range: tuple) -> bool:
    """Alert when an anomalous prediction is detected."""
    return send_warning(
        title="Anomalous Prediction Detected",
        message=f"Prediction {prediction:.2f} outside expected range [{expected_range[0]:.2f}, {expected_range[1]:.2f}]",
        source="anomaly_detection",
        data={
            "prediction": prediction,
            "expected_range": expected_range
        }
    )


def alert_model_retrained(model_version: str, performance: Dict) -> bool:
    """Alert when model is retrained."""
    return send_info(
        title=f"Model Retrained: {model_version}",
        message=f"Model retrained successfully with R²: {performance.get('R2', 'N/A')}",
        source="retraining",
        data=performance
    )


def alert_high_error_rate(error_rate: float, threshold: float) -> bool:
    """Alert when error rate exceeds threshold."""
    return send_error(
        title="High Error Rate Detected",
        message=f"Error rate {error_rate:.2f}% exceeds threshold {threshold:.2f}%",
        source="monitoring",
        data={
            "error_rate": error_rate,
            "threshold": threshold
        }
    )


# ============================================================
# TEST FUNCTION
# ============================================================

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Alert System")
    print("=" * 40)
    
    # Test different alert types
    send_info(
        title="System Started",
        message="Health Predictor API started successfully",
        source="startup"
    )
    
    send_warning(
        title="Memory Usage High",
        message="Memory usage is at 85%",
        source="system",
        data={"memory_usage": 85, "threshold": 80}
    )
    
    send_error(
        title="Model Prediction Failed",
        message="Failed to make prediction for request ID: abc-123",
        source="prediction",
        data={"request_id": "abc-123", "error": "Timeout"}
    )
    
    send_critical(
        title="Model Service Unavailable",
        message="Model service has been down for 5 minutes",
        source="monitoring",
        data={"downtime_minutes": 5}
    )
    
    # Test pre-defined templates
    alert_drift_detected(
        feature="Latitude",
        message="Mean shifted from 36.5 to 38.2 (3.4 std deviations)",
        data={"current_mean": 38.2, "ref_mean": 36.5}
    )
    
    alert_anomaly(
        prediction=1500.0,
        expected_range=(0, 100)
    )
    
    # Get alert history
    print("\n📋 Recent Alerts:")
    for alert in get_alert_history(limit=5):
        print(f"  [{alert.severity.upper()}] {alert.title}: {alert.message}")