"""
Utility Functions
Common utilities for logging, ID generation, and file handling
"""

import uuid
import logging
import os
from pathlib import Path
from datetime import datetime

from app.config.settings import settings


def generate_id(prefix: str = "") -> str:
    """
    Generate unique ID
    
    Args:
        prefix: Optional prefix for the ID
        
    Returns:
        Unique identifier string
    """
    unique_part = str(uuid.uuid4())[:8]
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    
    if prefix:
        return f"{prefix}_{timestamp}_{unique_part}"
    return f"{timestamp}_{unique_part}"


def setup_logging(name: str) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    log_path = Path(settings.log_path)
    log_path.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(name)
    # Set to DEBUG to see all messages
    logger.setLevel(logging.DEBUG)
    
    # File handler
    file_handler = logging.FileHandler(
        log_path / f"app_{datetime.utcnow().strftime('%Y%m%d')}.log"
    )
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler - always DEBUG
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def ensure_dir_exists(path: str) -> Path:
    """
    Ensure directory exists, create if not
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def cleanup_old_logs(days: int = 7) -> None:
    """
    Clean up log files older than specified days
    
    Args:
        days: Number of days to keep logs
    """
    try:
        log_path = Path(settings.log_path)
        cutoff_time = datetime.utcnow().timestamp() - (days * 86400)
        
        for log_file in log_path.glob("app_*.log"):
            if log_file.stat().st_mtime < cutoff_time:
                log_file.unlink()
                
    except Exception as e:
        logging.error(f"Error cleaning up logs: {str(e)}")
