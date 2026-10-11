"""
Ingestion module for The Golden Hour Defense Framework.
"""
from src.ingestion.i4c_webhook import app, I4CTicketPayload, GoldenHourEmergencyBrake

__all__ = ["app", "I4CTicketPayload", "GoldenHourEmergencyBrake"]
