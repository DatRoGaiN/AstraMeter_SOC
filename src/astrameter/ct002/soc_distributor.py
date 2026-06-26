"""SOC-based load distribution across multiple batteries."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from astrameter.powermeter.homeassistant import HomeAssistant

logger = logging.getLogger("astrameter")


class SOCDistributor:
    """
    Liest Battery-Ladezustände (SOC) von Home Assistant Sensoren und
    berechnet SOC-basierte Lastverteilungsgewichte für faire Batterie-Nutzung.
    
    Logik:
    - Liest SOC-Werte nur bei Änderung (gerundet auf ganze %)
    - Cached SOC-Werte für bis zu 5 Minuten
    - Fallback nach Timeout: Gleichmäßige 50/50 Verteilung
    
    Gewicht-Formel:
        weight_i = soc_i / sum(all_socs)
    
    Beispiel:
        SOC1=30%, SOC2=70% → weight1=0.3, weight2=0.7
    """

    def __init__(
        self,
        ha_powermeter: HomeAssistant,
        mac_to_soc_entity: dict[str, str],
        fallback_timeout_seconds: int = 300,
    ):
        """
        Args:
            ha_powermeter: Home Assistant Powermeter (bereits verbunden)
            mac_to_soc_entity: Dict von MAC→HA-Sensor, z.B. {"60323bd12622": "sensor.a1_..."}
            fallback_timeout_seconds: Nach dieser Zeit ohne SOC → Gleichmäßig (300s = 5min)
        """
        self.ha = ha_powermeter
        self.mac_to_soc_entity = mac_to_soc_entity
        self.fallback_timeout_seconds = fallback_timeout_seconds

        # Cache: gerundete SOC-Werte (MAC → SOC%)
        self.cached_soc: dict[str, int] = {}
        # Zeitstempel des letzten erfolgreichen SOC-Reads
        self.last_valid_soc_read = time.time()
        # Zeitstempel des letzten Update-Versuchs (verhindert zu häufiges Polling)
        self.last_soc_update_attempt = 0.0
        # Minimum Interval zwischen Updates (60 Sekunden)
        self.min_update_interval = 60.0

        logger.info(
            "SOC Distributor initialized with %d batteries",
            len(mac_to_soc_entity),
        )
        for mac, entity in mac_to_soc_entity.items():
            logger.debug(f"  MAC {mac}: {entity}")

    def get_soc_weight(self, consumer_mac: str) -> float:
        """
        Gibt das Lastverteilungsgewicht für einen Akku basierend auf SOC zurück.

        Returns:
            float: Gewicht (0.0 - 1.0+)
                - Bei 2 Akkus mit 30%/70% SOC: 0.3 und 0.7
                - Bei Fehler: 1.0 (neutral)

        Aufruf HÄUFIG, Update SELTEN:
            - Diese Methode wird bei jedem Target-Compute aufgerufen (viele x/sec)
            - Aber SOC-Update nur 1x/min oder bei Änderung
        """
        # Versuche SOC zu aktualisieren (max 1x/Minute)
        self._try_update_soc_if_needed()

        # Berechne Gewichte aus gecachtem SOC
        weights = self._compute_weights()
        return weights.get(consumer_mac, 1.0)

    def _try_update_soc_if_needed(self) -> None:
        """
        Aktualisiert SOC-Cache nur wenn:
        - Mindestens 60 Sekunden seit letztem Versuch vergangen
        - ODER sofort beim ersten Aufruf
        """
        now = time.time()
        if now - self.last_soc_update_attempt < self.min_update_interval:
            return

        self.last_soc_update_attempt = now
        self._update_soc_from_ha()

    def _update_soc_from_ha(self) -> None:
        """
        Liest SOC-Werte von Home Assistant und aktualisiert Cache wenn sich etwas ändert.
        
        Nur bei Änderung (gerundet auf ganze %):
            - Sonst zu viel Rauschen und zu häufige Neuberechnung
        """
        new_soc: dict[str, int] = {}
        valid_count = 0

        # Lese alle Sensoren
        for mac, sensor_entity in self.mac_to_soc_entity.items():
            soc_value = self._read_soc_sensor(sensor_entity)

            if soc_value is not None:
                # Runde auf ganze % (wichtig!)
                soc_rounded = round(float(soc_value))
                soc_rounded = max(0, min(100, soc_rounded))
                new_soc[mac] = soc_rounded
                valid_count += 1

        # Nur speichern wenn mind. ein Wert gültig war
        if valid_count == 0:
            # Kein Sensor erreichbar - Check Timeout
            timeout_elapsed = time.time() - self.last_valid_soc_read
            if timeout_elapsed > self.fallback_timeout_seconds:
                logger.warning(
                    "SOC sensors not reachable for %.0fs (timeout: %.0fs). "
                    "Falling back to equal distribution.",
                    timeout_elapsed,
                    self.fallback_timeout_seconds,
                )
                # Fallback: Gleiche Verteilung für alle
                self.cached_soc = {mac: 50 for mac in self.mac_to_soc_entity}
            return

        # Check: Hat sich etwas geändert?
        if new_soc != self.cached_soc:
            old_soc = self.cached_soc.copy()
            self.cached_soc = new_soc
            self.last_valid_soc_read = time.time()

            # Log der Änderungen
            for mac, soc in new_soc.items():
                old_val = old_soc.get(mac, "?")
                if old_val != soc:
                    logger.info(f"SOC updated: {mac} = {soc}% (was {old_val}%)")

    def _read_soc_sensor(self, sensor_entity: str) -> float | None:
        """
        Liest einen einzelnen SOC-Sensor über die HomeAssistant-Connection.
        
        Nutzt die bestehende _entity_values aus der HA-Powermeter-Klasse.
        """
        try:
            # Das ist die Schnittstelle zur HomeAssistant-Klasse
            # Die Sensoren werden via WebSocket abonniert und sind hier gecacht
            value = self.ha._entity_values.get(sensor_entity)

            if value is None or value == "unknown" or value == "unavailable":
                logger.debug(f"SOC sensor {sensor_entity} not available")
                return None

            return float(value)
        except (ValueError, AttributeError, TypeError) as e:
            logger.debug(f"Error reading SOC sensor {sensor_entity}: {e}")
            return None

    def _compute_weights(self) -> dict[str, float]:
        """
        Berechnet faire Gewichte aus den gecachten SOC-Werten.
        
        Formel (DEINE Idee):
            weight_i = soc_i / sum(all_socs)
        
        Fallback bei keinem/Fehler:
            - Alle Gewichte = 1.0 / n (gleichmäßig)
        """
        weights = {}

        # Kein Cache vorhanden? → Gleichmäßig
        if not self.cached_soc:
            n = len(self.mac_to_soc_entity)
            return {mac: 1.0 / n for mac in self.mac_to_soc_entity}

        # Berechne Summe
        total_soc = sum(self.cached_soc.values())

        # Alle bei 0%? → Gleichmäßig
        if total_soc == 0:
            n = len(self.cached_soc)
            return {mac: 1.0 / n for mac in self.cached_soc}

        # Normale Berechnung
        for mac, soc in self.cached_soc.items():
            weights[mac] = soc / total_soc

        return weights

    def get_status(self) -> dict:
        """Für Debugging/Status: Gib aktuellen SOC-Cache zurück."""
        return {
            "cached_soc": self.cached_soc.copy(),
            "weights": self._compute_weights(),
            "last_valid_read": self.last_valid_soc_read,
            "timeout_seconds": self.fallback_timeout_seconds,
        }
