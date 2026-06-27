"""SOC (State of Charge) based load distribution for multiple batteries."""

from __future__ import annotations

from astrameter.config.logger import logger


class SOCDistributor:
    """Manages SOC-based weight distribution for multiple batteries.
    
    Distributes charging/discharging power between N batteries proportional
    to their State of Charge (SOC) values using the formula:
    
        weight_n = SOC_n / (SOC_1 + SOC_2 + ... + SOC_N)
    
    This ensures:
    - Battery with higher SOC gets higher share (discharges first)
    - Battery with lower SOC gets lower share (charges first)
    - Power distribution is proportional to each battery's capacity
    """

    def __init__(self, battery_macs: dict[str, str] | None = None, **kwargs):
        """Initialize SOC distributor with battery MAC addresses.
        
        Can be initialized in two ways:
        
        1. Using keyword arguments (legacy, for 2 batteries):
            SOCDistributor(battery_1_mac="aa:bb:cc", battery_2_mac="dd:ee:ff")
        
        2. Using battery_macs dict (flexible, for N batteries):
            SOCDistributor(battery_macs={
                "battery_1": "aa:bb:cc",
                "battery_2": "dd:ee:ff",
                "battery_3": "11:22:33"
            })
        
        Args:
            battery_macs: Dict mapping battery names to MAC addresses
            **kwargs: Legacy support for battery_1_mac, battery_2_mac, etc.
        """
        self._battery_macs: dict[str, str] = {}  # {battery_name: mac_address}
        self._mac_to_name: dict[str, str] = {}   # {mac_address: battery_name}
        
        # Support both initialization methods
        if battery_macs:
            # Method 1: battery_macs dict provided
            for name, mac in battery_macs.items():
                mac_lower = mac.lower()
                self._battery_macs[name] = mac_lower
                self._mac_to_name[mac_lower] = name
        else:
            # Method 2: Extract from kwargs (battery_1_mac, battery_2_mac, ...)
            battery_num = 1
            while f"battery_{battery_num}_mac" in kwargs:
                mac = kwargs[f"battery_{battery_num}_mac"]
                mac_lower = mac.lower()
                name = f"battery_{battery_num}"
                self._battery_macs[name] = mac_lower
                self._mac_to_name[mac_lower] = name
                battery_num += 1
        
        # SOC cache: {mac_address: soc_value}
        # SOC values are typically 0-100 (percentage)
        self._soc_cache: dict[str, float] = {}
        
        battery_count = len(self._battery_macs)
        logger.info(
            "SOCDistributor initialized with %d batteries: %s",
            battery_count,
            ", ".join(
                f"{name}={mac}" 
                for name, mac in sorted(self._battery_macs.items())
            ),
        )

    def update_soc(self, consumer_id: str, soc: float) -> None:
        """Update the cached SOC value for a battery.
        
        Args:
            consumer_id: Battery identifier (MAC address or name)
            soc: State of Charge value (0-100 percentage)
        """
        consumer_id_lower = consumer_id.lower()
        soc_value = max(0.0, min(100.0, float(soc)))
        self._soc_cache[consumer_id_lower] = soc_value
        
        battery_name = self._mac_to_name.get(consumer_id_lower, consumer_id_lower[:16])
        logger.debug(
            "SOC updated: %s (mac=%s) soc=%.1f%%",
            battery_name,
            consumer_id_lower[:16],
            soc_value,
        )

    def get_soc(self, consumer_id: str) -> float | None:
        """Get the cached SOC value for a battery.
        
        Args:
            consumer_id: Battery identifier (MAC address or name)
            
        Returns:
            SOC value (0-100) or None if not available
        """
        return self._soc_cache.get(consumer_id.lower())

    def get_soc_weight(self, consumer_id: str) -> float:
        """Calculate the SOC-based weight for a battery.
        
        Uses the formula:
            weight_n = SOC_n / (SOC_1 + SOC_2 + ... + SOC_N)
        
        This distributes power proportionally to battery capacity across all batteries.
        
        Examples (3 batteries):
            - All at 50% SOC: weights = [0.333, 0.333, 0.333] (equal split)
            - SOCs [50%, 30%, 20%]: weights = [0.500, 0.300, 0.200] (proportional)
            - SOCs [80%, 15%, 5%]: weights = [0.800, 0.150, 0.050] (high variance)
        
        Args:
            consumer_id: Battery identifier (MAC address or name)
            
        Returns:
            Weight factor (0.0-1.0 range, all weights sum to 1.0)
        """
        consumer_id_lower = consumer_id.lower()
        
        # Get SOC for this battery
        soc_self = self._soc_cache.get(consumer_id_lower)
        
        if soc_self is None:
            battery_name = self._mac_to_name.get(consumer_id_lower, consumer_id_lower[:16])
            logger.debug(
                "SOCDistributor: missing SOC for %s",
                battery_name,
            )
            return 1.0  # Neutral weight if SOC unknown
        
        # Clamp to [0, 100]
        soc_self = max(0.0, min(100.0, soc_self))
        
        # Calculate sum of all known SOC values
        soc_sum = sum(
            max(0.0, min(100.0, soc))
            for soc in self._soc_cache.values()
        )
        
        # Guard against division by zero (all batteries at 0% SOC)
        if soc_sum <= 0.0:
            # Equal distribution if all at 0%
            weight = 1.0 / max(1, len(self._battery_macs))
        else:
            # weight = SOC_self / (SOC_1 + SOC_2 + ... + SOC_N)
            weight = soc_self / soc_sum
        
        battery_name = self._mac_to_name.get(consumer_id_lower, consumer_id_lower[:16])
        logger.debug(
            "SOCDistributor weight: %s soc=%.1f%% soc_sum=%.1f%% weight=%.3f",
            battery_name,
            soc_self,
            soc_sum,
            weight,
        )
        
        return weight

    def get_all_weights(self) -> dict[str, float]:
        """Get weights for all configured batteries.
        
        Returns:
            Dict mapping battery MAC addresses to their weights.
            All weights sum to 1.0 (or less if some batteries have unknown SOC).
        """
        weights = {}
        for mac in self._battery_macs.values():
            weights[mac] = self.get_soc_weight(mac)
        return weights

    def clear(self) -> None:
        """Clear all cached SOC values."""
        self._soc_cache.clear()
        logger.debug("SOCDistributor cache cleared")