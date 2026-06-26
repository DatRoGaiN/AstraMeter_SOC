# SOC-Based Load Distribution - Integrations Guide

## Overview

This feature adds SOC (State of Charge) based load distribution to AstraMeter, enabling fair battery usage in multi-battery setups based on each battery's current charge level.

**Formel:** `weight_i = soc_i / sum(all_socs)`

## Configuration

Add these parameters to your `config.ini` in the `[HOMEASSISTANT]` section:

```ini
[HOMEASSISTANT]
IP = 192.168.1.105
PORT = 8123
HTTPS = True
ACCESSTO KEN = YOUR_ACCESS_TOKEN
CURRENT_POWER_ENTITY = sensor.angepasste_hausleistung

# Enable SOC-based distribution
SOC_DISTRIBUTION_ENABLED = True

# Map battery MACs to Home Assistant SOC sensors
# Format: MAC=sensor.entity_id (comma-separated for multiple batteries)
SOC_SENSOR_MAP = 60323bd12622=sensor.a1_b2500_1_a1_battery_level,60323bd14c21=sensor.a2_b2500_2_a2_battery_level

# Fallback to equal distribution after this timeout (seconds)
# Use 0 to disable fallback
SOC_FALLBACK_TIMEOUT_SECONDS = 300
```

## How it Works

1. **SOC Reading** (once per minute)
   - AstraMeter reads SOC sensors from Home Assistant
   - Values are rounded to whole percentages
   - Only updates cache if SOC changes (prevents noise)

2. **Weight Calculation**
   - For each battery: `weight = battery_soc / total_soc_sum`
   - Example: Battery1=30%, Battery2=70% → weight1=0.3, weight2=0.7

3. **Load Distribution**
   - The LoadBalancer applies these weights to the fair-share calculation
   - Higher SOC batteries get higher targets (discharge more)
   - Lower SOC batteries get lower targets (discharge less)

4. **Fallback Logic**
   - If any SOC sensor is unavailable, cache is kept for up to 5 minutes
   - After timeout, falls back to 50/50 equal distribution
   - Prevents oscillation due to temporary sensor issues

## Files Changed

1. **NEW:** `src/astrameter/ct002/soc_distributor.py`
   - SOC reader and weight calculator
   - HomeAssistant integration
   - Fallback logic

2. **MODIFIED:** `src/astrameter/ct002/balancer.py`
   - Integration of SOC distributor
   - SOC-weighted fair-share calculation in `_compute_desired_contribution`

3. **MODIFIED:** `src/astrameter/ct002/ct002.py`
   - SOC distributor initialization
   - Parameter passing to LoadBalancer

4. **MODIFIED:** `src/astrameter/main.py`
   - Configuration reading for SOC parameters
   - SOC distributor setup and passing to CT002

5. **UPDATED:** `config.ini.example`
   - Documentation of new parameters

## Testing

1. Configure SOC sensors in `config.ini`
2. Start AstraMeter
3. Check logs for SOC initialization:
   ```
   SOC Distributor initialized with 2 batteries
     MAC 60323bd12622: sensor.a1_b2500_1_a1_battery_level
     MAC 60323bd14c21: sensor.a2_b2500_2_a2_battery_level
   ```

4. Monitor battery behavior:
   - Battery with higher SOC should discharge more
   - Battery with lower SOC should discharge less
   - Both should converge toward equal SOC over time

## Fallback Behavior

If SOC sensors stop responding:

- **0-5 minutes:** Continue using last known SOC values
- **After 5 minutes:** Fall back to equal 50/50 distribution
- **When sensors return:** Resume SOC-based distribution

This prevents oscillation and ensures stable operation during temporary sensor issues.

## Troubleshooting

**Q: SOC updates not showing in logs**
A: SOC values are only logged when they change (after rounding to whole %). Check every 1-2 minutes.

**Q: Batteries still not balanced**
A: Ensure battery MACs are correct and sensors are in Home Assistant. Check HA logs for sensor availability.

**Q: Fallback to equal distribution?**
A: Increase `SOC_FALLBACK_TIMEOUT_SECONDS` or check Home Assistant connectivity.
