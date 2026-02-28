import re


def dms_to_dd(dms_str):
    """Convert a DMS string like 46°59'00"N to decimal degrees."""
    dms_str = dms_str.strip()

    # Extract components
    parts = re.split(r'[°\'"]+', dms_str)
    degrees = float(parts[0])
    minutes = float(parts[1])
    seconds = float(parts[2])
    direction = parts[3].strip()

    dd = degrees + minutes / 60 + seconds / 3600

    if direction in ('S', 'W'):
        dd *= -1

    return dd