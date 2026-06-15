#!/usr/bin/env python
"""
Standalone watchdog check – to be called by cron every 15 minutes.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitoring.watchdog import check_and_alert

if __name__ == "__main__":
    check_and_alert()