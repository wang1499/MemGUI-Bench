#!/bin/bash
echo "Killing emulators..."
pkill -9 -f emulator || true
pkill -9 -f "benchmark_run.py" || true
pkill -9 -f "android_world" || true
echo "Cleaning lock files..."
rm -f ~/.android/avd/*.lock 2>/dev/null || true
rm -f /tmp/*emulator* 2>/dev/null || true
rm -f /tmp/*5554* 2>/dev/null || true
rm -f /tmp/*5556* 2>/dev/null || true
echo "Done!"