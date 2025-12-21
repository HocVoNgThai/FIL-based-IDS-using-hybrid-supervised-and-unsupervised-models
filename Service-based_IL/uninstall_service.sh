#!/bin/bash
set -e

# ===== CONFIG =====
SYSTEMD_DIR="/etc/systemd/system"

SERVICES=(
  "ids_cicextract.service"
  "ids_flowzmqserver.service"
  "ids_dashboard.service"
  "ids_il.service"
  "ids_il.timer"
)

# ===== CHECK ROOT =====
if [[ "$EUID" -ne 0 ]]; then
  echo "[!] Please run as root"
  exit 1
fi

echo "[+] Removing IDS services..."

# ===== STOP & DISABLE =====
for svc in "${SERVICES[@]}"; do
  if systemctl list-unit-files | grep -q "$svc"; then
    echo "  → Stopping $svc"
    systemctl stop "$svc" || true
    systemctl disable "$svc" || true
  fi
done

# ===== REMOVE SERVICE FILES =====
for svc in "${SERVICES[@]}"; do
  if [[ -f "$SYSTEMD_DIR/$svc" ]]; then
    echo "  → Removing $svc"
    rm -f "$SYSTEMD_DIR/$svc"
  fi
done

# ===== RELOAD =====
systemctl daemon-reexec
systemctl daemon-reload

echo ""
echo "\033[0;32m[✓]\033[0m IDS services removed completely"
