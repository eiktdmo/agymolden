#!/usr/bin/env bash
# ==============================================================================
# configure_unal_vpn_connection.sh
#
# Configures the UNAL GlobalProtect VPN connection directly in the console
# with passwordless sudo, saved credentials, background toggle, and desktop shortcut.
# ==============================================================================

set -euo pipefail

# 1. Require username and password as arguments or prompt interactively
UNAL_USER="${1:-}"
UNAL_PASS="${2:-}"

if [[ -z "$UNAL_USER" ]]; then
    read -rp "UNAL Username (e.g. areyesv): " UNAL_USER
fi

if [[ -z "$UNAL_PASS" ]]; then
    read -rsp "UNAL Password: " UNAL_PASS
    echo ""
fi

echo "Configuring UNAL VPN for user: $UNAL_USER..."

# 2. Install openconnect if not present
if ! command -v openconnect >/dev/null 2>&1; then
    echo "Installing openconnect..."
    sudo apt update -qq && sudo apt install -y -qq openconnect libnotify-bin
fi

# 3. Add passwordless sudo rule for openconnect
echo "$USER ALL=(ALL) NOPASSWD: /usr/sbin/openconnect" | sudo tee /etc/sudoers.d/openconnect > /dev/null
sudo chmod 0440 /etc/sudoers.d/openconnect

# 4. Save credentials securely
mkdir -p "$HOME/.config/unal-vpn"
printf "%s\n%s\n" "$UNAL_USER" "$UNAL_PASS" > "$HOME/.config/unal-vpn/credentials.txt"
chmod 600 "$HOME/.config/unal-vpn/credentials.txt"

# 5. Create background connect / toggle script
mkdir -p "$HOME/.local/bin"
cat << 'TOGGLE_EOF' > "$HOME/.local/bin/unal-vpn-toggle.sh"
#!/bin/bash
PIDFILE="/tmp/unal-vpn.pid"

if pgrep -x openconnect > /dev/null; then
    sudo pkill openconnect
    rm -f "$PIDFILE"
    command -v notify-send >/dev/null && notify-send -i network-vpn -a "UNAL VPN" "VPN Desconectada" "Conexión cerrada"
    echo "VPN Desconectada."
    exit 0
fi

CREDS_FILE="$HOME/.config/unal-vpn/credentials.txt"
USER_NAME=$(head -n 1 "$CREDS_FILE")

tail -n 1 "$CREDS_FILE" | sudo /usr/sbin/openconnect \
    --protocol=gp \
    --background \
    --pid-file="$PIDFILE" \
    --user="$USER_NAME" \
    --passwd-on-stdin \
    vpnpa.unal.edu.co

if [ $? -eq 0 ]; then
    command -v notify-send >/dev/null && notify-send -i network-vpn -a "UNAL VPN" "VPN Conectada" "Conectado a vpnpa.unal.edu.co"
    echo "VPN Conectada exitosamente a vpnpa.unal.edu.co"
else
    command -v notify-send >/dev/null && notify-send -i dialog-error -a "UNAL VPN" "Error" "No se pudo conectar"
    echo "Error al conectar a la VPN."
fi
TOGGLE_EOF
chmod +x "$HOME/.local/bin/unal-vpn-toggle.sh"

# 6. Create Desktop launcher for KDE
mkdir -p "$HOME/.local/share/applications"
cat << DESKTOP_EOF > "$HOME/.local/share/applications/unal-vpn.desktop"
[Desktop Entry]
Name=VPN UNAL (GlobalProtect)
Comment=Connect or Disconnect UNAL GlobalProtect VPN
Exec=$HOME/.local/bin/unal-vpn-toggle.sh
Icon=network-vpn
Terminal=false
Type=Application
Categories=Network;
DESKTOP_EOF

update-desktop-database "$HOME/.local/share/applications/" 2>/dev/null || true

echo "✓ Configuration complete!"
echo "Test connection now by running: ~/.local/bin/unal-vpn-toggle.sh"
