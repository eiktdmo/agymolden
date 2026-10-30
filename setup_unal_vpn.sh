#!/usr/bin/env bash
# ==============================================================================
# setup_unal_vpn.sh
# 
# Automated setup for UNAL GlobalProtect VPN (vpnpa.unal.edu.co) on Kubuntu / Ubuntu.
# Configures:
#   1. Required packages (openconnect, libnotify-bin)
#   2. Passwordless sudo rule for /usr/sbin/openconnect
#   3. Secure credentials file (~/.config/unal-vpn/credentials.txt)
#   4. 1-click toggle connection script (~/.local/bin/unal-vpn-toggle.sh)
#   5. KDE Plasma desktop entry (~/.local/share/applications/unal-vpn.desktop)
# ==============================================================================

set -euo pipefail

echo "=========================================================="
echo "    Configuring UNAL GlobalProtect VPN (Kubuntu Setup)    "
echo "=========================================================="

# 1. Install dependencies
echo ""
echo "[1/5] Installing openconnect and libnotify-bin..."
sudo apt update
sudo apt install -y openconnect libnotify-bin

# 2. Configure passwordless sudo for openconnect
echo ""
echo "[2/5] Setting up passwordless sudo for /usr/sbin/openconnect..."
echo "$USER ALL=(ALL) NOPASSWD: /usr/sbin/openconnect" | sudo tee /etc/sudoers.d/openconnect > /dev/null
sudo chmod 0440 /etc/sudoers.d/openconnect
echo "✓ Sudoers rule created successfully."

# 3. Prompt for UNAL Credentials
echo ""
echo "[3/5] Setting up UNAL Credentials..."
mkdir -p "$HOME/.config/unal-vpn"

read -rp "Enter your UNAL username (e.g. areyesv): " UNAL_USER
while [[ -z "$UNAL_USER" ]]; do
    read -rp "Username cannot be empty. Please enter your UNAL username: " UNAL_USER
done

read -rsp "Enter your UNAL password: " UNAL_PASS
echo ""
while [[ -z "$UNAL_PASS" ]]; do
    read -rsp "Password cannot be empty. Please enter your UNAL password: " UNAL_PASS
    echo ""
done

cat << EOF > "$HOME/.config/unal-vpn/credentials.txt"
$UNAL_USER
$UNAL_PASS
EOF
chmod 600 "$HOME/.config/unal-vpn/credentials.txt"
echo "✓ Credentials securely saved in ~/.config/unal-vpn/credentials.txt (chmod 600)."

# 4. Create the toggle connection script
echo ""
echo "[4/5] Creating toggle connection script..."
mkdir -p "$HOME/.local/bin"

cat << 'EOF' > "$HOME/.local/bin/unal-vpn-toggle.sh"
#!/bin/bash
PIDFILE="/tmp/unal-vpn.pid"

# If already running, disconnect
if pgrep -x openconnect > /dev/null; then
    sudo pkill openconnect
    rm -f "$PIDFILE"
    notify-send -i network-vpn -a "UNAL VPN" "VPN Desconectada" "Se ha cerrado la conexión a vpnpa.unal.edu.co"
    exit 0
fi

CREDS_FILE="$HOME/.config/unal-vpn/credentials.txt"
if [ ! -f "$CREDS_FILE" ]; then
    notify-send -i dialog-error -a "UNAL VPN" "Error de Credenciales" "No se encontró el archivo $CREDS_FILE"
    exit 1
fi

USER_NAME=$(head -n 1 "$CREDS_FILE")

# Connect in background with password from credentials file
tail -n 1 "$CREDS_FILE" | sudo /usr/sbin/openconnect \
    --protocol=gp \
    --background \
    --pid-file="$PIDFILE" \
    --user="$USER_NAME" \
    --passwd-on-stdin \
    vpnpa.unal.edu.co

if [ $? -eq 0 ]; then
    notify-send -i network-vpn -a "UNAL VPN" "VPN Conectada" "Conexión establecida con vpnpa.unal.edu.co"
else
    notify-send -i dialog-error -a "UNAL VPN" "Error de Conexión" "No se pudo conectar a la VPN de la UNAL"
fi
EOF

chmod +x "$HOME/.local/bin/unal-vpn-toggle.sh"
echo "✓ Script created at $HOME/.local/bin/unal-vpn-toggle.sh."

# 5. Create Desktop Entry
echo ""
echo "[5/5] Creating Desktop Entry for KDE Plasma..."
mkdir -p "$HOME/.local/share/applications"

cat << EOF > "$HOME/.local/share/applications/unal-vpn.desktop"
[Desktop Entry]
Name=VPN UNAL (GlobalProtect)
Comment=Connect or Disconnect UNAL GlobalProtect VPN
Exec=$HOME/.local/bin/unal-vpn-toggle.sh
Icon=network-vpn
Terminal=false
Type=Application
Categories=Network;
EOF

update-desktop-database "$HOME/.local/share/applications/" 2>/dev/null || true
echo "✓ Desktop entry created at ~/.local/share/applications/unal-vpn.desktop."

echo ""
echo "=========================================================="
echo "                   SETUP COMPLETE!                        "
echo "=========================================================="
echo "You can now connect to UNAL VPN in 3 ways:"
echo ""
echo " 1. Desktop GUI: Search for 'VPN UNAL' in your Application Menu (Alt+Space)."
echo " 2. Terminal:    Run ~/.local/bin/unal-vpn-toggle.sh"
echo " 3. AgyMolden:   Set VPN Connect Command to: $HOME/.local/bin/unal-vpn-toggle.sh"
echo "=========================================================="
