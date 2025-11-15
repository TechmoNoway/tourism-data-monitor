#!/bin/bash
#
# Deployment Script for Tourism Data Scheduler
# Run this on your production Linux server
#

echo "=================================="
echo "Tourism Data Scheduler Deployment"
echo "=================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}Please run as root (sudo)${NC}"
    exit 1
fi

# Configuration
PROJECT_DIR="/var/www/tourism_data_monitor"
SERVICE_FILE="tourism-scheduler.service"
LOG_DIR="/var/log/tourism-scheduler"

echo "Step 1: Creating log directory..."
mkdir -p $LOG_DIR
chown www-data:www-data $LOG_DIR
echo -e "${GREEN}✓ Log directory created${NC}"

echo ""
echo "Step 2: Installing systemd service..."
cp deployment/$SERVICE_FILE /etc/systemd/system/
echo -e "${GREEN}✓ Service file copied${NC}"

echo ""
echo "Step 3: Reloading systemd..."
systemctl daemon-reload
echo -e "${GREEN}✓ Systemd reloaded${NC}"

echo ""
echo "Step 4: Enabling service..."
systemctl enable tourism-scheduler.service
echo -e "${GREEN}✓ Service enabled (will start on boot)${NC}"

echo ""
echo "Step 5: Starting service..."
systemctl start tourism-scheduler.service
echo -e "${GREEN}✓ Service started${NC}"

echo ""
echo "=================================="
echo "Deployment Complete!"
echo "=================================="
echo ""
echo "Service Status:"
systemctl status tourism-scheduler.service --no-pager
echo ""
echo "Useful Commands:"
echo "  Check status:  sudo systemctl status tourism-scheduler"
echo "  Stop service:  sudo systemctl stop tourism-scheduler"
echo "  Start service: sudo systemctl start tourism-scheduler"
echo "  Restart:       sudo systemctl restart tourism-scheduler"
echo "  View logs:     sudo journalctl -u tourism-scheduler -f"
echo "  Check output:  tail -f $LOG_DIR/output.log"
echo ""
