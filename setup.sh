#!/bin/bash
# Complete SIH Lunar SLAM Setup Script
# Run: chmod +x setup_sih.sh && ./setup_sih.sh

set -e  # Exit on error

echo "=========================================="
echo "SIH Lunar SLAM Complete Setup Script"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if running as root (we need sudo for some steps)
if [[ $EUID -eq 0 ]]; then
   print_error "Don't run this script as root! Run as normal user."
   exit 1
fi

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIH_DIR="$HOME/SIH"

echo -e "${GREEN}[INFO]${NC} Project directory: $SIH_DIR"
echo -e "${GREEN}[INFO]${NC} Script directory: $SCRIPT_DIR"

# ============================================
# STEP 1: Install ROS 2 Humble
# ============================================
print_step "1/7 Installing ROS 2 Humble..."

sudo apt update && sudo apt install -y curl gnupg2 lsb-release software-properties-common

sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install -y ros-humble-desktop python3-colcon-common-extensions python3-rosdep

sudo rosdep init 2>/dev/null || true
rosdep update

# Source ROS in current shell and add to bashrc
source /opt/ros/humble/setup.bash
grep -q "source /opt/ros/humble/setup.bash" ~/.bashrc || echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

print_step "ROS 2 Humble installed successfully"

# ============================================
# STEP 2: Install Dependencies
# ============================================
print_step "2/7 Installing dependencies..."

sudo apt update && sudo apt install -y \
  libeigen3-dev libopencv-dev libpcl-dev liboctomap-dev \
  ros-humble-nav2-common ros-humble-nav2-core ros-humble-nav2-costmap-2d \
  ros-humble-nav2-planner ros-humble-nav2-controller ros-humble-nav2-bt-navigator \
  ros-humble-nav2-lifecycle-manager ros-humble-nav2-map-server ros-humble-nav2-amcl \
  ros-humble-nav2-smac-planner ros-humble-nav2-dwb-controller \
  ros-humble-nav2-regulated-pure-pursuit-controller ros-humble-nav2-voxel-grid \
  ros-humble-nav2-collision-monitor ros-humble-nav2-waypoint-follower \
  ros-humble-nav2-velocity-smoother ros-humble-nav2-rviz-plugins \
  ros-humble-pcl-ros ros-humble-cv-bridge ros-humble-image-transport \
  ros-humble-tf2-eigen ros-humble-gazebo-ros-pkgs gazebo libgazebo-dev \
  libeigen3-dev libopencv-dev libpcl-dev liboctomap-dev nlohmann-json3-dev

# ============================================
# STEP 3: Install GTSAM from source
# ============================================
print_step "3/7 Installing GTSAM from source..."

cd /tmp
if [ ! -d "gtsam" ]; then
    git clone https://github.com/borglab/gtsam.git
fi
cd gtsam
git checkout 4.2.0 2>/dev/null || true
mkdir -p build && cd build
cmake -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF ..
make -j$(nproc)
sudo make install
sudo ldconfig

# ============================================
# STEP 4: Setup project directory and fix CMakeLists.txt
# ============================================
print_step "4/7 Setting up project and fixing CMakeLists.txt..."

SIH_DIR="$HOME/SIH"
mkdir -p "$SIH_DIR"
cd "$SIH_DIR"

# The CMakeLists.txt should already be fixed in your repo
# If not, the fixed version is already in your repo

# ============================================
# STEP 5: Generate crater heightmap
# ============================================
print_step "5/7 Generating crater heightmap..."


# python3 create_crater_heightmap.py

# ============================================
# STEP 5: Build thermal camera plugin
# ============================================
print_step "5.5/7 Building thermal camera plugin..."

mkdir -p ~/home/blueberry/Desktop/Project/SIH/sim/models/thermal_camera_plugin/build
cd ~/Desktop/Project/SIH/sim/models/thermal_camera_plugin/build
cmake .. -DCMAKE_PREFIX_PATH="/opt/ros/humble:/usr"
make -j$(nproc)
sudo make install

# ============================================
# STEP 6: Build the project
# ============================================
print_step "6/7 Building the project..."

cd ~/SIH
source /opt/ros/humble/setup.bash
colcon build --packages-select lunar_slam --symlink-install --event-handlers console_direct+

# Source the workspace
source install/setup.bash

# ============================================
# STEP 7: Generate crater heightmap and launch
# ============================================
print_step "7/7 Generating crater and launching..."

cd ~/SIH
python3 create_crater_heightmap.py

echo ""
echo -e "${GREEN}==========================================${NC}"
echo -e "${GREEN}Setup complete!${NC}"
echo -e "${GREEN}==========================================${NC}"
echo ""
echo "To launch the simulation:"
echo "  source /opt/ros/humble/setup.bash"
echo "  cd ~/SIH"
echo "  source install/setup.bash"
echo "  ros2 launch lunar_slam lunar_slam.launch.py"
echo ""
echo "To collect dataset (in another terminal):"
echo "  ros2 run lunar_slam mobility_dataset_collector"
echo ""
echo "To teleoperate the rover:"
echo "  ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r /cmd_vel:=/cmd_vel"

# Source the workspace for current shell
source install/setup.bash 2>/dev/null || true

echo -e "${GREEN}Setup complete! Ready to launch.${NC}"
EOF
chmod +x /home/blueberry/setup_sih.sh
echo -e "${GREEN}Created setup_sih.sh${NC}"
