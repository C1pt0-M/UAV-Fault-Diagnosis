#!/usr/bin/env bash
set -e

# RflySim PX4 SITL launcher with an additional direct PX4 -> QGC MAVLink link.
# This preserves the RflySim/CopterSim link and opens a normal UDP link for QGC.

RFLY_SIL_UDP_RECEIVE=16540
RFLY_SIL_UDP_SEND=17540
RFLY_GCS_UDP_SEND=18570
RFLY_SIH_UDP_SEND=19450

sitl_num=2
[ -n "$1" ] && sitl_num="$1"

src_path="${RFLY_PX4_FIRMWARE_DIR:-/mnt/d/PX4PSP/Firmware}"
build_path=${src_path}/build/px4_sitl_default

PX4SiltFrame=iris
[ -n "$3" ] && [ "$3" != "0" ] && [ "$3" != "1" ] && PX4SiltFrame="$3"
export PX4_SIM_MODEL=$PX4SiltFrame

airframe_path=${build_path}/etc/init.d/airframes
airframe_path_sitl=${build_path}/etc/init.d-posix/airframes

FileN0=$(find ${airframe_path_sitl} -name "*_${PX4SiltFrame}" | sed -e 's|.*/||' | grep -E "^[0-9]{2,}_${PX4SiltFrame}$" | head -n 1)
FileN1=$(find ${airframe_path_sitl} -name "*[0-9]_gazebo-classic_${PX4SiltFrame}" | head -n 1 | sed -e 's|/.*/||')
FileN2=$(find ${airframe_path} -name "*[0-9]_${PX4SiltFrame}" | head -n 1 | sed -e 's|/.*/||')
if [ -n "$FileN0" ]; then
    echo "Using Airframe File: ${FileN0}"
elif [ -n "$FileN1" ]; then
    echo "Using Gazebo-classic Airframe File: ${FileN1}"
    PX4SiltFrame="gazebo-classic_${PX4SiltFrame}"
    export PX4_SIM_MODEL=$PX4SiltFrame
elif [ -n "$FileN2" ]; then
    echo "Using Airframe File: ${FileN2}"
    if [ ! -f ${airframe_path_sitl}/${FileN2} ]; then
        echo "Copy Airframe file ${FileN2}"
        cp -rf ${airframe_path}/${FileN2} ${airframe_path_sitl}/${FileN2}
    fi
else
    echo "Wrong Airframe Name: ${PX4SiltFrame}${FileN0}"
    exit 1
fi

rcs_script_path=${build_path}/etc/init.d-posix
rcs_romfs_path=${src_path}/ROMFS/px4fmu_common/init.d-posix

RFLY_SIL_UDP_RECEIVE=$(($RFLY_SIL_UDP_RECEIVE-1))
RFLY_SIL_UDP_SEND=$(($RFLY_SIL_UDP_SEND-1))
RFLY_GCS_UDP_SEND=$(($RFLY_GCS_UDP_SEND-1))
RFLY_SIH_UDP_SEND=$(($RFLY_SIH_UDP_SEND-1))

n=1
[ -n "$2" ] && n="$2"
sitl_num=$(($sitl_num + $n))

cat ${rcs_romfs_path}/px4-rc.simulator | sed s/4560/4559/ > $rcs_script_path/px4-rc.simulator
# Keep the stock simulator startup. CopterSim listens on TCP 4560 for the PX4
# lockstep motion/sensor channel; the MAVLink UDP link alone only carries the
# companion telemetry/control traffic.
cat ${rcs_romfs_path}/rcS | sed s/px4_instance+1/px4_instance/ | \
    sed s/'$px4_instance'/'$((px4_instance-1))'/ > $rcs_script_path/rcS

generate_rfly_mavlink_rc() {
    echo 'udp_rfly_remote_port=${RFLY_SIM_FORWARD_PORT:-$udp_offboard_port_remote}'
    cat ${rcs_romfs_path}/px4-rc.mavlink | sed s/14540/${RFLY_SIL_UDP_RECEIVE}/ | \
        sed s/14580/${RFLY_SIL_UDP_SEND}/ | \
        sed s/18570/${RFLY_GCS_UDP_SEND}/ | \
        sed s/19450/${RFLY_SIH_UDP_SEND}/ | \
        sed s/'-m onboard -o $udp_offboard_port_remote'/'-m onboard -o $udp_rfly_remote_port -t $RFLY_SIM_HOST_IP'/ | \
        sed 's/^.*udp_sihsim_port_remote=.*$/# &/' | \
        sed 's/-o \$udp_sihsim_port_remote//g' | \
        sed s/'\[ $px4_instance -gt 9 \]'/'#\[ $px4_instance -gt 9 \]'/
}

if [ $sitl_num -le 9 ]; then
    generate_rfly_mavlink_rc > $rcs_script_path/px4-rc.mavlink
else
    generate_rfly_mavlink_rc | \
        sed s/'mavlink stream -r 50 -s POSITION_TARGET_LOCAL_NED'/'# mavlink stream -r 50 -s POSITION_TARGET_LOCAL_NED'/ | \
        sed s/'mavlink stream -r 50 -s ATTITUDE_QUATERNION'/'# mavlink stream -r 50 -s ATTITUDE_QUATERNION'/ | \
        sed s/'mavlink stream -r 50 -s ATTITUDE_TARGET'/'# mavlink stream -r 50 -s ATTITUDE_TARGET'/ | \
        sed s/'mavlink stream -r 50 -s SERVO_OUTPUT_RAW_0'/'# mavlink stream -r 50 -s SERVO_OUTPUT_RAW_0'/ | \
        sed s/'mavlink stream -r 20 -s RC_CHANNELS'/'# mavlink stream -r 20 -s RC_CHANNELS'/ | \
        sed s/'mavlink stream -r 10 -s OPTICAL_FLOW_RAD'/'# mavlink stream -r 10 -s OPTICAL_FLOW_RAD'/ > $rcs_script_path/px4-rc.mavlink
fi

cat >> $rcs_script_path/px4-rc.mavlink <<'EOF'

# Dashboard wrapper direct QGC link. Keeps RflySim/CopterSim on 17540->16540,
# while exposing a normal PX4 vehicle link to QGroundControl.
if [ -n "$RFLY_QGC_HOST_IP" ]; then
    udp_qgc_direct_port_local=$((14550+px4_instance))
    udp_qgc_direct_port_remote=${RFLY_QGC_UDP_PORT:-14600}
    mavlink start -x -u $udp_qgc_direct_port_local -r 4000000 -o $udp_qgc_direct_port_remote -t $RFLY_QGC_HOST_IP -f
fi

# Separate PX4 -> Dashboard MAVLink feed for pymavlink telemetry enrichment.
# QGC owns 14600; the Dashboard bridge listens on 14031 by default.
if [ -n "$RFLY_DASHBOARD_HOST_IP" ]; then
    udp_dashboard_pymav_port_local=$((14650+px4_instance))
    udp_dashboard_pymav_port_remote=${RFLY_DASHBOARD_PYMAV_UDP_PORT:-14031}
    mavlink start -x -u $udp_dashboard_pymav_port_local -r 4000000 -o $udp_dashboard_pymav_port_remote -t $RFLY_DASHBOARD_HOST_IP -f
    mavlink stream -u $udp_dashboard_pymav_port_local -s HIGHRES_IMU -r 50
    mavlink stream -u $udp_dashboard_pymav_port_local -s ACTUATOR_CONTROL_TARGET0 -r 50
    mavlink stream -u $udp_dashboard_pymav_port_local -s ATTITUDE_TARGET -r 50
    mavlink stream -u $udp_dashboard_pymav_port_local -s ATTITUDE -r 50
    mavlink stream -u $udp_dashboard_pymav_port_local -s LOCAL_POSITION_NED -r 50
    mavlink stream -u $udp_dashboard_pymav_port_local -s VIBRATION -r 10
    mavlink stream -u $udp_dashboard_pymav_port_local -s ESC_STATUS -r 10
fi
EOF

sleep 1

copyFile="true"
while [ $n -lt $sitl_num ]; do
    working_dir="$build_path/instance_$n"
    [ ! -d "$working_dir" ] && mkdir -p "$working_dir" && copyFile="false"
    sleep 2
    pushd "$working_dir" >/dev/null
    echo "starting instance $n in $(pwd)"
    ../bin/px4 -i $n -d "$build_path/etc" -s etc/init.d-posix/rcS >out.log 2>err.log &
    popd >/dev/null

    n=$(($n + 1))
    sleep 1
done

sleep 5
if [[ "$(uname -a)" != *"WSL2"* || "$copyFile" != "false" ]]; then
    echo Restoring rcS files
    cp -rf ${rcs_romfs_path}/rcS $rcs_script_path/rcS
    cp -rf ${rcs_romfs_path}/px4-rc.mavlink $rcs_script_path/px4-rc.mavlink
    cp -rf ${rcs_romfs_path}/px4-rc.simulator $rcs_script_path/px4-rc.simulator
else
    echo "First run SITL in WSL2, Please wait for a while..."
fi

echo PX4 instances start finished
