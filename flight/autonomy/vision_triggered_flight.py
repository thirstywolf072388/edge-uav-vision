import asyncio
import cv2
import time
from ultralytics import YOLO
from mavsdk import System

TARGET_CLASSES = ["person", "bird", "dog", "cat", "cow", "bear"]
COOLDOWN_SECONDS = 60
METERS_PER_DEGREE_LAT = 111111

model = YOLO("yolov8n.pt")


async def fly_mission(drone):
    """Reuses the verified waypoint logic: takeoff, fly out, return, land."""
    async for position in drone.telemetry.position():
        home_lat = position.latitude_deg
        home_lon = position.longitude_deg
        home_alt = position.absolute_altitude_m
        break

    garden_distance_m = 50
    lat_offset = garden_distance_m / METERS_PER_DEGREE_LAT
    garden_lat = home_lat + lat_offset
    garden_lon = home_lon
    flight_altitude = home_alt + 10

    print("  [MISSION] Arming...")
    await drone.action.arm()
    print("  [MISSION] Taking off...")
    await drone.action.takeoff()
    await asyncio.sleep(8)

    print(f"  [MISSION] Flying to target area ({garden_distance_m}m)...")
    await drone.action.goto_location(garden_lat, garden_lon, flight_altitude, 0)
    await asyncio.sleep(20)

    print("  [MISSION] Returning home...")
    await drone.action.goto_location(home_lat, home_lon, flight_altitude, 0)
    await asyncio.sleep(20)

    print("  [MISSION] Landing...")
    await drone.action.land()
    await asyncio.sleep(10)
    print("  [MISSION] Complete.")


async def watch_and_trigger(drone):
    video_path = "data/youtube_test.mkv"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    print("Connecting to drone...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Connected to drone")
            break

    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("Position lock achieved. Watching for targets...")
            break

    last_trigger_time = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        results = model(frame, verbose=False, conf=0.5)

        for box in results[0].boxes:
            class_name = model.names[int(box.cls)]
            confidence = float(box.conf)

            if class_name in TARGET_CLASSES:
                now = time.time()
                if now - last_trigger_time >= COOLDOWN_SECONDS:
                    print(f"\n🚨 Frame {frame_count}: '{class_name}' detected "
                          f"({confidence:.2f}). Launching mission...\n")
                    last_trigger_time = now
                    await fly_mission(drone)
                    print("\nResuming watch...\n")
                    break  # only one trigger per frame

    cap.release()
    print(f"Video ended. Processed {frame_count} frames.")


async def run():
    drone = System()
    await drone.connect(system_address="udp://:14540")
    await watch_and_trigger(drone)


if __name__ == "__main__":
    asyncio.run(run())
