import asyncio
from mavsdk import System

METERS_PER_DEGREE_LAT = 111111

async def print_status(drone, label):
    """Print real telemetry so we know ground truth, not assumptions."""
    async for position in drone.telemetry.position():
        print(f"[{label}] lat={position.latitude_deg:.6f} "
              f"lon={position.longitude_deg:.6f} "
              f"alt={position.relative_altitude_m:.2f}m")
        break
    async for in_air in drone.telemetry.in_air():
        print(f"[{label}] in_air={in_air}")
        break
    async for flight_mode in drone.telemetry.flight_mode():
        print(f"[{label}] flight_mode={flight_mode}")
        break

async def wait_until_in_air(drone, timeout=15):
    print("Waiting for confirmed takeoff...")
    elapsed = 0
    async for in_air in drone.telemetry.in_air():
        if in_air:
            print("Confirmed: drone is airborne")
            return True
        elapsed += 1
        if elapsed > timeout:
            print("TIMEOUT: never confirmed airborne")
            return False

async def run():
    drone = System()
    await drone.connect(system_address="udp://:14540")

    print("Waiting for drone connection...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Connected to drone")
            break

    print("Waiting for position lock...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("Position lock achieved")
            break

    async for position in drone.telemetry.position():
        home_lat = position.latitude_deg
        home_lon = position.longitude_deg
        home_alt = position.absolute_altitude_m
        print(f"Home position: {home_lat}, {home_lon}, alt {home_alt}m")
        break

    garden_distance_m = 50
    lat_offset = garden_distance_m / METERS_PER_DEGREE_LAT
    garden_lat = home_lat + lat_offset
    garden_lon = home_lon
    flight_altitude = home_alt + 10

    print(f"Garden waypoint: {garden_lat}, {garden_lon}")

    print("Arming...")
    await drone.action.arm()
    await print_status(drone, "post-arm")

    print("Taking off...")
    await drone.action.takeoff()
    await asyncio.sleep(3)
    await print_status(drone, "post-takeoff-3s")

    airborne = await wait_until_in_air(drone)
    if not airborne:
        print("ABORTING: drone never confirmed airborne. Check PX4 log for cause.")
        return

    print(f"Flying to garden waypoint ({garden_distance_m}m away)...")
    await drone.action.goto_location(garden_lat, garden_lon, flight_altitude, 0)

    # Poll every 2 seconds instead of blindly sleeping, so we can SEE progress
    for i in range(10):
        await asyncio.sleep(2)
        await print_status(drone, f"enroute-check-{i}")

    print("Returning home...")
    await drone.action.goto_location(home_lat, home_lon, flight_altitude, 0)

    for i in range(10):
        await asyncio.sleep(2)
        await print_status(drone, f"return-check-{i}")

    print("Landing...")
    await drone.action.land()
    await asyncio.sleep(10)
    await print_status(drone, "final")

    print("Mission complete.")

if __name__ == "__main__":
    asyncio.run(run())
