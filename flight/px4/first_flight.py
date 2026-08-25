import asyncio
from mavsdk import System

async def run():
    drone = System()
    # Connect to the PX4 SITL instance running in your other terminal
    await drone.connect(system_address="udp://:14540")

    print("Waiting for drone connection...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Connected to drone")
            break

    print("Waiting for global position + home position lock...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("Position lock achieved")
            break

    print("Arming...")
    await drone.action.arm()

    print("Taking off...")
    await drone.action.takeoff()
    await asyncio.sleep(10)  # let it climb and hover

    print("Landing...")
    await drone.action.land()
    await asyncio.sleep(10)

    print("Done.")

if __name__ == "__main__":
    asyncio.run(run())
