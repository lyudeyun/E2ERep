#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the ScenarioManager implementations.
It must not be modified and is for reference only!
"""

from __future__ import print_function
import math
import signal
import sys
import time

import py_trees
import carla
import threading

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog

from leaderboard.autoagents.agent_wrapper import AgentWrapperFactory, AgentError, TickRuntimeError
from leaderboard.envs.sensor_interface import SensorReceivedNoData
from leaderboard.utils.result_writer import ResultOutputProvider

# Standing-still rule (early termination, off unless ScenarioManager.still_limit is set): every
# STILL_EVERY ticks the ego counts as standing still if its speed is below STILL_SPEED; the time it
# has been standing still without a traffic reason (waiting_exempt) is accumulated, and the route
# is ended once it reaches still_limit seconds of game time.
STILL_EVERY = 10        # ticks (0.5 s at 20 Hz)
STILL_SPEED = 0.1       # m/s
NEAR_RADIUS = 60.0      # m, actors considered for "traffic moving nearby"
_TL_RANK = {'Red': 3, 'Yellow': 2, 'Green': 1}


class StillSignals(object):
    """Per-route signals of the standing-still rule, from the simulator (map, lights, actors)."""

    def __init__(self):
        self.prev = {}     # actor id -> (x, y) at the previous update
        self.types = {}    # vehicle / walker actor ids -> type id

    def update(self, ego, tick_count):
        """
        v                 ego speed (m/s)
        tl_at, tl_state   ego at a traffic light / its state (Red, Yellow, Green, Off, Unknown)
        tl_ahead          worst state of the traffic lights on the lane ahead within 30 m (or None)
        junction          0 in a junction, 5/10/15/20 = junction that many m ahead, None = none within 20 m
        n_near            vehicles + walkers within NEAR_RADIUS m of the ego
        max_disp          largest displacement of those actors since the previous update (m)
        near_moving       distance to the nearest of them that moved > 0.25 m since then (or None)
        front_moving      the same, only actors within 25 m in front (within 60 deg of the heading)
        """
        world = CarlaDataProvider.get_world()
        if tick_count % 40 == 0 or not self.types:
            self.types = {a.id: a.type_id for a in world.get_actors() if a.type_id.startswith(('vehicle.', 'walker.'))}
        et = ego.get_transform()
        el, yaw = et.location, math.radians(et.rotation.yaw)
        vel = ego.get_velocity()
        cur, max_disp, near_moving, front_moving, n_near = {}, 0.0, None, None, 0
        for a in world.get_snapshot():
            if a.id == ego.id or a.id not in self.types:
                continue
            loc = a.get_transform().location
            cur[a.id] = (loc.x, loc.y)
            dx, dy = loc.x - el.x, loc.y - el.y
            d = math.hypot(dx, dy)
            if d > NEAR_RADIUS:
                continue
            n_near += 1
            if a.id in self.prev:
                disp = math.hypot(loc.x - self.prev[a.id][0], loc.y - self.prev[a.id][1])
                max_disp = max(max_disp, disp)
                if disp > 0.25:
                    near_moving = d if near_moving is None else min(near_moving, d)
                    bearing = (math.atan2(dy, dx) - yaw + math.pi) % (2 * math.pi) - math.pi
                    if d < 25 and abs(bearing) < math.radians(60):
                        front_moving = d if front_moving is None else min(front_moving, d)
        self.prev = cur

        wp = CarlaDataProvider.get_map().get_waypoint(el)
        lights = [str(tl.get_state()) for tl in world.get_traffic_lights_from_waypoint(wp, 30.0)]
        junction = 0 if wp.is_junction else next(
            (d for d in (5, 10, 15, 20) if any(w.is_junction for w in wp.next(d))), None)
        return dict(v=math.hypot(vel.x, vel.y), tl_at=bool(ego.is_at_traffic_light()),
                    tl_state=str(ego.get_traffic_light_state()),
                    tl_ahead=max(lights, key=lambda x: _TL_RANK.get(x, 0)) if lights else None,
                    junction=junction, n_near=n_near, max_disp=max_disp,
                    near_moving=near_moving, front_moving=front_moving)


def waiting_exempt(s):
    """Standing still for a traffic reason: a red/yellow light ahead or at the ego, or in or near
    (<= 20 m) a junction while traffic moves within NEAR_RADIUS m."""
    red = s['tl_ahead'] in ('Red', 'Yellow') or (s['tl_at'] and s['tl_state'] in ('Red', 'Yellow'))
    return red or (s['junction'] is not None and s['near_moving'] is not None)


class ScenarioManager(object):

    """
    Basic scenario manager class. This class holds all functionality
    required to start, run and stop a scenario.

    The user must not modify this class.

    To use the ScenarioManager:
    1. Create an object via manager = ScenarioManager()
    2. Load a scenario via manager.load_scenario()
    3. Trigger the execution of the scenario manager.run_scenario()
       This function is designed to explicitly control start and end of
       the scenario execution
    4. If needed, cleanup with manager.stop_scenario()
    """

    def __init__(self, timeout, statistics_manager, debug_mode=0):
        """
        Setups up the parameters, which will be filled at load_scenario()
        """
        self.route_index = None
        self.scenario = None
        self.scenario_tree = None
        self.ego_vehicles = None
        self.other_actors = None

        self._debug_mode = debug_mode
        self._agent_wrapper = None
        self._running = False
        self.time_limit = None  # game time (s) at which the route is ended, see leaderboard_evaluator.py
        self.still_limit = None  # standing-still rule: seconds without a traffic reason; None = off
        self.stopped_still = False
        self._timestamp_last_run = 0.0
        self._timeout = float(timeout)

        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = 0.0
        self.start_game_time = 0.0
        self.end_system_time = 0.0
        self.end_game_time = 0.0

        self._watchdog = None
        self._agent_watchdog = None
        self._scenario_thread = None

        self._statistics_manager = statistics_manager

        self.tick_count = 0

        # Use the callback_id inside the signal handler to allow external interrupts
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, signum, frame):
        """
        Terminate scenario ticking when receiving a signal interrupt
        """
        if self._agent_watchdog and not self._agent_watchdog.get_status():
            raise RuntimeError("Agent took longer than {}s to send its command".format(self._timeout))
        elif self._watchdog and not self._watchdog.get_status():
            raise RuntimeError("The simulation took longer than {}s to update".format(self._timeout))
        self._running = False

    def cleanup(self):
        """
        Reset all parameters
        """
        self._timestamp_last_run = 0.0
        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = 0.0
        self.start_game_time = 0.0
        self.end_system_time = 0.0
        self.end_game_time = 0.0

        self._spectator = None
        self._watchdog = None
        self._agent_watchdog = None

    def load_scenario(self, scenario, agent, route_index, rep_number):
        """
        Load a new scenario
        """

        GameTime.restart()
        self._agent_wrapper = AgentWrapperFactory.get_wrapper(agent)
        self.route_index = route_index
        self.scenario = scenario
        self.scenario_tree = scenario.scenario_tree
        self.ego_vehicles = scenario.ego_vehicles
        self.other_actors = scenario.other_actors
        self.repetition_number = rep_number

        self._spectator = CarlaDataProvider.get_world().get_spectator()

        # To print the scenario tree uncomment the next line
        # py_trees.display.render_dot_tree(self.scenario_tree)

        self._agent_wrapper.setup_sensors(self.ego_vehicles[0])

    def build_scenarios_loop(self, debug):
        """
        Keep periodically trying to start the scenarios that are close to the ego vehicle
        Additionally, do the same for the spawned vehicles
        """
        while self._running:
            self.scenario.build_scenarios(self.ego_vehicles[0], debug=debug)
            self.scenario.spawn_parked_vehicles(self.ego_vehicles[0])
            time.sleep(1)

    def run_scenario(self):
        """
        Trigger the start of the scenario and wait for it to finish/fail
        """
        self.start_system_time = time.time()
        self.start_game_time = GameTime.get_time()

        # Detects if the simulation is down
        self._watchdog = Watchdog(self._timeout)
        self._watchdog.start()

        # Stop the agent from freezing the simulation
        self._agent_watchdog = Watchdog(self._timeout)
        self._agent_watchdog.start()

        self._running = True

        # Thread for build_scenarios
        self._scenario_thread = threading.Thread(target=self.build_scenarios_loop, args=(self._debug_mode > 0, ))
        self._scenario_thread.start()

        self.stopped_still = False
        still, still_s, last_tick, last_t = StillSignals(), 0.0, None, GameTime.get_time()
        while self._running:
            self._tick_scenario()
            if self.time_limit is not None and GameTime.get_time() >= self.time_limit:
                self._running = False  # same exit as a finished scenario tree
            if self.still_limit and self._running and self.tick_count % STILL_EVERY == 0 \
                    and self.tick_count != last_tick:
                now = GameTime.get_time()
                s = still.update(self.ego_vehicles[0], self.tick_count)
                standing = s['v'] < STILL_SPEED and not waiting_exempt(s)
                still_s = still_s + (now - last_t) if standing else 0.0
                last_tick, last_t = self.tick_count, now
                if still_s >= self.still_limit:
                    self.stopped_still = True
                    self._running = False

    def _tick_scenario(self):
        """
        Run next tick of scenario and the agent and tick the world.
        """
        if self._running and self.get_running_status():
            CarlaDataProvider.get_world().tick(self._timeout)

        timestamp = CarlaDataProvider.get_world().get_snapshot().timestamp

        if self._timestamp_last_run < timestamp.elapsed_seconds and self._running:
            self._timestamp_last_run = timestamp.elapsed_seconds

            self._watchdog.update()
            # Update game time and actor information
            GameTime.on_carla_tick(timestamp)
            CarlaDataProvider.on_carla_tick()
            self.tick_count += 1
            self._watchdog.pause()

            if self.tick_count > 4000:
                raise TickRuntimeError("RuntimeError, tick_count > 4000")

            try:
                self._agent_watchdog.resume()
                self._agent_watchdog.update()
                ego_action = self._agent_wrapper()
                self._agent_watchdog.pause()

            # Special exception inside the agent that isn't caused by the agent
            except SensorReceivedNoData as e:
                raise RuntimeError(e)

            except Exception as e:
                raise AgentError(e)

            self._watchdog.resume()
            self.ego_vehicles[0].apply_control(ego_action)

            # Tick scenario. Add the ego control to the blackboard in case some behaviors want to change it
            py_trees.blackboard.Blackboard().set("AV_control", ego_action, overwrite=True)
            self.scenario_tree.tick_once()

            if self._debug_mode > 1:
                self.compute_duration_time()

                # Update live statistics
                self._statistics_manager.compute_route_statistics(
                    self.route_index,
                    self.scenario_duration_system,
                    self.scenario_duration_game,
                    failure_message=""
                )
                self._statistics_manager.write_live_results(
                    self.route_index,
                    self.ego_vehicles[0].get_velocity().length(),
                    ego_action,
                    self.ego_vehicles[0].get_location()
                )

            if self._debug_mode > 2:
                print("\n")
                py_trees.display.print_ascii_tree(self.scenario_tree, show_status=True)
                sys.stdout.flush()

            if self.scenario_tree.status != py_trees.common.Status.RUNNING:
                self._running = False

            ego_trans = self.ego_vehicles[0].get_transform()
            self._spectator.set_transform(carla.Transform(ego_trans.location + carla.Location(z=70),
                                                          carla.Rotation(pitch=-90)))

    def get_running_status(self):
        """
        returns:
           bool: False if watchdog exception occured, True otherwise
        """
        if self._watchdog:
            return self._watchdog.get_status()
        return True

    def stop_scenario(self):
        """
        This function triggers a proper termination of a scenario
        """
        if self._watchdog:
            self._watchdog.stop()

        if self._agent_watchdog:
            self._agent_watchdog.stop()

        self.compute_duration_time()

        if self.get_running_status():
            if self.scenario is not None:
                self.scenario.terminate()

            if self._agent_wrapper is not None:
                self._agent_wrapper.cleanup()
                self._agent_wrapper = None

            self.analyze_scenario()

        # Make sure the scenario thread finishes to avoid blocks
        self._running = False
        self._scenario_thread.join()
        self._scenario_thread = None

    def compute_duration_time(self):
        """
        Computes system and game duration times
        """
        self.end_system_time = time.time()
        self.end_game_time = GameTime.get_time()

        self.scenario_duration_system = self.end_system_time - self.start_system_time
        self.scenario_duration_game = self.end_game_time - self.start_game_time

    def analyze_scenario(self):
        """
        Analyzes and prints the results of the route
        """
        ResultOutputProvider(self)
