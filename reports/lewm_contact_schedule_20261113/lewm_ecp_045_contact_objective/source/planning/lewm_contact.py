"""Steer a triggered PushObj intervention toward a useful contact, then the pusher goal."""

import heapq
import json
from pathlib import Path

import numpy as np
from shapely.affinity import affine_transform
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import nearest_points, unary_union

from env.pusht.pusht_env import PushTEnv
from .lewm_proprio import LeWMProprioMPCPlanner


def rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def wrapped(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def first_waypoint(start, target, obstacle):
    """Route a pusher center around the buffered object to a contact setup point."""
    if np.linalg.norm(start - target) < 1e-6:
        return target
    if obstacle.contains(Point(start)):
        point = nearest_points(Point(start), obstacle.boundary)[1]
        boundary = np.asarray(point.coords[0])
        outward = boundary - start
        return boundary + outward / max(np.linalg.norm(outward), 1e-6)

    def clear(a, b):
        return LineString([a, b]).relate_pattern(obstacle, "F********")

    if clear(start, target):
        return target
    nodes = np.vstack((start, target, np.asarray(obstacle.exterior.coords)[:-1]))
    queue = [(0.0, 0, 0)]
    distances = {0: 0.0}
    while queue:
        distance, index, first = heapq.heappop(queue)
        if distance > distances[index]:
            continue
        if index == 1:
            return nodes[first]
        for nxt in range(1, len(nodes)):
            if nxt == index or not clear(nodes[index], nodes[nxt]):
                continue
            cost = distance + np.linalg.norm(nodes[nxt] - nodes[index])
            if cost < distances.get(nxt, np.inf):
                distances[nxt] = cost
                heapq.heappush(queue, (cost, nxt, nxt if index == 0 else first))
    return start


class ContactGuide:
    def __init__(self, shape, staging_clearance, workspace):
        env = PushTEnv(shape="+" if shape == "plus" else shape)
        env.reset()
        body = env.block
        self.cog = np.asarray(tuple(body.center_of_gravity))
        self.radius = env.agent.shapes.copy().pop().radius
        geom = unary_union([Polygon([tuple(v) for v in poly.get_vertices()]) for poly in body.shapes])
        self.geometry = orient(geom)
        self.obstacle = self.geometry.buffer(self.radius + 2, join_style=2)
        boundary = np.asarray(self.geometry.exterior.coords)
        edge = np.diff(boundary, axis=0)
        normal = np.stack((-edge[:, 1], edge[:, 0]), axis=1) / np.linalg.norm(edge, axis=1)[:, None]
        fraction = np.array([0.25, 0.5, 0.75])
        contact = (boundary[:-1, None] + fraction[None, :, None] * edge[:, None]).reshape(-1, 2)
        force = np.repeat(normal, len(fraction), axis=0)
        staging = contact - (self.radius + staging_clearance) * force
        valid = np.array([not self.obstacle.contains(Point(p)) for p in staging])
        self.contact, self.force, self.staging = contact[valid], force[valid], staging[valid]
        moment = np.cross(self.contact - self.cog, self.force)
        self.torque = moment / np.max(np.linalg.norm(boundary - self.cog, axis=1))
        self.workspace = np.asarray(workspace)
        if not len(self.contact):
            raise ValueError(f"No reachable contact candidates for {shape}")

    def direction(self, pusher, state, goal, position_tol, angle_tol):
        origin = state[2:4]
        transform = rotation(state[4])
        contact = self.contact @ transform.T + origin
        force = self.force @ transform.T
        staging = self.staging @ transform.T + origin
        delta = goal[2:4] + rotation(goal[4]) @ self.cog - (origin + transform @ self.cog)
        delta_norm = np.linalg.norm(delta)
        translate = force @ (delta / max(delta_norm, 1e-6))
        angle_error = wrapped(goal[4] - state[4])
        torque = np.sign(angle_error) * self.torque
        rotate_first = np.linalg.norm(goal[2:4] - origin) < position_tol and abs(angle_error) >= angle_tol
        torque_weight = min(abs(angle_error) / angle_tol, 1)
        if rotate_first:
            score = 0.2 * translate + torque
        else:
            score = translate + 0.35 * torque_weight * torque
        score -= 0.1 * np.minimum(np.linalg.norm(staging - pusher, axis=1) / 100, 3)
        inside = ((staging >= self.workspace[0]) & (staging <= self.workspace[1])).all(axis=1)
        if not inside.any():
            return np.zeros(2), "no_contact_waypoint"
        i = int(np.argmax(np.where(inside, score, -np.inf)))
        radial = pusher - contact[i]
        axial = -float(radial @ force[i])
        lateral = np.linalg.norm(radial + axial * force[i])
        if self.radius - 2 <= axial <= self.radius + 13 and lateral <= 8:
            return 75 * force[i], "push"
        obstacle = affine_transform(self.obstacle, [*transform.flatten(), *origin])
        waypoint = first_waypoint(pusher, staging[i], obstacle)
        return waypoint - pusher, "approach_contact"


class LeWMContactMPCPlanner(LeWMProprioMPCPlanner):
    def __init__(self, contact, **kwargs):
        super().__init__(**kwargs)
        self.position_tol = float(contact["position_tol_px"])
        self.angle_tol = np.radians(float(contact["angle_tol_deg"]))
        self.guide = ContactGuide(contact["shape"], float(contact["staging_clearance_px"]), contact["workspace_px"])

    def _plan_with_bias(self, obs_0, obs_g, actions=None, step=None):
        _, self._physical_state = self.evaluator.get_init_cond()
        self._physical_goal = self.evaluator.state_g
        error = self._physical_goal - self._physical_state
        object_position_error = np.linalg.norm(error[:, 2:4], axis=1)
        object_angle_error = np.abs(wrapped(error[:, 4]))
        self._object_ready = (object_position_error < self.position_tol) & (object_angle_error < self.angle_tol)
        self._bias_phase = {}
        result = super()._plan_with_bias(obs_0, obs_g, actions=actions, step=step)
        for entry in self._applied:
            sample = entry["sample_idx"]
            phase = self._bias_phase[sample]
            entry["phase"] = phase
            entry["direction_px"] = entry.pop("pusher_to_goal_px")
            entry["object_position_error_px"] = float(object_position_error[sample])
            entry["object_angle_error_deg"] = float(np.degrees(object_angle_error[sample]))
        if self._bias_phase:
            with Path("lewm_contact_guidance.jsonl").open("a") as file:
                file.write(json.dumps({"replan": self.iter, "phase_by_sample": self._bias_phase}) + "\n")
        return result

    def _proprio_active(self):
        return super()._proprio_active() & self._object_ready

    def _bias_direction(self, obs_0, obs_g, indices):
        direction = super()._bias_direction(obs_0, obs_g, indices).astype(np.float64)
        for j, i in enumerate(indices):
            if self._object_ready[i]:
                self._bias_phase[int(i)] = "pusher_goal"
            else:
                direction[j], self._bias_phase[int(i)] = self.guide.direction(
                    obs_0["proprio"][i, -1, :2], self._physical_state[i], self._physical_goal[i],
                    self.position_tol, self.angle_tol,
                )
        return direction
