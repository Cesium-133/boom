import numpy as np
import plotly.graph_objects as go

from .kinematics import (
    calculate_missile_position,
    calculate_uav_position,
    calculate_decoy_trajectory,
    calculate_smoke_cloud_position,
)


class SimulationVisualizer:
    def __init__(self, config: dict):
        self.config = config
        self.fig = go.Figure()

        sim = self.config["simulation"]
        self.duration: float = float(sim["duration"])
        self.dt: float = float(sim["time_step"])
        self.times = np.arange(0.0, self.duration + 1e-9, self.dt)

        self.constants = self.config["constants"]
        self.targets = self.config["targets"]
        self.missiles = self.config.get("missiles", {})
        self.uavs = self.config.get("uavs", {})

        # Derived common values
        self.true_target_center = np.array(self.targets["true_target"]["base_center"], dtype=float) + np.array([0.0, 0.0, self.targets["true_target"]["height"] / 2.0])

        # Rendering options with safe defaults for compatibility
        rend = self.config.get("rendering", {})
        self.use_cylinder_mesh = bool(rend.get("use_cylinder_mesh", True))
        self.use_cloud_mesh = bool(rend.get("use_cloud_mesh", True))
        self.cylinder_slices = int(rend.get("cylinder_slices", 48))
        self.sphere_rings = int(rend.get("sphere_rings", 10))
        self.sphere_sectors = int(rend.get("sphere_sectors", 20))

    def _cylinder_wireframe(self, base_center: np.ndarray, radius: float, height: float, slices: int = 60) -> list:
        """Create wireframe cylinder (stable rendering across environments)."""
        theta = np.linspace(0, 2 * np.pi, slices)
        xb = radius * np.cos(theta) + base_center[0]
        yb = radius * np.sin(theta) + base_center[1]
        zb = np.full_like(theta, base_center[2])
        xt = xb
        yt = yb
        zt = np.full_like(theta, base_center[2] + height)

        traces = [
            go.Scatter3d(x=xb, y=yb, z=zb, mode="lines", line=dict(color="green", width=3), name="True Target Base"),
            go.Scatter3d(x=xt, y=yt, z=zt, mode="lines", line=dict(color="green", width=3), name="True Target Top"),
        ]
        # Add a few verticals
        for k in range(0, slices, max(1, slices // 8)):
            traces.append(
                go.Scatter3d(
                    x=[xb[k], xt[k]], y=[yb[k], yt[k]], z=[zb[k], zt[k]],
                    mode="lines", line=dict(color="green", width=2), name="True Target Side",
                    showlegend=False,
                )
            )
        return traces

    def _cylinder_mesh(self, base_center: np.ndarray, radius: float, height: float, slices: int = 48) -> go.Mesh3d:
        """Create a solid cylinder mesh with top and bottom caps.

        Vertices layout:
          - First ring: bottom circle (slices verts)
          - Second ring: top circle (slices verts)
          - Two extra vertices: bottom center, top center
        Faces: triangles for side quads and fan for caps.
        """
        theta = np.linspace(0, 2 * np.pi, slices, endpoint=False)
        xb = radius * np.cos(theta) + base_center[0]
        yb = radius * np.sin(theta) + base_center[1]
        zb = np.full_like(theta, base_center[2])
        xt = xb
        yt = yb
        zt = np.full_like(theta, base_center[2] + height)

        x = np.concatenate([xb, xt, [base_center[0]], [base_center[0]]])
        y = np.concatenate([yb, yt, [base_center[1]], [base_center[1]]])
        z = np.concatenate([zb, zt, [base_center[2]], [base_center[2] + height]])

        bottom_center_index = 2 * slices
        top_center_index = 2 * slices + 1

        i = []
        j = []
        k = []

        # Side faces
        for s in range(slices):
            s_next = (s + 1) % slices
            b0 = s
            b1 = s_next
            t0 = s + slices
            t1 = s_next + slices
            # Two triangles per quad: (b0, b1, t1) and (b0, t1, t0)
            i.extend([b0, b0])
            j.extend([b1, t1])
            k.extend([t1, t0])

        # Bottom cap (fan)
        for s in range(slices):
            s_next = (s + 1) % slices
            i.append(bottom_center_index)
            j.append(s_next)
            k.append(s)

        # Top cap (fan)
        for s in range(slices):
            s_next = (s + 1) % slices
            i.append(top_center_index)
            j.append(s + slices)
            k.append(s_next + slices)

        return go.Mesh3d(
            x=x, y=y, z=z, i=i, j=j, k=k,
            opacity=0.35, color="green", name="True Target",
        )

    def _get_scene_data_at_time(self, t: float) -> dict:
        """
        Compute the positions of ALL objects at a specific time t.
        Returns a dictionary with keys like 'missile_positions', 'uav_positions', 'decoys', 'clouds', 'los'.
        """
        g = float(self.constants["g"])
        missile_speed = float(self.constants["missile_speed"])
        cloud_radius = float(self.constants["smoke_cloud_radius"])
        sink_speed = float(self.constants["smoke_cloud_sink_speed"]) 

        true_target = self.targets["true_target"]
        true_base = np.array(true_target["base_center"], dtype=float)
        true_height = float(true_target["height"]) 
        true_center = true_base + np.array([0.0, 0.0, true_height / 2.0])
        fake_target = np.array(self.targets["fake_target"], dtype=float)

        # Missile(s)
        missile_positions = {}
        los_segments = {}
        for name, m in self.missiles.items():
            init = np.array(m["initial_pos"], dtype=float)
            m_pos = calculate_missile_position(init, true_center, missile_speed, t)
            missile_positions[name] = m_pos
            los_segments[name] = (m_pos, true_center)

        # UAV(s) and their deployments
        uav_positions = {}
        decoy_points = []  # active decoy positions
        cloud_centers = []  # active cloud centers
        cloud_radii = []

        for name, u in self.uavs.items():
            init = np.array(u["initial_pos"], dtype=float)
            fp = u["flight_plan"]
            velocity = float(fp["velocity"]) 
            direction = _safe_unit(np.array(fp["direction"], dtype=float))
            u_pos = calculate_uav_position(init, velocity, direction, t)
            uav_positions[name] = u_pos

            # Process deployments
            for dep in fp.get("deployments", []):
                deploy_time = float(dep["deploy_time"]) 
                fuse_time = float(dep["fuse_time"]) 
                detonation_time = deploy_time + fuse_time
                if t >= deploy_time:
                    # Decoy ballistic trajectory starts at deploy_time
                    t_after_deploy = t - deploy_time
                    deploy_pos = calculate_uav_position(init, velocity, direction, deploy_time)
                    # Assume decoy initial velocity relative to ground equals UAV velocity forward, tossed slightly upward
                    deploy_velocity = direction * velocity + np.array([0.0, 0.0, 20.0])
                    decoy_pos = calculate_decoy_trajectory(deploy_pos, deploy_velocity, g, t_after_deploy)
                    # Add decoy point while it's airborne AND before detonation
                    if decoy_pos[2] >= 0.0 and t < detonation_time:
                        decoy_points.append(decoy_pos)

                # After fuse_time from deploy, create cloud that sinks
                if t >= detonation_time:
                    # Detonation location is the decoy position at the instant of fuse detonation
                    deploy_pos_at_fuse = calculate_uav_position(init, velocity, direction, deploy_time)
                    deploy_v_at_fuse = direction * velocity + np.array([0.0, 0.0, 20.0])
                    det_pos = calculate_decoy_trajectory(
                        deploy_pos_at_fuse, deploy_v_at_fuse, g, fuse_time
                    )
                    t_after_detonation = t - detonation_time
                    cloud_center = calculate_smoke_cloud_position(det_pos, sink_speed, t_after_detonation)
                    if cloud_center[2] >= 0.0:
                        cloud_centers.append(cloud_center)
                        cloud_radii.append(cloud_radius)

        return {
            "true_center": true_center,
            "fake_target": fake_target,
            "missile_positions": missile_positions,
            "uav_positions": uav_positions,
            "decoy_points": decoy_points,
            "cloud_centers": cloud_centers,
            "cloud_radii": cloud_radii,
            "los_segments": los_segments,
        }

    def _precompute_trajectories(self):
        """Precompute trajectories for missiles and UAVs for drawing faint trails."""
        missile_trails = {name: [] for name in self.missiles}
        uav_trails = {name: [] for name in self.uavs}
        true_target = self.targets["true_target"]
        true_center = np.array(true_target["base_center"], dtype=float) + np.array([0.0, 0.0, true_target["height"] / 2.0])
        missile_speed = float(self.constants["missile_speed"]) 

        for t in self.times:
            for name, m in self.missiles.items():
                init = np.array(m["initial_pos"], dtype=float)
                pos = calculate_missile_position(init, true_center, missile_speed, float(t))
                missile_trails[name].append(pos)
            for name, u in self.uavs.items():
                init = np.array(u["initial_pos"], dtype=float)
                fp = u["flight_plan"]
                velocity = float(fp["velocity"]) 
                direction = _safe_unit(np.array(fp["direction"], dtype=float))
                pos = calculate_uav_position(init, velocity, direction, float(t))
                uav_trails[name].append(pos)

        # Convert lists to arrays
        for key in missile_trails:
            missile_trails[key] = np.array(missile_trails[key])
        for key in uav_trails:
            uav_trails[key] = np.array(uav_trails[key])
        return missile_trails, uav_trails

    def _sphere_mesh(self, center: np.ndarray, radius: float, rings: int = 10, sectors: int = 20) -> go.Mesh3d:
        """Generate a sphere mesh for smoke cloud visualization."""
        phi = np.linspace(0, np.pi, rings)
        theta = np.linspace(0, 2 * np.pi, sectors)
        phi, theta = np.meshgrid(phi, theta)
        x = center[0] + radius * np.sin(phi) * np.cos(theta)
        y = center[1] + radius * np.sin(phi) * np.sin(theta)
        z = center[2] + radius * np.cos(phi)

        x = x.flatten()
        y = y.flatten()
        z = z.flatten()

        # Create faces
        i = []
        j = []
        k = []
        for r in range(sectors - 1):
            for s in range(rings - 1):
                p1 = r * rings + s
                p2 = p1 + 1
                p3 = p1 + rings
                p4 = p3 + 1
                i.extend([p1, p1, p2])
                j.extend([p3, p4, p3])
                k.extend([p4, p2, p4])
        return go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, opacity=0.25, color="lightgray", name="Smoke Cloud")

    def create_animation(self) -> go.Figure:
        """
        Main method to generate the Plotly animation.
        Steps:
          1) Layout and static objects
          2) Precompute and render faint trajectory lines
          3) Build frames for each time step
          4) Add slider and play/pause controls
        """
        # 1) Layout and static objects
        self.fig.update_layout(
            scene=dict(
                xaxis_title="X (m)",
                yaxis_title="Y (m)",
                zaxis_title="Z (m)",
                xaxis=dict(showgrid=True, zeroline=True),
                yaxis=dict(showgrid=True, zeroline=True),
                zaxis=dict(showgrid=True, zeroline=True),
                aspectmode="data",
            ),
            title="Missile Interception 3D Simulation",
            showlegend=True,
            margin=dict(l=0, r=0, t=40, b=0),
            height=800,
        )

        # True target as cylinder (mesh + wireframe, or wireframe-only), fake target as point
        tt = self.targets["true_target"]
        base = np.array(tt["base_center"], dtype=float)
        radius = float(tt["radius"]) 
        height = float(tt["height"]) 
        if self.use_cylinder_mesh:
            self.fig.add_trace(self._cylinder_mesh(base, radius, height, slices=self.cylinder_slices))
        # Wireframe overlay or fallback
        for tr in self._cylinder_wireframe(base, radius, height, slices=max(16, self.cylinder_slices)):
            self.fig.add_trace(tr)

        fake = np.array(self.targets["fake_target"], dtype=float)
        self.fig.add_trace(
            go.Scatter3d(x=[fake[0]], y=[fake[1]], z=[fake[2]], mode="markers", marker=dict(size=5, color="red"), name="Fake Target")
        )

        # 2) Precompute trajectories for faint trails
        missile_trails, uav_trails = self._precompute_trajectories()
        for name, trail in missile_trails.items():
            self.fig.add_trace(
                go.Scatter3d(
                    x=trail[:, 0], y=trail[:, 1], z=trail[:, 2],
                    mode="lines", line=dict(color="orange", width=2, dash="dot"),
                    name=f"{name} Trail", opacity=0.4,
                )
            )
        for name, trail in uav_trails.items():
            self.fig.add_trace(
                go.Scatter3d(
                    x=trail[:, 0], y=trail[:, 1], z=trail[:, 2],
                    mode="lines", line=dict(color="blue", width=2, dash="dot"),
                    name=f"{name} Trail", opacity=0.4,
                )
            )

        # Compute world bounds from trajectories and targets for a proper initial camera and axis ranges
        xs, ys, zs = [], [], []
        def _acc(arr):
            if arr.size:
                xs.extend(arr[:, 0].tolist())
                ys.extend(arr[:, 1].tolist())
                zs.extend(arr[:, 2].tolist())
        for trail in missile_trails.values():
            _acc(trail)
        for trail in uav_trails.values():
            _acc(trail)
        # Include targets
        tt = self.targets["true_target"]
        base = np.array(tt["base_center"], dtype=float)
        radius = float(tt["radius"]) 
        height = float(tt["height"]) 
        xs.extend([base[0] - radius, base[0] + radius])
        ys.extend([base[1] - radius, base[1] + radius])
        zs.extend([base[2], base[2] + height])
        fake = np.array(self.targets["fake_target"], dtype=float)
        xs.append(fake[0]); ys.append(fake[1]); zs.append(fake[2])

        if xs and ys and zs:
            x_min, x_max = float(np.min(xs)), float(np.max(xs))
            y_min, y_max = float(np.min(ys)), float(np.max(ys))
            z_min, z_max = float(np.min(zs)), float(np.max(zs))
            # Pad ranges
            def pad(a, b, p=0.05):
                rng = max(1.0, b - a)
                d = rng * p
                return a - d, b + d
            xr = pad(x_min, x_max, 0.1)
            yr = pad(y_min, y_max, 0.1)
            zr = pad(max(0.0, z_min), z_max, 0.1)
            self.fig.update_layout(
                scene=dict(
                    xaxis=dict(range=list(xr), showgrid=True, zeroline=True),
                    yaxis=dict(range=list(yr), showgrid=True, zeroline=True),
                    zaxis=dict(range=list(zr), showgrid=True, zeroline=True),
                    aspectmode="cube",
                )
            )

            # Ground plane at z=0 for spatial reference
            gp_x = [xr[0], xr[1], xr[1], xr[0], xr[0]]
            gp_y = [yr[0], yr[0], yr[1], yr[1], yr[0]]
            gp_z = [0, 0, 0, 0, 0]
            self.fig.add_trace(
                go.Scatter3d(x=gp_x, y=gp_y, z=gp_z, mode="lines", line=dict(color="#888", width=2), name="Ground", opacity=0.4)
            )

            # Initial camera
            center = dict(x=(xr[0]+xr[1])/2, y=(yr[0]+yr[1])/2, z=(zr[0]+zr[1])/2)
            self.fig.update_layout(
                scene_camera=dict(
                    eye=dict(x=1.6, y=1.6, z=1.2),
                    center=center,
                )
            )

        # 3) Build frames for each time step
        frames = []
        for idx, t in enumerate(self.times):
            scene = self._get_scene_data_at_time(float(t))

            # Dynamic markers: missiles and UAVs
            data_traces = []
            # Missiles
            for name, pos in scene["missile_positions"].items():
                data_traces.append(
                    go.Scatter3d(
                        x=[pos[0]], y=[pos[1]], z=[pos[2]],
                        mode="markers", marker=dict(size=6, color="orange"),
                        name=f"{name}"
                    )
                )
            # UAVs
            for name, pos in scene["uav_positions"].items():
                data_traces.append(
                    go.Scatter3d(
                        x=[pos[0]], y=[pos[1]], z=[pos[2]],
                        mode="markers", marker=dict(size=6, color="blue"),
                        name=f"{name}"
                    )
                )

            # Decoy points (in-flight)
            if scene["decoy_points"]:
                d = np.array(scene["decoy_points"])  # shape (N,3)
                data_traces.append(
                    go.Scatter3d(
                        x=d[:, 0], y=d[:, 1], z=d[:, 2],
                        mode="markers", marker=dict(size=4, color="gray"),
                        name="Decoys"
                    )
                )

            # Smoke clouds: always use sphere mesh for proper 3D ball appearance
            for c_center, c_radius in zip(scene["cloud_centers"], scene["cloud_radii"]):
                data_traces.append(self._sphere_mesh(np.array(c_center), float(c_radius),
                                                     rings=self.sphere_rings, sectors=self.sphere_sectors))

            # Missile Line of Sight segments
            for name, (m_pos, tgt) in scene["los_segments"].items():
                data_traces.append(
                    go.Scatter3d(
                        x=[m_pos[0], tgt[0]], y=[m_pos[1], tgt[1]], z=[m_pos[2], tgt[2]],
                        mode="lines", line=dict(color="orange", width=2),
                        name=f"{name} LOS"
                    )
                )

            frames.append(go.Frame(data=data_traces, name=f"t={t:.1f}"))

        self.fig.frames = frames

        # Add first frame's dynamic data so something is visible immediately
        if frames:
            for tr in frames[0].data:
                self.fig.add_trace(tr)

        # 4) Slider and Play/Pause controls
        steps = []
        for i, t in enumerate(self.times):
            step = dict(
                method="animate",
                args=[[f"t={t:.1f}"], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}],
                label=f"{t:.1f}",
            )
            steps.append(step)

        sliders = [
            dict(
                active=0,
                currentvalue={"prefix": "t = ", "suffix": " s", "visible": True},
                pad={"t": 50},
                steps=steps,
            )
        ]

        self.fig.update_layout(
            sliders=sliders,
            updatemenus=[
                {
                    "type": "buttons",
                    "showactive": False,
                    "y": 0,
                    "x": 0.1,
                    "xanchor": "right",
                    "yanchor": "top",
                    "pad": {"t": 50, "r": 10},
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    "frame": {"duration": int(self.dt * 1000), "redraw": True},
                                    "fromcurrent": True,
                                    "transition": {"duration": 0},
                                },
                            ],
                        },
                        {
                            "label": "Pause",
                            "method": "animate",
                            "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}],
                        },
                    ],
                }
            ],
        )

        return self.fig


def _safe_unit(v: np.ndarray) -> np.ndarray:
    """Unit vector with zero-vector guard."""
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


