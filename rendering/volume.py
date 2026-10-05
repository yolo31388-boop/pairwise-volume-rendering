"""
游戏内体积渲染与体积雾系统 - 核心模块

实现内容：
- render_volume():  基于光线步进(ray marching)的体积渲染，含散射
- volume_fog():     体积雾，含高度衰减密度场与 3D 噪声
- volumetric_light(): 体积光，含相函数散射与体积阴影
- volume_clouds():  体积云，含噪声密度场与光照（上亮下暗）
- volume_shadow():  体积阴影，沿光线步进累计光学深度，天然软阴影
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import math
import time

Vec3 = Tuple[float, float, float]


@dataclass
class Config:
    pass


DEFAULTS: Dict = {
    "fog_density": 0.05,        # 雾基础密度
    "fog_height": 0.0,          # 雾基准高度
    "fog_falloff": 0.1,         # 雾高度衰减系数
    "noise_scale": 0.3,         # 雾噪声空间频率
    "noise_strength": 0.5,      # 噪声对密度的调制强度 [0,1]
    "extinction_coeff": 1.0,    # 消光系数
    "scatter_coeff": 0.8,       # 散射系数
    "anisotropy": 0.6,          # HG 相函数各向异性 g
    "ambient": 0.1,             # 环境光
    "light_direction": (0.0, -1.0, 0.0),  # 光线传播方向（向下）
    "light_color": (1.0, 1.0, 1.0),
    "cloud_base": 50.0,         # 云层底部
    "cloud_top": 80.0,          # 云层顶部
    "cloud_coverage": 0.5,      # 云覆盖率 [0,1]
    "cloud_density": 0.4,       # 云密度
    "cloud_noise_scale": 0.08,  # 云噪声频率
    "steps": 64,                # 主光线步进数
    "shadow_steps": 48,         # 阴影光线步进数
    "shadow_max_dist": 200.0,   # 阴影步进最大距离
}


def _v_add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _v_scale(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def _v_dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _v_normalize(a: Vec3) -> Vec3:
    length = math.sqrt(_v_dot(a, a))
    if length < 1e-12:
        return (0.0, 0.0, 0.0)
    return (a[0] / length, a[1] / length, a[2] / length)


def _fade(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


class VolumeRendering:
    def __init__(self, config: Optional[Dict] = None):
        self.config = dict(DEFAULTS)
        if config:
            self.config.update(config)
        self._state = {}
        self._history = []
        self._time = 0.0

    # ------------------------------------------------------------------
    # 噪声：确定性 3D 值噪声 + 分形叠加(fBm)
    # ------------------------------------------------------------------
    @staticmethod
    def _hash3(ix: int, iy: int, iz: int) -> float:
        n = (ix * 374761393 + iy * 668265263 + iz * 1440662683) & 0xFFFFFFFF
        n = ((n ^ (n >> 13)) * 1274126177) & 0xFFFFFFFF
        n = n ^ (n >> 16)
        return n / 0xFFFFFFFF

    def _value_noise(self, x: float, y: float, z: float) -> float:
        x0, y0, z0 = math.floor(x), math.floor(y), math.floor(z)
        tx, ty, tz = _fade(x - x0), _fade(y - y0), _fade(z - z0)

        def corner(dx, dy, dz):
            return self._hash3(x0 + dx, y0 + dy, z0 + dz)

        c00 = corner(0, 0, 0) + tx * (corner(1, 0, 0) - corner(0, 0, 0))
        c10 = corner(0, 1, 0) + tx * (corner(1, 1, 0) - corner(0, 1, 0))
        c01 = corner(0, 0, 1) + tx * (corner(1, 0, 1) - corner(0, 0, 1))
        c11 = corner(0, 1, 1) + tx * (corner(1, 1, 1) - corner(0, 1, 1))
        c0 = c00 + ty * (c10 - c00)
        c1 = c01 + ty * (c11 - c01)
        return c0 + tz * (c1 - c0)

    def _fbm(self, x: float, y: float, z: float, octaves: int = 4) -> float:
        value, amplitude, frequency, total = 0.0, 0.5, 1.0, 0.0
        for _ in range(octaves):
            value += amplitude * self._value_noise(
                x * frequency, y * frequency, z * frequency)
            total += amplitude
            amplitude *= 0.5
            frequency *= 2.0
        return value / total

    # ------------------------------------------------------------------
    # 密度场
    # ------------------------------------------------------------------
    def density_at(self, point: Vec3) -> float:
        """雾密度：基础密度 × 高度衰减 × 噪声调制（随时间流动）。"""
        x, y, z = point
        height = max(0.0, y - self.config["fog_height"])
        height_term = math.exp(-self.config["fog_falloff"] * height)
        scale = self.config["noise_scale"]
        noise = self._fbm(x * scale + self._time, y * scale, z * scale)
        strength = self.config["noise_strength"]
        modulation = (1.0 - strength) + strength * 2.0 * noise
        return max(0.0, self.config["fog_density"] * height_term * modulation)

    def cloud_density_at(self, point: Vec3) -> float:
        """云密度场：层内噪声超过覆盖率阈值才有云，边缘渐变。"""
        x, y, z = point
        base, top = self.config["cloud_base"], self.config["cloud_top"]
        if y < base or y > top:
            return 0.0
        thickness = max(1e-6, top - base)
        frac = (y - base) / thickness
        gradient = min(1.0, frac * 4.0) * min(1.0, (1.0 - frac) * 4.0)
        scale = self.config["cloud_noise_scale"]
        noise = self._fbm(x * scale + self._time * 0.5, y * scale, z * scale)
        coverage = min(1.0, max(0.0, self.config["cloud_coverage"]))
        density = (noise - (1.0 - coverage)) / max(1e-6, coverage)
        return max(0.0, density) * self.config["cloud_density"] * gradient

    # ------------------------------------------------------------------
    # 光照
    # ------------------------------------------------------------------
    def _toward_light(self) -> Vec3:
        d = self.config["light_direction"]
        return _v_normalize((-d[0], -d[1], -d[2]))

    def phase_function(self, cos_theta: float, g: Optional[float] = None) -> float:
        """Henyey-Greenstein 相函数：描述光在体积中的散射方向分布。"""
        if g is None:
            g = self.config["anisotropy"]
        denom = (1.0 + g * g - 2.0 * g * cos_theta) ** 1.5
        return (1.0 - g * g) / (4.0 * math.pi * max(1e-12, denom))

    def _optical_depth_to_light(
        self,
        point: Vec3,
        light_dir: Optional[Vec3] = None,
        max_dist: Optional[float] = None,
        steps: Optional[int] = None,
        density_fn: Optional[Callable[[Vec3], float]] = None,
    ) -> Tuple[float, int]:
        """从点向光源方向步进，累计光学深度（体积阴影的核心）。"""
        direction = _v_normalize(light_dir) if light_dir else self._toward_light()
        max_dist = max_dist or self.config["shadow_max_dist"]
        steps = steps or self.config["shadow_steps"]
        density_fn = density_fn or (
            lambda p: self.density_at(p) + self.cloud_density_at(p))
        step = max_dist / steps
        optical_depth = 0.0
        for i in range(steps):
            p = _v_add(point, _v_scale(direction, (i + 0.5) * step))
            optical_depth += density_fn(p) * self.config["extinction_coeff"] * step
        return optical_depth, steps

    def volume_shadow(
        self,
        point: Vec3,
        light_dir: Optional[Vec3] = None,
    ) -> Dict:
        """体积阴影：沿光线步进累计光学深度，exp(-τ) 给出连续软阴影。"""
        optical_depth, steps = self._optical_depth_to_light(point, light_dir)
        return {
            "shadow": math.exp(-optical_depth),
            "optical_depth": optical_depth,
            "steps": steps,
        }

    def volumetric_light(self, point: Vec3, view_dir: Optional[Vec3] = None) -> Dict:
        """体积光：密度 × 相函数散射 × 体积阴影，可形成光束效果。"""
        shadow = self.volume_shadow(point)["shadow"]
        density = self.density_at(point) + self.cloud_density_at(point)
        if view_dir is not None:
            cos_theta = _v_dot(_v_normalize(view_dir), self._toward_light())
        else:
            cos_theta = 1.0
        phase = self.phase_function(cos_theta)
        scatter = self.config["scatter_coeff"]
        ambient = self.config["ambient"]
        light_color = self.config["light_color"]
        brightness = density * scatter * (ambient + phase * shadow)
        return {
            "intensity": tuple(brightness * c for c in light_color),
            "shadow": shadow,
            "density": density,
            "phase": phase,
        }

    # ------------------------------------------------------------------
    # 光线步进体积渲染
    # ------------------------------------------------------------------
    def ray_march(
        self,
        origin: Vec3,
        direction: Vec3,
        max_dist: float,
        steps: Optional[int] = None,
        density_fn: Optional[Callable[[Vec3], float]] = None,
        lit: bool = True,
    ) -> Dict:
        """沿视线步进：累计体积散射与透射率（Beer-Lambert）。"""
        steps = steps or self.config["steps"]
        density_fn = density_fn or (
            lambda p: self.density_at(p) + self.cloud_density_at(p))
        direction = _v_normalize(direction)
        step = max_dist / steps
        extinction = self.config["extinction_coeff"]
        scatter = self.config["scatter_coeff"]
        ambient = self.config["ambient"]
        light_color = self.config["light_color"]
        phase = self.phase_function(_v_dot(direction, self._toward_light()))

        transmittance = 1.0
        color = [0.0, 0.0, 0.0]
        for i in range(steps):
            p = _v_add(origin, _v_scale(direction, (i + 0.5) * step))
            density = density_fn(p)
            if density <= 0.0:
                continue
            if lit:
                shadow = math.exp(-self._optical_depth_to_light(p)[0])
                inscatter = density * scatter * (ambient + phase * shadow)
            else:
                inscatter = density * scatter * ambient
            for c in range(3):
                color[c] += transmittance * inscatter * light_color[c] * step
            transmittance *= math.exp(-density * extinction * step)
            if transmittance < 1e-4:
                break
        return {
            "color": tuple(color),
            "transmittance": transmittance,
            "optical_depth": -math.log(max(transmittance, 1e-12)),
            "steps": steps,
        }

    def render_volume(
        self,
        origin: Vec3,
        direction: Vec3,
        max_dist: float = 100.0,
        steps: Optional[int] = None,
    ) -> Dict:
        """体积渲染主入口：光线步进 + 散射，雾与云共同参与。"""
        result = self.ray_march(origin, direction, max_dist, steps)
        result["ray_marching"] = True
        return result

    def volume_fog(self, point: Vec3) -> float:
        """体积雾：返回指定位置的雾密度（含高度衰减与噪声）。"""
        return self.density_at(point)

    def volume_clouds(
        self,
        origin: Vec3,
        direction: Vec3,
        max_dist: float = 200.0,
        steps: Optional[int] = None,
    ) -> Dict:
        """体积云：在云密度场中步进渲染，含光照。"""
        return self.ray_march(
            origin, direction, max_dist, steps,
            density_fn=self.cloud_density_at)

    def cloud_lighting(self, point: Vec3) -> float:
        """云的光照：向光源穿过云层的透射率，云顶亮、云底暗。"""
        optical_depth, _ = self._optical_depth_to_light(
            point, density_fn=self.cloud_density_at)
        return math.exp(-optical_depth)

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def update(self, dt: float):
        self._time += dt
        self._history.append({"dt": dt, "time": self._time})

    def reset(self):
        self._state = {}
        self._history = []
        self._time = 0.0
